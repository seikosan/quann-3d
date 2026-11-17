"""Export a saved maintenance cloud classifier to ONNX and CPU INT8."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from train_maintenance import MaintenanceCloud


DEFAULT_CHECKPOINT = ROOT / "maintenance" / "results" / "quann_binary_backprop.pt"
DEFAULT_OUT_DIR = ROOT / "maintenance" / "results" / "edge"
LATENCY_BATCHES = (1, 16, 64)
PARITY_TOLERANCE = 1e-5


class EdgeClassifier(nn.Module):
    """Turn (batch, 10, 36) already-scaled point clouds into class probabilities."""

    def __init__(self, model: MaintenanceCloud):
        super().__init__()
        self.model = model

    def forward(self, cloud: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(cloud), dim=-1)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path) -> tuple[EdgeClassifier, dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"unsupported maintenance checkpoint: {checkpoint_path}")
    model_config = checkpoint.get("config", {})
    dimension = int(model_config.get("D", 36))
    n_classes = int(model_config.get("n_classes", 2))
    model = MaintenanceCloud(vocab_size=max(1, len(checkpoint.get("token_weights", {}))),
                            n_classes=n_classes, D=dimension)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return EdgeClassifier(model).eval(), checkpoint


def _representative_clouds(count: int, dimension: int, seed: int = 20261022) -> np.ndarray:
    """Deterministic, normalized cloud-shaped inputs for parity and CPU timing."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(count, 10, dimension)).astype(np.float32)


def _new_ort_session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def _measure_latency(session, inputs: np.ndarray, warmup: int, repeats: int) -> dict:
    feed = {session.get_inputs()[0].name: inputs}
    for _ in range(warmup):
        session.run(None, feed)
    samples_ms = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run(None, feed)
        samples_ms.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": round(statistics.median(samples_ms), 4),
        "p95_ms": round(float(np.percentile(samples_ms, 95)), 4),
        "repeats": repeats,
    }


def export(checkpoint_path: Path, out_dir: Path, warmup: int = 10,
           repeats: int = 80) -> dict:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    checkpoint_path = checkpoint_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261022)

    classifier, checkpoint = load_model(checkpoint_path)
    dimension = classifier.model.D
    n_classes = classifier.model.class_head[-1].out_features
    cloud_shape = (2, 10, dimension)
    sample = torch.from_numpy(_representative_clouds(cloud_shape[0], dimension))
    fp32_path = out_dir / "model.onnx"
    int8_path = out_dir / "model.int8.onnx"

    torch.onnx.export(
        classifier,
        (sample,),
        str(fp32_path),
        input_names=["cloud"],
        output_names=["probabilities"],
        dynamic_axes={"cloud": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(str(fp32_path))
    onnx.checker.check_model(graph)

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=["MatMul", "Gemm"])
    fp32_session = _new_ort_session(fp32_path)
    int8_session = _new_ort_session(int8_path)

    parity_by_batch = {}
    fp32_latency = {}
    int8_latency = {}
    int8_parity_by_batch = {}
    for batch_size in LATENCY_BATCHES:
        cloud = _representative_clouds(batch_size, dimension)
        with torch.inference_mode():
            torch_output = classifier(torch.from_numpy(cloud)).cpu().numpy()
        ort_output = fp32_session.run(None, {"cloud": cloud})[0]
        int8_output = int8_session.run(None, {"cloud": cloud})[0]
        fp32_diff = np.abs(torch_output - ort_output)
        int8_diff = np.abs(torch_output - int8_output)
        parity_by_batch[str(batch_size)] = float(fp32_diff.max())
        int8_parity_by_batch[str(batch_size)] = float(int8_diff.max())
        fp32_latency[str(batch_size)] = _measure_latency(fp32_session, cloud, warmup, repeats)
        int8_latency[str(batch_size)] = _measure_latency(int8_session, cloud, warmup, repeats)

    max_parity = max(parity_by_batch.values())
    if max_parity > PARITY_TOLERANCE:
        raise RuntimeError(f"torch/ONNX parity {max_parity:.3e} exceeds {PARITY_TOLERANCE:.1e}")

    point_table = checkpoint.get("token_weights", {})
    point_table_path = None
    if point_table:
        point_table_path = out_dir / "point_table.json"
        payload = {str(token): np.asarray(weight.value, dtype=np.float32).tolist()
                   for token, weight in sorted(point_table.items())}
        point_table_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    labels = ["rul_gt_30", "rul_le_30"] if n_classes == 2 else [f"class_{i}" for i in range(n_classes)]
    artifacts = {}
    for name, path in (("model_onnx", fp32_path), ("model_int8_onnx", int8_path)):
        artifacts[name] = {"path": path.name, "size_bytes": path.stat().st_size,
                           "sha256": sha256_of(path)}
    if point_table_path is not None:
        artifacts["point_table"] = {"path": point_table_path.name,
                                    "size_bytes": point_table_path.stat().st_size,
                                    "sha256": sha256_of(point_table_path)}

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_of(checkpoint_path),
            "selection": "best saved binary C-MAPSS checkpoint by held-out PR-AUC/F2",
            "selected": checkpoint_path.name,
        },
        "model": {"D": dimension, "window_points": 10, "n_classes": n_classes,
                  "class_labels": labels, "regime": checkpoint_path.stem.removeprefix("quann_binary_")},
        "input": {
            "name": "cloud", "dtype": "float32", "shape": ["batch", 10, dimension],
            "description": "10-step normalized sensor-feature point cloud; input is already scaled",
            "preprocessing_note": "The training scaler is not included in this checkpoint; raw sensor preprocessing is upstream.",
        },
        "output": {"name": "probabilities", "dtype": "float32",
                   "shape": ["batch", n_classes], "activation": "softmax"},
        "artifacts": artifacts,
        "parity": {
            "torch_vs_onnx_max_abs_by_batch": parity_by_batch,
            "torch_vs_onnx_max_abs": max_parity,
            "tolerance": PARITY_TOLERANCE,
            "passed": max_parity <= PARITY_TOLERANCE,
            "torch_vs_int8_onnx_max_abs_by_batch": int8_parity_by_batch,
        },
        "latency_cpu_onnxruntime_ms": {"fp32": fp32_latency, "int8": int8_latency},
        "benchmark": {"provider": "CPUExecutionProvider", "ort_version": ort.__version__,
                      "torch_version": torch.__version__, "platform": platform.platform(),
                      "warmup": warmup, "repeats": repeats,
                      "latency_statistic": "median and p95 per session.run call"},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=80)
    args = parser.parse_args(argv)
    manifest = export(args.checkpoint, args.out_dir, args.warmup, args.repeats)
    print(json.dumps({
        "checkpoint": manifest["checkpoint"]["selected"],
        "fp32_size_bytes": manifest["artifacts"]["model_onnx"]["size_bytes"],
        "int8_size_bytes": manifest["artifacts"]["model_int8_onnx"]["size_bytes"],
        "parity_max_abs": manifest["parity"]["torch_vs_onnx_max_abs"],
        "fp32_latency_ms": {batch: item["median_ms"] for batch, item in
                            manifest["latency_cpu_onnxruntime_ms"]["fp32"].items()},
        "int8_latency_ms": {batch: item["median_ms"] for batch, item in
                            manifest["latency_cpu_onnxruntime_ms"]["int8"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
