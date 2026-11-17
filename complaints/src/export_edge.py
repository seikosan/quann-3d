"""Export the trained complaint reader to ONNX and measure parity/latency."""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from config import RESULTS_DIR
from explain_region import load_model


class ZOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, mask):
        return self.model(tokens, mask)[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(checkpoint_path=None, tokenizer_path=None, output_dir=None,
                  max_len=None, repeats=100):
    checkpoint_path = checkpoint_path or os.path.join(RESULTS_DIR, "reader.pt")
    tokenizer_path = tokenizer_path or os.path.join(RESULTS_DIR, "tokenizer.json")
    output_dir = output_dir or os.path.join(RESULTS_DIR, "edge")
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, checkpoint = load_model(checkpoint_path, tokenizer_path, device="cpu")
    max_len = max_len or checkpoint["config"]["max_len"]
    wrapper = ZOnly(model).eval()
    ids = tokenizer.encode("A sample complaint for edge parity.")[:max_len] or [2]
    tokens = torch.zeros((1, max_len), dtype=torch.long)
    tokens[0, :len(ids)] = torch.tensor(ids)
    mask = tokens.ne(0)
    onnx_path = os.path.join(output_dir, "reader.onnx")
    torch.onnx.export(
        wrapper, (tokens, mask), onnx_path, input_names=["tokens", "mask"],
        output_names=["z"], dynamic_axes={"tokens": {0: "batch"},
                                          "mask": {0: "batch"}, "z": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    exported_tokenizer = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(exported_tokenizer)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_feed = {"tokens": tokens.numpy(), "mask": mask.numpy()}
    with torch.no_grad():
        torch_z = wrapper(tokens, mask).numpy()
    onnx_z = session.run(["z"], input_feed)[0]
    max_abs_delta = float(np.max(np.abs(torch_z - onnx_z)))
    for _ in range(10):
        session.run(["z"], input_feed)
    started = time.perf_counter()
    for _ in range(repeats):
        session.run(["z"], input_feed)
    latency_ms = (time.perf_counter() - started) * 1000.0 / repeats
    manifest = {
        "format": "ONNX", "opset": 17,
        "files": {"model": "reader.onnx", "tokenizer": "tokenizer.json"},
        "sha256": {"reader.onnx": sha256_file(onnx_path),
                   "tokenizer.json": sha256_file(exported_tokenizer)},
        "config": checkpoint["config"],
        "model_config": checkpoint["model_config"],
        "self_test": {"input": "A sample complaint for edge parity.",
                       "max_abs_delta_z": max_abs_delta,
                       "passed": bool(max_abs_delta < 1e-4)},
        "batch1_cpu_latency_ms": latency_ms,
        "latency_repeats": repeats,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    if not manifest["self_test"]["passed"]:
        raise AssertionError(f"Torch/ONNX parity failed: {max_abs_delta}")
    return manifest, os.path.getsize(onnx_path) + os.path.getsize(exported_tokenizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    manifest, size = export_bundle(args.checkpoint, args.tokenizer,
                                   args.output_dir, repeats=args.repeats)
    print(json.dumps({"size_kb": round(size / 1024, 2),
                      "parity_max_abs_delta_z": manifest["self_test"]["max_abs_delta_z"],
                      "batch1_cpu_latency_ms": manifest["batch1_cpu_latency_ms"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
