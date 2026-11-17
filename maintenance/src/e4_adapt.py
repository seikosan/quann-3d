"""E4: local 0th-order CWRU operating-condition adaptation, CPU-only."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from loader_cwru_exp import FEATURE_DIM, load_expanded_records
from train_maintenance import MaintenanceCloud


SEED = 20261016
DEFAULT_UPDATE_BUDGET = 24


def _log(path: Path, message: str) -> None:
    line = message if message.startswith("=== ") else f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _stack(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([r["features"] for r in records]).astype(np.float32)
    y = np.concatenate([np.full(r["segments"], int(r["fault_type"] != "normal"), dtype=np.int64)
                        for r in records])
    return x, y


def _predict(model: MaintenanceCloud, tokens: np.ndarray, n: int | None = None) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tokens), 128):
            batch = torch.as_tensor(tokens[start:start + 128], dtype=torch.long)
            outputs.append(torch.softmax(model(model.points_for(batch)), dim=1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, n or 2), dtype=np.float32)


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    return {"n": int(len(y)), "auc_roc": float(roc_auc_score(y, probabilities[:, 1])),
            "accuracy": float(accuracy_score(y, predictions)),
            "class_recall": {"normal": float(np.mean(predictions[y == 0] == 0)) if np.any(y == 0) else None,
                             "fault": float(np.mean(predictions[y == 1] == 1)) if np.any(y == 1) else None}}


def _sample_balanced(x: np.ndarray, y: np.ndarray, tokens: np.ndarray,
                     budget: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"B adaptation stream needs both binary classes, found {classes.tolist()}")
    per_class = max(1, budget // len(classes))
    selected = []
    for class_id in classes:
        candidates = np.flatnonzero(y == class_id)
        selected.extend(rng.choice(candidates, min(len(candidates), per_class), replace=False).tolist())
    # Fill any odd remaining budget from unused candidates, without oversampling.
    if len(selected) < budget:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(selected, dtype=int))
        if len(remaining):
            selected.extend(rng.choice(remaining, min(budget - len(selected), len(remaining)), replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def zero_order_adapt(model: MaintenanceCloud, tokens: np.ndarray, labels: np.ndarray,
                     update_budget: int = DEFAULT_UPDATE_BUDGET, on_step=None) -> dict:
    """Update QUANN point intervals only; never constructs a gradient or optimizer."""
    before = model.convergence_info()
    indices = _sample_balanced(np.empty((len(labels), 0)), labels, tokens, update_budget)
    model.eval()
    for update_number, index in enumerate(indices, 1):
        token = torch.as_tensor(tokens[index:index + 1], dtype=torch.long)
        model.quann_update(token, int(labels[index]))
        if on_step is not None:
            on_step(update_number, model)
    after = model.convergence_info()
    return {"requested_update_budget": update_budget, "actual_updates": int(len(indices)),
            "points_before": before["points"], "points_after": after["points"],
            "points_delta": after["points"] - before["points"],
            "converged_before": before["converged"], "converged_after": after["converged"],
            "converged_delta": after["converged"] - before["converged"],
            "mean_width_before": before["mean_width"], "mean_width_after": after["mean_width"],
            "gradient_calls": 0, "backprop_calls": 0}


def _train_condition_a(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> tuple[MaintenanceCloud, MiniBatchKMeans, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-7] = 1.0
    scaled = ((x - mean) / scale).astype(np.float32)
    clusterer = MiniBatchKMeans(n_clusters=min(12, max(2, len(scaled) // 20)),
                                random_state=seed, n_init=3,
                                batch_size=min(256, max(32, len(scaled))))
    clusterer.fit(scaled)
    tokens = clusterer.predict(scaled).reshape(-1, 1)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = MaintenanceCloud(vocab_size=int(tokens.max()) + 1, n_classes=2, D=FEATURE_DIM)
    optimizer = torch.optim.AdamW(list(model.cloud_encoder.parameters()) + list(model.class_head.parameters()),
                                  lr=.002, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in (0, 1):
        candidates = np.flatnonzero(y == class_id)
        selected.extend(rng.choice(candidates, min(len(candidates), 70), replace=False).tolist())
    rng.shuffle(selected)
    for index in selected:
        token = torch.as_tensor(tokens[index:index + 1], dtype=torch.long)
        point = model.points_for(token).detach()
        target = torch.tensor([int(y[index])], dtype=torch.long)
        model.train()
        loss = torch.nn.functional.cross_entropy(model(point), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward(); optimizer.step()
        model.eval(); model.quann_update(token, int(y[index]))
    return model, clusterer, mean, scale


def _backprop_finetune(model: MaintenanceCloud, tokens: np.ndarray, labels: np.ndarray,
                       indices: np.ndarray, on_step=None) -> dict:
    parameters = list(model.cloud_encoder.parameters()) + list(model.class_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=1e-4)
    model.train()
    for update_number, index in enumerate(indices, 1):
        token = torch.as_tensor(tokens[index:index + 1], dtype=torch.long)
        points = model.points_for(token).detach()
        target = torch.tensor([int(labels[index])], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(points), target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if on_step is not None:
            on_step(update_number, model)
    model.eval()
    return {"actual_updates": int(len(indices)), "gradient_calls": int(len(indices)),
            "backprop_calls": int(len(indices)), "updated_parameters": "CloudEncoderND + classifier reader"}


def run_e4(data_dir: str | Path | None = None, out_path: str | Path | None = None,
           figure_path: str | Path | None = None, log_path: str | Path | None = None,
           update_budget: int = DEFAULT_UPDATE_BUDGET) -> dict:
    project = ROOT / "maintenance"
    data_dir = Path(data_dir or project / "data" / "cwru" / "expanded")
    out_path = Path(out_path or project / "results" / "e4_adapt.json")
    figure_path = Path(figure_path or project / "figures" / "e4_adapt_curve.png")
    log_path = Path(log_path or project / "logs" / "run_e4_smoke.log")
    for directory in (out_path.parent, figure_path.parent, log_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    _log(log_path, f"=== [{datetime.now().date()}] 1/4: load local CWRU recordings and identify RPM conditions ===")
    records = load_expanded_records(data_dir, lambda message: _log(log_path, message))
    condition_a = [r for r in records if r["load_hp"] in (0, 1, 2)]
    condition_b = [r for r in records if r["load_hp"] == 3]
    if not condition_a or not condition_b:
        raise ValueError(f"need load 0-2 and load 3 recordings, found A={len(condition_a)} B={len(condition_b)}")
    train_x, train_y = _stack(condition_a)
    _log(log_path, f"condition A recordings={len(condition_a)}, rpm={sorted({r['rpm'] for r in condition_a})}; B recordings={len(condition_b)}, rpm={sorted({r['rpm'] for r in condition_b})}")

    _log(log_path, "=== E4 2/4: train hybrid on condition A; reserve the later half of each B recording ===")
    base, clusterer, mean, scale = _train_condition_a(train_x, train_y)
    b_support_x, b_support_y, b_test_x, b_test_y = [], [], [], []
    support_records, test_records = [], []
    for record in condition_b:
        x = ((record["features"] - mean) / scale).astype(np.float32)
        y = np.full(record["segments"], int(record["fault_type"] != "normal"), dtype=np.int64)
        cut = max(2, len(x) // 2)
        cut = min(cut, len(x) - 1)
        b_support_x.append(x[:cut]); b_support_y.append(y[:cut])
        b_test_x.append(x[cut:]); b_test_y.append(y[cut:])
        support_records.append(record["recording_id"]); test_records.append(record["recording_id"])
    support_x = np.concatenate(b_support_x)
    support_y = np.concatenate(b_support_y)
    test_x = np.concatenate(b_test_x)
    test_y = np.concatenate(b_test_y)
    support_tokens = clusterer.predict(support_x).reshape(-1, 1)
    test_tokens = clusterer.predict(test_x).reshape(-1, 1)
    if len(np.unique(test_y)) != 2 or len(np.unique(support_y)) != 2:
        raise ValueError("condition-B support and test stream must each contain normal and fault segments")

    _log(log_path, "=== E4 3/4: compare frozen, 0th-order point adaptation, and backprop reader fine-tune ===")
    # Materialize every train-fitted vocabulary point before any arm is measured.
    # Otherwise an unseen test token would be lazily initialized during prediction
    # and look like an adaptation-induced point-count increase.
    all_b_tokens = torch.as_tensor(np.concatenate([support_tokens, test_tokens]), dtype=torch.long)
    base.points_for(all_b_tokens)
    frozen = copy.deepcopy(base)
    zero_model = copy.deepcopy(base)
    bp_model = copy.deepcopy(base)
    frozen_prob = _predict(frozen, test_tokens)
    frozen_metrics = _metrics(test_y, frozen_prob)
    update_ids = _sample_balanced(support_x, support_y, support_tokens, update_budget)

    checkpoint_every = max(1, len(update_ids) // 6)
    zero_curve = [(0, frozen_metrics["auc_roc"])]
    before_points = zero_model.convergence_info()
    # Use the same fixed, class-balanced local label-feedback stream for both adaptation caps.

    def collect_zero(step, model):
        if step % checkpoint_every == 0 or step == len(update_ids):
            metric = _metrics(test_y, _predict(model, test_tokens))
            zero_curve.append((step, metric["auc_roc"]))

    zero_detail = zero_order_adapt(zero_model, support_tokens[update_ids], support_y[update_ids],
                                   update_budget=len(update_ids), on_step=collect_zero)
    zero_prob = _predict(zero_model, test_tokens)
    zero_metrics = _metrics(test_y, zero_prob)

    bp_curve = [(0, frozen_metrics["auc_roc"])]

    def collect_bp(step, model):
        if step % checkpoint_every == 0 or step == len(update_ids):
            bp_curve.append((step, _metrics(test_y, _predict(model, test_tokens))["auc_roc"]))

    bp_detail = _backprop_finetune(bp_model, support_tokens, support_y, update_ids, on_step=collect_bp)
    bp_prob = _predict(bp_model, test_tokens)
    bp_metrics = _metrics(test_y, bp_prob)
    after_bp_points = bp_model.convergence_info()
    bp_detail.update({"points_before": before_points["points"], "points_after": after_bp_points["points"],
                      "points_delta": after_bp_points["points"] - before_points["points"],
                      "converged_before": before_points["converged"], "converged_after": after_bp_points["converged"],
                      "converged_delta": after_bp_points["converged"] - before_points["converged"]})

    frozen_points = base.convergence_info()
    zero_steps_meta = {"metrics": zero_metrics, **zero_detail}
    arms = {
        "frozen": {"metrics": frozen_metrics, "points": frozen_points,
                   "points_delta": 0, "update_steps": 0},
        "zero_order_adapt": zero_steps_meta,
        "backprop_finetune": {"metrics": bp_metrics, **bp_detail},
    }
    gradient_gain = bp_metrics["auc_roc"] - frozen_metrics["auc_roc"]
    zero_gain = zero_metrics["auc_roc"] - frozen_metrics["auc_roc"]
    captured = (100.0 * zero_gain / gradient_gain) if gradient_gain > 1e-12 else None
    _log(log_path, f"E4 AUC frozen={frozen_metrics['auc_roc']:.4f}, zero_order={zero_metrics['auc_roc']:.4f}, backprop={bp_metrics['auc_roc']:.4f}; gradient_gain_capture_pct={captured}")

    _log(log_path, "=== E4 4/4: save condition-B adaptation curve and result JSON ===")
    fig, ax = plt.subplots(figsize=(9.5, 6), constrained_layout=True)
    steps = sorted(set((0, len(update_ids), *[step for step, _ in zero_curve[1:]],
                        *[step for step, _ in bp_curve[1:]])))
    ax.plot(steps, [frozen_metrics["auc_roc"]] * len(steps), marker="o", color="#3978a8", label="frozen hybrid")
    ax.plot([step for step, _ in zero_curve], [value for _, value in zero_curve],
            marker="o", color="#32936f", label="0th-order QUANN point adaptation")
    ax.plot([step for step, _ in bp_curve], [value for _, value in bp_curve],
            marker="o", color="#c83e4d", label="backprop reader fine-tune")
    ax.set_title("CWRU condition-B ROC-AUC by local update budget")
    ax.set_xlabel("condition-B labeled local updates")
    ax.set_ylabel("ROC-AUC on later B segments")
    ax.set_ylim(0, 1.02); ax.grid(alpha=.24); ax.legend(frameon=False)
    fig.savefig(figure_path, dpi=165, facecolor="white"); plt.close(fig)

    rpm_by_load = {}
    for row in condition_a + condition_b:
        rpm_by_load.setdefault(str(row["load_hp"]), set()).add(float(row["rpm"]))
    result = {
        "experiment": "E4 local QUANN operating-condition adaptation demonstration",
        "condition_proxy": {"kind": "measured_or_catalog_nominal_rpm", "source": "CWRU .mat RPM field when present; catalog nominal RPM otherwise (fallback logged)",
                            "condition_A": {"training_load_hp": [0, 1, 2], "rpm_values": sorted({v for k, values in rpm_by_load.items() if int(k) in (0, 1, 2) for v in values})},
                            "condition_B": {"held_load_hp": 3, "rpm_values": sorted(rpm_by_load.get("3", set()))}},
        "protocol": {"task": "binary normal vs any fault", "training": "QUANN points + backprop reader on condition A",
                     "B_stream": "first half of each load-3 recording, fixed balanced labeled support budget",
                     "B_evaluation": "later half of each load-3 recording; same source files but later segments, not recording-disjoint",
                     "local_feedback": "Condition-B labels are assumed available from local inspection/maintenance feedback; labels and raw data remain local in this simulation.",
                     "zero_order_constraint": "quann_update only; reader parameters frozen; zero gradient/backprop calls",
                     "vocabulary_point_initialization": "train-fitted tokenizer vocabulary materialized before all arms; no condition-B labels used for initialization",
                     "backprop_ceiling": "same support examples, AdamW updates reader only; QUANN point locations frozen",
                     "update_budget": update_budget, "device": "cpu", "seed": SEED},
        "condition_B_recordings": len(condition_b),
        "condition_B_support_segments": int(len(support_y)), "condition_B_test_segments": int(len(test_y)),
        "condition_B_support_recording_ids": sorted(support_records),
        "condition_B_test_recording_ids": sorted(test_records),
        "arms": arms,
        "auc_by_update_step": {"frozen": [[step, frozen_metrics["auc_roc"]] for step in steps],
                               "zero_order_adapt": zero_curve, "backprop_finetune": bp_curve},
        "gradient_gain_capture": {"frozen_auc": frozen_metrics["auc_roc"], "zero_order_auc": zero_metrics["auc_roc"],
                                  "backprop_auc": bp_metrics["auc_roc"], "zero_order_gain": zero_gain,
                                  "backprop_gain": gradient_gain, "zero_order_fraction_of_backprop_gain_percent": captured,
                                  "interpretation": (f"0th-order gain is {captured:.1f}% of the positive backprop AUC gain." if captured is not None
                                                     else "Backprop did not improve AUC over frozen; a gradient-gain capture percentage is undefined.")},
        "figure": str(figure_path),
        "limitations": ["Synthetic operating-condition evaluation uses CWRU load/RPM as a proxy, not a live industrial site.",
                        "B support and evaluation are chronological halves of the same recordings, so stationary vibration segments may be correlated.",
                        "The zero-order arm uses local labeled maintenance feedback; it demonstrates no-gradient point adaptation, not fully unsupervised adaptation.",
                        "The test curve is diagnostic and does not imply statistical significance or generalization to unseen machines."],
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    compact_metrics = {name: arm["metrics"] for name, arm in arms.items() if "metrics" in arm}
    _log(log_path, f"saved {out_path}; arms={compact_metrics}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-budget", type=int, default=DEFAULT_UPDATE_BUDGET)
    args = parser.parse_args()
    run_e4(update_budget=args.update_budget)
