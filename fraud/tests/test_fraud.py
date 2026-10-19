"""Unit tests for the fraud sub-project modules.

Run:
    python3 fraud/tests/test_fraud.py        # plain
    python3 -m pytest fraud/tests/ -v        # pytest (if installed)

Covers: features (numeric), quantize (tokens), datasets (split/sample),
metrics, and the QCloudFraudClassifier (forward + QUANN update + save/load).
No network / no GPU; uses the bundled User0 file.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config
import datasets
import features
import quantize
import metrics
import audit_false_alarms
import explain_pilot
import train_duels
import train_rich_points
from qcloud_fraud import QCloudFraudClassifier


def _small():
    return pd.read_csv(config.SMALL_CSV)


# -------------------------------------------------------------------------- features
def test_features_shape_and_finite():
    d = _small()
    ft = features.FeatureTransformer()
    X = ft.fit_transform(d)
    assert X.shape[0] == len(d)
    assert X.shape[1] == len(ft.feature_names) > 10
    assert X.dtype == np.float32
    assert not np.isnan(X).any() and not np.isinf(X).any()


def test_features_leakage_safe_fit_transform_differs_from_transform():
    """fit on one frame, transform another -> same feature names, stable dims."""
    d = _small()
    ft = features.FeatureTransformer().fit(d.iloc[:1000])
    Xa = ft.transform(d.iloc[:1000])
    Xb = ft.transform(d.iloc[1000:2000])
    assert Xa.shape[1] == Xb.shape[1] == len(ft.feature_names)


# -------------------------------------------------------------------------- quantize
def test_quantize_vocab_and_ranges():
    d = _small()
    fq = quantize.FeatureQuantizer()
    tok = fq.fit_transform(d)
    assert tok.shape == (len(d), fq.n_features)
    assert fq.n_features == 20
    assert tok.dtype == np.int64
    assert tok.min() >= 2 and tok.max() < fq.vocab_size
    for j, (key, kind, n) in enumerate(quantize.QUANT_SPECS):
        col = tok[:, j]
        lo, hi = fq.offsets[key], fq.offsets[key] + fq.n_per_feature[key]
        assert col.min() >= lo and col.max() < hi, f"{key} out of range"


def test_quantize_fit_transform_reproducible_on_same_input():
    d = _small()
    fq = quantize.FeatureQuantizer()
    a = fq.fit_transform(d)
    b = fq.transform(d)
    assert np.array_equal(a, b)


# -------------------------------------------------------------------------- datasets
def test_split_no_user_leakage_big_style():
    d = _small()  # single user -> falls back to stratified random
    tr, va, te = datasets.split_by_user(d)
    assert len(tr) + len(va) + len(te) == len(d)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_enrich_sample_respects_fraud_frac_cap():
    d = _small()
    y = datasets.get_label(d)
    sel = datasets.enrich_sample(d, n_total=5000, fraud_frac=0.5)
    got = y[sel].mean()
    assert 0 < got <= 0.5 + 1e-9
    assert len(sel) <= 5000


# -------------------------------------------------------------------------- metrics
def test_metrics_shapes_and_threshold():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(900), np.ones(100)])
    prob = np.where(y == 1, rng.uniform(0.6, 1), rng.uniform(0, 0.6))
    thr = metrics.best_threshold(y, prob)
    m = metrics.evaluate(y, prob, threshold=thr)
    assert 0 <= thr <= 1
    assert m["auc_pr"] > 0.5 and m["auc_roc"] > 0.9
    assert m["n"] == 1000 and m["n_fraud"] == 100


def test_duels_models_train_and_return_metrics_on_toy_frame():
    """Exercise all CPU duel trainers and their shared metric/threshold path."""
    rng = np.random.default_rng(2026)
    n = 500
    labels = np.tile(np.array([0, 1]), n // 2)
    rng.shuffle(labels)
    amount = rng.lognormal(mean=3.0, sigma=1.0, size=n) + labels * 40
    frame = pd.DataFrame({
        "User": np.arange(n) % 20,
        "Card": np.arange(n) % 50,
        "Year": np.full(n, 2020),
        "Month": rng.integers(1, 13, size=n),
        "Day": rng.integers(1, 29, size=n),
        "Time": [f"{int(h):02d}:{int(m):02d}" for h, m in
                 zip(rng.integers(0, 24, n), rng.integers(0, 60, n))],
        "Amount": [f"${v:.2f}" for v in amount],
        "Use Chip": rng.choice(["Chip", "Swipe", "Online"], size=n),
        "Merchant Name": rng.integers(1, 40, size=n),
        "Merchant City": rng.choice(["A", "B", "C", "D"], size=n),
        "Merchant State": rng.choice(["CA", "NY", "TX"], size=n),
        "Zip": rng.choice([94102, 10001, np.nan], size=n),
        "MCC": rng.integers(1000, 1020, size=n),
        "Errors?": rng.choice([None, "Bad PIN", "Insufficient Balance"], size=n),
        "Is Fraud?": np.where(labels == 1, "Yes", "No"),
    })
    transformer = features.FeatureTransformer().fit(frame.iloc[:300])
    Xtr = transformer.transform(frame.iloc[:300])
    Xva = transformer.transform(frame.iloc[300:400])
    Xte = transformer.transform(frame.iloc[400:])
    ytr, yva, yte = labels[:300], labels[300:400], labels[400:]
    configs = {
        "xgboost": {"n_estimators": 3, "early_stopping_rounds": 2, "n_jobs": 1,
                    "max_depth": 2},
        "lightgbm": {"n_estimators": 3, "early_stopping_rounds": 2, "n_jobs": 1,
                     "max_depth": 2},
        "ftt": {"epochs": 1, "batch_size": 64, "d_model": 32,
                "n_layers": 1, "n_heads": 4, "cat_dim": 8, "ff_dim": 16},
    }
    required = {"auc_roc", "auc_pr", "fbeta", "precision", "recall", "lift", "threshold"}
    for name, options in configs.items():
        result, _ = train_duels.run_model(
            name, Xtr, ytr, Xva, yva, Xte, yte, transformer.feature_names,
            model_options=options,
        )
        assert required <= result.keys()
        assert all(np.isfinite(result[key]) for key in required)


def test_quann_duels_smoke_writes_split_sizes_and_all_model_metrics(monkeypatch, tmp_path):
    """Exercise the QUANN-selected path and serialized metadata on a tiny view."""
    rng = np.random.default_rng(20261015)
    sizes = (1200, 400, 400)
    n = sum(sizes)
    labels = np.concatenate([
        np.tile(np.array([0, 1]), size // 2) for size in sizes
    ])
    rng.shuffle(labels)
    amount = rng.lognormal(mean=3.0, sigma=1.0, size=n) + labels * 40
    frame = pd.DataFrame({
        "User": np.arange(n) % 100,
        "Card": np.arange(n) % 250,
        "Year": np.full(n, 2020),
        "Month": rng.integers(1, 13, size=n),
        "Day": rng.integers(1, 29, size=n),
        "Time": [f"{int(h):02d}:{int(m):02d}" for h, m in
                 zip(rng.integers(0, 24, n), rng.integers(0, 60, n))],
        "Amount": [f"${v:.2f}" for v in amount],
        "Use Chip": rng.choice(["Chip", "Swipe", "Online"], size=n),
        "Merchant Name": rng.integers(1, 40, size=n),
        "Merchant City": rng.choice(["A", "B", "C", "D"], size=n),
        "Merchant State": rng.choice(["CA", "NY", "TX"], size=n),
        "Zip": rng.choice([94102, 10001, np.nan], size=n),
        "MCC": rng.integers(1000, 1020, size=n),
        "Errors?": rng.choice([None, "Bad PIN", "Insufficient Balance"], size=n),
        "Is Fraud?": np.where(labels == 1, "Yes", "No"),
    })
    boundaries = np.cumsum((0, *sizes))
    split_indices = tuple(np.arange(boundaries[i], boundaries[i + 1])
                          for i in range(3))

    def tiny_build_data(df, n_train, n_val, n_test, train_frac, test_frac, seed):
        assert (n_train, n_val, n_test) == (1_020_000, 100_000, 4_700_000)
        return dict(zip(("tr_sel", "va_sel", "te_sel"), split_indices))

    monkeypatch.setattr(train_duels.datasets, "load_big", lambda: frame)
    monkeypatch.setattr(train_duels.train_quann, "build_data", tiny_build_data)
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "artifacts"))
    out_path = tmp_path / "quann_smoke.json"
    train_duels.main([
        "--smoke", "--split", "quann", "--models", "xgboost,lightgbm,ftt",
        "--n-jobs", "1", "--out", str(out_path),
    ])

    import json

    payload = json.loads(out_path.read_text())
    assert payload["data"]["split_name"] == "quann"
    assert payload["data"]["split"] == {"train": 1200, "val": 400, "test": 400}
    assert payload["data"]["split_full"] == {"train": 1200, "val": 400, "test": 400}
    assert payload["data"]["fraud_by_split_full"] == {
        "train": int(labels[split_indices[0]].sum()),
        "val": int(labels[split_indices[1]].sum()),
        "test": int(labels[split_indices[2]].sum()),
    }
    for name in ("xgboost", "lightgbm", "ftt"):
        assert set(train_duels.METRIC_KEYS) <= payload["models"][name].keys()


# -------------------------------------------------------------------------- qcloud_fraud
def test_qcloud_fraud_forward_and_predict():
    d = _small()
    fq = quantize.FeatureQuantizer()
    tok = fq.fit_transform(d)
    m = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=16, d_cloud=32, d_cloud_dim=6)
    t = torch.tensor(tok[:8])
    mask = (t != 0) & (t != 1)
    lp, pts = m(t, mask, points_override=m.points_for(t, mask))
    assert tuple(lp.shape) == (8, 1) and tuple(pts.shape) == (8, fq.n_features, 6)
    p = m.predict_proba(t, use_quann_points=True)
    assert p.shape[0] == 8 and (p >= 0).all() and (p <= 1).all()


def test_qcloud_fraud_quann_update_changes_points_and_saveload():
    d = _small()
    fq = quantize.FeatureQuantizer()
    tok = fq.fit_transform(d)
    y = datasets.get_label(d)
    m = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=16, d_cloud=32, d_cloud_dim=6)
    # a fraud example
    i = int(np.flatnonzero(y)[0])
    ex = torch.tensor(tok[i:i + 1])
    msk = (ex != 0) & (ex != 1)
    tid = int(ex[0, 0])
    before = m.get_weight(tid).value.copy()
    st = m.update_token_weights(ex, msk, int(y[i]))
    assert st["base_error"] >= 0
    assert st["rewarded"] + st["censured"] + st["unchanged"] == ex.shape[1]
    # after a few steps, weights exist and stay within [-1, 1] init box
    for _ in range(5):
        m.update_token_weights(ex, msk, int(y[i]))
    assert len(m.token_weights) == ex.shape[1]
    for j in range(ex.shape[1]):
        v = m.get_weight(int(ex[0, j])).value
        assert v.shape == (m.D,) and np.all(np.isfinite(v))
    # save / load round-trip
    path = "/tmp/_test_qcloud.pt"
    m.save_quann(path)
    m2 = QCloudFraudClassifier.load_quann(path, torch.device("cpu"))
    assert m2.vocab_size == m.vocab_size
    assert len(m2.token_weights) == len(m.token_weights)
    assert np.allclose(m2.get_weight(tid).value, m.get_weight(tid).value)
    os.remove(path)


def test_rich_points_tiny_d16_train_and_reader_step():
    """Exercise appended K=20 tokenization, three point steps, and one reader step."""
    d = _small().iloc[:100].copy()
    fq = quantize.FeatureQuantizer().fit(d)
    tok = fq.transform(d)
    y = datasets.get_label(d)
    assert tok.shape == (100, 20)
    assert fq.vocab_size < 400
    assert {"amount_abs_log", "amount_neg", "zip_missing", "has_error",
            "err_bad_pin", "err_insufficient_balance", "err_bad_zipcode"} <= set(fq.feature_order)

    t, mask, target = train_rich_points._tensors(tok, y, torch.device("cpu"))
    model = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=16,
                                  d_cloud=32, d_cloud_dim=16, hidden=16)
    point_result = train_rich_points.train_point_steps(model, t, target, steps=3)
    reader_result = train_rich_points.train_reader(
        model, t, target, epochs=1, batch_size=100,
        pos_weight=torch.tensor(20.0), max_steps=1)
    assert point_result["steps"] == 3
    assert point_result["steps_per_sec"] > 0
    assert reader_result["steps"] == 1
    assert model.D == 16 and model.get_weight(int(t[0, 0])).value.shape == (16,)


def test_explain_pilot_select_upper_slice():
    probs = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.05, 0.95])
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 0])

    sel = explain_pilot.select_upper_slice(probs, labels, 5)
    assert len(sel) == 5
    assert set(sel) == {1, 3, 6, 7, 5}

    sel = explain_pilot.select_upper_slice(probs, labels, 2)
    assert len(sel) == 2
    assert set(sel) == {7, 1}

    sel = explain_pilot.select_upper_slice(probs, labels, 8)
    assert len(sel) == 8
    assert set(sel) == set(range(8))


def test_explain_pilot_decode_ablation_aggregate_and_cases():
    d = _small().iloc[:16]
    fq = quantize.FeatureQuantizer(specs=quantize.LEGACY_QUANT_SPECS).fit(d)
    tok = fq.transform(d)
    decoded = explain_pilot.decode_row(fq, tok[0])
    assert len(decoded) == 13
    assert all(isinstance(s, str) and s for s in decoded)

    model = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=8,
                                  d_cloud=16, d_cloud_dim=4, hidden=8)
    t = torch.tensor(tok)
    mask = (t != 0) & (t != 1)
    contribs = explain_pilot.point_ablation(model, t, mask, chunk=8).numpy()
    assert contribs.shape == tok.shape
    assert np.isfinite(contribs).all()

    rng = np.random.default_rng(0)
    agg = explain_pilot.aggregate(fq, tok, rng.normal(size=tok.shape), top_tokens=3)
    assert {"features", "top_tokens", "top_positive", "top_negative"} <= agg.keys()
    labels = datasets.get_label(d)
    probs = np.clip(1.0 / (1.0 + np.exp(-contribs[:, 0])), 0.0, 1.0)
    cases = explain_pilot.select_cases(fq, labels, probs, tok, contribs,
                                       n_cases=2, top_k=3)
    assert cases
    assert all("text" in c for c in cases)


def test_audit_false_alarms_small_smoke():
    """End-to-end audit on the bundled User0 sample: population, slices, joint, cases."""
    import json
    import tempfile

    tmp = tempfile.mkdtemp(prefix="fraud_audit_test_")
    out = os.path.join(tmp, "false_alarm_audit_dev.json")
    audit_false_alarms.main([
        "--data", "small", "--n-fraud", "27", "--n-normal", "300",
        "--threshold", "0.5", "--tn-n", "100", "--slices", "6",
        "--pairs", "3", "--cases", "4", "--chunk", "256", "--out", out,
    ])
    payload = json.loads(open(out).read())
    assert {"population", "fp_aggregate", "tn_aggregate",
            "slices", "joint", "cases", "method"} <= payload.keys()
    pop = payload["population"]
    assert pop["n"] > 0 and 0 <= pop["threshold"] <= 1
    assert pop["tp"] + pop["fp"] + pop["fn"] <= pop["n"]
    assert pop["fp_population"]["n"] == pop["fp"]
    for s in payload["slices"]:
        assert 0 <= s["frac_of_rows"] <= 1
        if s["mean_prob_with"] is not None:
            assert 0 <= s["mean_prob_with"] <= 1
            assert 0 <= s["frac_unflagged_if_absent"] <= 1
    for j in payload["joint"]:
        assert j["n_rows_both"] > 0 and 0 <= j["frac_unflagged_joint"] <= 1
        assert len(j["tokens"]) == 2
    assert all(np.isfinite(c["p_fraud"]) for c in payload["cases"])
    assert all(len(c["decoded"]) == 13 for c in payload["cases"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
