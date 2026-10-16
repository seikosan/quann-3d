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


# -------------------------------------------------------------------------- qcloud_fraud
def test_qcloud_fraud_forward_and_predict():
    d = _small()
    fq = quantize.FeatureQuantizer()
    tok = fq.fit_transform(d)
    m = QCloudFraudClassifier(vocab_size=fq.vocab_size, d_model=16, d_cloud=32, d_cloud_dim=6)
    t = torch.tensor(tok[:8])
    mask = (t != 0) & (t != 1)
    lp, pts = m(t, mask, points_override=m.points_for(t, mask))
    assert tuple(lp.shape) == (8, 1) and tuple(pts.shape) == (8, 13, 6)
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
