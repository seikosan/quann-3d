from __future__ import annotations

import os
import sys

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

import config
from generator import generate_transactions
from quantize import FeatureQuantizer
from train_aml import AMLCloud
from e1_explain import run_e1


def test_generator_typologies_have_distinct_signatures():
    _, windows = generate_transactions(n_transactions=4_000, seed=731)
    by_class = windows.groupby("typology").median(numeric_only=True)
    assert set(windows.typology.unique()) == set(config.CLASSES)
    assert by_class.loc["structuring", "amount_median"] > 9_000
    assert by_class.loc["velocity", "inter_tx_interval_min"] < 1.0
    assert by_class.loc["layering", "chain_hops"] >= 9
    assert by_class.loc["velocity", "rate_per_hour"] > by_class.loc["normal", "rate_per_hour"]
    assert windows["window_id"].nunique() == 400


def test_quantize_cloud_update_and_multiclass_prediction_smoke():
    _, windows = generate_transactions(n_transactions=800, seed=19)
    quantizer = FeatureQuantizer().fit(windows.iloc[:60])
    assert "counterparty_id" in quantizer.feature_order
    tokens = quantizer.transform(windows.iloc[:60])
    y = np.asarray([config.CLASSES.index(name) for name in windows.iloc[:60].typology])
    model = AMLCloud(quantizer.vocab_size)
    token_tensor = torch.as_tensor(tokens[:1], dtype=torch.long)
    model.quann_update(token_tensor, int(y[0]))
    with torch.no_grad():
        logits, cloud = model(token_tensor, points_override=model.points_for(token_tensor))
    assert logits.shape == (1, 4)
    assert cloud.shape == (1, len(quantizer.feature_order), 8)
    assert model.convergence_info()["points"] > 0


def test_e1_end_to_end_smoke(tmp_path):
    result = run_e1(
        n_transactions=800,
        out_path=str(tmp_path / "e1.json"),
        figure_path=str(tmp_path / "e1.png"),
        checkpoint_path=str(tmp_path / "missing-e0-checkpoint.pt"),
        log_path=None,
    )
    assert result["encoder_source"] == "e1_cpu_fallback_fit"
    assert result["summary"]["probe_classifier_argmax_accuracy"] >= 0.0
    assert set(result["summary"]["class_distance_separation"]) == set(config.CLASSES)
    assert len(result["probes"]) == 16  # four held-out examples per class in this tiny run
    layering = [probe for probe in result["probes"] if probe["true_class"] == "layering"]
    assert all("çok-hop zincir işareti" in probe["observable_reason"] for probe in layering)
    assert all("SAR kararı değildir" in probe["sar_draft"] for probe in layering)
    assert (tmp_path / "e1.json").stat().st_size > 0
    assert (tmp_path / "e1.png").stat().st_size > 0
