import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from audit_drift import weekly_distribution_drift
from datasets_cfpb import load_cfpb
from explain_region import make_region_map
from export_edge import export_bundle
from textcloud import ComplaintCloud


def _tiny_frame():
    rows = []
    for index in range(12):
        rows.append({
            "consumer_complaint_narrative": (
                "billing payment problem" if index % 2 == 0 else "credit loan issue"
            ),
            "date_received": f"2020-01-{index + 1:02d}",
            "product": "Credit card",
            "issue": "Billing" if index % 2 == 0 else "Loan",
        })
    rows.extend([
        {"consumer_complaint_narrative": "missing date", "date_received": None,
         "product": "Card", "issue": "Billing"},
        {"consumer_complaint_narrative": "missing product", "date_received": "2020-02-01",
         "product": None, "issue": "Billing"},
        {"consumer_complaint_narrative": "  ", "date_received": "2020-02-02",
         "product": "Card", "issue": "Billing"},
    ])
    return pd.DataFrame(rows)


def test_dataset_filters_deterministic_sample_and_chronological_split(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pq.write_table(pa.Table.from_pandas(_tiny_frame()), data_dir / "part.parquet")
    first = load_cfpb(n_sample=8, seed=13, data_dir=str(tmp_path))
    second = load_cfpb(n_sample=8, seed=13, data_dir=str(tmp_path))
    assert len(first) == 8
    assert first["narrative"].tolist() == second["narrative"].tolist()
    assert first["date"].is_monotonic_increasing
    assert first["split"].value_counts().to_dict() == {"train": 5, "drift": 3}
    assert set(first["split"]) == {"train", "drift"}
    assert first["week"].notna().all()


def test_textcloud_forward_shape_and_optional_issue_logits():
    model = ComplaintCloud(vocab_size=30, d_model=12, d_cloud_dim=8,
                           d_cloud=16, hidden=12, d_z=6, n_issue=3)
    tokens = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])
    mask = tokens.ne(0)
    state, z, logits = model(tokens, mask)
    assert state.shape == (2, 16)
    assert z.shape == (2, 6)
    assert logits.shape == (2, 3)


def test_kmeans_region_names_use_top_bpe_tokens():
    from cloudlm.tokenizer import BPETokenizer

    texts = ["billing payment problem"] * 4 + ["credit loan issue"] * 4
    tokenizer = BPETokenizer(vocab_size=30).train(texts, vocab_size=30)
    frame = pd.DataFrame({"narrative": texts})
    states = np.vstack([np.tile([0.0, 0.0], (4, 1)),
                        np.tile([5.0, 5.0], (4, 1))])
    result = make_region_map(frame, states, tokenizer, n_regions=2, seed=0)
    assert result["n_regions"] == 2
    assert sum(region["n_records"] for region in result["regions"].values()) == 8
    assert all(region["top_tokens"][:3] for region in result["regions"].values())
    assert all("centroid" in region and "examples" in region
               for region in result["regions"].values())


def test_weekly_drift_shares_tv_kl_and_new_region_flag():
    report = weekly_distribution_drift(
        ["2020-W01"] * 100 + ["2020-W02"] * 100,
        [0] * 100 + [1] * 10 + [0] * 90,
        n_regions=2,
    )
    assert report["2020-W01"]["region_shares"] == [1.0, 0.0]
    assert report["2020-W02"]["new_regions"] == [1]
    assert report["2020-W02"]["tv_vs_previous_week"] == pytest.approx(0.1)
    assert report["2020-W02"]["kl_vs_previous_week"] > 0


def test_edge_export_torch_onnx_parity(tmp_path):
    from cloudlm.tokenizer import BPETokenizer

    texts = ["complaint about card billing", "complaint about loan payment"]
    tokenizer = BPETokenizer(vocab_size=32).train(texts, vocab_size=32)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    model_config = {"vocab_size": tokenizer.num_tokens, "d_model": 8,
                    "d_cloud_dim": 4, "d_cloud": 8, "hidden": 8,
                    "d_z": 4, "n_issue": 2}
    model = ComplaintCloud(**model_config)
    checkpoint_path = tmp_path / "reader.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": model_config,
                "config": {"max_len": 8, "n_sample": 2, "seed": 0}},
               checkpoint_path)
    manifest, bundle_size = export_bundle(
        str(checkpoint_path), str(tokenizer_path), str(tmp_path / "edge"),
        max_len=8, repeats=2,
    )
    assert manifest["self_test"]["passed"]
    assert manifest["self_test"]["max_abs_delta_z"] < 1e-4
    assert bundle_size > 0
    assert os.path.exists(tmp_path / "edge" / "manifest.json")
