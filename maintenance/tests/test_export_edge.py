from __future__ import annotations

from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from export_edge import DEFAULT_CHECKPOINT, load_model


def test_saved_binary_checkpoint_serves_dynamic_batch_clouds():
    if not DEFAULT_CHECKPOINT.is_file():
        return
    model, checkpoint = load_model(DEFAULT_CHECKPOINT)
    assert checkpoint["config"]["n_classes"] == 2
    for batch in (1, 16, 64):
        with torch.inference_mode():
            probabilities = model(torch.zeros(batch, 10, 36))
        assert probabilities.shape == (batch, 2)
        assert torch.allclose(probabilities.sum(dim=-1), torch.ones(batch), atol=1e-6)
