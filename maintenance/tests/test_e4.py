import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "maintenance" / "src"))
sys.path.insert(0, str(ROOT))

from e4_adapt import _sample_balanced, zero_order_adapt
from train_maintenance import MaintenanceCloud


def test_balanced_local_budget_is_fixed_and_class_balanced():
    labels = np.asarray([0] * 30 + [1] * 6)
    tokens = np.arange(len(labels), dtype=np.int64).reshape(-1, 1)
    selected = _sample_balanced(np.empty((len(labels), 0)), labels, tokens, budget=10)
    assert len(selected) == 10
    assert int(np.sum(labels[selected] == 0)) == 5
    assert int(np.sum(labels[selected] == 1)) == 5


def test_zero_order_adaptation_does_not_change_reader_parameters_or_call_gradients():
    torch.manual_seed(3)
    model = MaintenanceCloud(vocab_size=3, n_classes=2, D=32)
    tokens = np.asarray([[0], [1], [0], [2]], dtype=np.int64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    details = zero_order_adapt(model, tokens, labels, update_budget=4)
    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert details["actual_updates"] == 4
    assert details["gradient_calls"] == 0
    assert details["backprop_calls"] == 0
    assert details["points_after"] > 0
