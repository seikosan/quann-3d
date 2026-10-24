import torch
import torch.nn as nn
import torch.nn.functional as F

from quann_cloud.src.qcloud_quann import QUANN, QUANNConfig


def test_quann_engine_runs_without_backprop_and_reports_step_metrics():
    model = nn.Linear(2, 1)

    def loss_fn(module, inputs, targets):
        return F.mse_loss(module(inputs), targets).item()

    engine = QUANN(
        model,
        device=torch.device("cpu"),
        cfg=QUANNConfig(seed=7, k_weights=2),
        loss_fn=loss_fn,
    )
    inputs = torch.tensor([[0.5, -0.25]])
    targets = torch.tensor([[0.75]])
    stats = engine.step(inputs, targets)

    assert set(stats) == {"base_err", "rewarded", "censured", "neutral", "converged"}
    assert stats["base_err"] >= 0.0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
