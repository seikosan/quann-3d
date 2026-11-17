"""purecloudlm/train.py duman testi — CPU, sentetik token, BPE/corpus yok."""

import math
import random

import torch

from purecloudlm import PureCloudLM
from purecloudlm.train import TrainConfig, make_batch, train_epoch


def _tiny_model() -> PureCloudLM:
    return PureCloudLM(
        vocab_size=32,
        d_model=16,
        d_cloud=12,
        d_cloud_dim=8,
        n_layers=1,
        pos_len=5,
        hidden=12,
        proj_hidden=12,
        point_hidden=8,
        use_pair_approx=False,
        num_sampled_pairs=4,
    )


def test_make_batch_pads_inputs_and_picks_next_token_target() -> None:
    tokens = [[2, 3, 4, 5, 6, 7], [2, 3]]
    inputs, targets, mask = make_batch(
        tokens, seq_len=4, eos_id=9, device=torch.device("cpu"))
    assert inputs.shape == (2, 4)
    assert targets.tolist() == [6, 0], "hedef seq_len sonraki token; yoksa 0"
    assert mask.tolist() == [[True, True, True, True], [True, True, False, False]]


def test_make_batch_masks_eos() -> None:
    tokens = [[2, 9, 4, 5, 6]]
    _, _, mask = make_batch(tokens, seq_len=4, eos_id=9, device=torch.device("cpu"))
    assert mask.tolist() == [[True, False, True, True]]


def test_train_epoch_runs_and_returns_finite_loss() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    cfg = TrainConfig(batch_size=4, seq_len=4, epochs=1)
    rng = random.Random(0)
    all_tokens = [[rng.randint(2, 31) for _ in range(8)] for _ in range(16)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    avg, nan_count, n_batches = train_epoch(
        model, all_tokens, optimizer, torch.device("cpu"), cfg, eos_id=1)

    assert n_batches == 4 and nan_count == 0
    assert math.isfinite(avg)


def test_train_epoch_loss_decreases_on_tiny_set() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    cfg = TrainConfig(batch_size=8, seq_len=6, epochs=1)
    sabit = [5, 6, 7, 8, 9, 10, 11, 12]
    all_tokens = [list(sabit) for _ in range(16)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first, _, _ = train_epoch(model, all_tokens, optimizer, torch.device("cpu"), cfg, 1)
    second, _, _ = train_epoch(model, all_tokens, optimizer, torch.device("cpu"), cfg, 1)
    assert second < first, "tekrarlanan tek örnek üzerinde loss düşmeli"
