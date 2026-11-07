"""Shape, masking, geometry, and checkpoint tests for ContextCloudLM."""

from __future__ import annotations

import torch

from contextcloud.model import (
    ContextCloudLM,
    SentenceBulutEncoder,
    WindowBulutEncoder,
    WindowGeometryEncoder,
)
from purecloudlm import PureCloudLM


def _tiny_context_model() -> ContextCloudLM:
    """Create a small model suitable for fast CPU tests."""
    return ContextCloudLM(
        vocab_size=32,
        d_model=16,
        d_sentence=12,
        d_window=12,
        d_cloud_dim=4,
        d_window_dim=4,
        proj_hidden=12,
        window_size=2,
        window_stride=1,
        hidden=12,
        max_tokens=8,
    )


def test_sentence_encoder_single_sentence() -> None:
    """Test that one padded token sequence produces a finite sentence cloud."""
    encoder = SentenceBulutEncoder(
        vocab_size=32,
        d_model=16,
        d_cloud_dim=4,
        d_cloud=12,
        proj_hidden=12,
        hidden=12,
        max_tokens=8,
        geometry_max_tokens=8,
    ).eval()
    tokens = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])

    with torch.no_grad():
        states = encoder(tokens)

    assert states.shape == (2, 12)
    assert torch.isfinite(states).all()


def test_window_encoder_creates_sliding_clouds() -> None:
    """Test window output shape and validity for a sentence sequence."""
    encoder = WindowBulutEncoder(
        sentence_dim=12,
        window_size=2,
        stride=1,
        d_window_dim=4,
        d_cloud=10,
        hidden=8,
        geometry_max_len=4,
    ).eval()
    sentence_states = torch.randn(2, 4, 12)
    sentence_mask = torch.tensor(
        [[True, True, True, False], [False, False, False, False]]
    )

    with torch.no_grad():
        windows, valid_windows = encoder(sentence_states, sentence_mask)

    assert windows.shape == (2, 3, 10)
    assert valid_windows.tolist() == [[True, True, True], [False, False, False]]
    assert torch.isfinite(windows).all()
    assert torch.count_nonzero(windows[1]) == 0


def test_context_model_forward_2d_and_3d() -> None:
    """Test single-sentence and multi-sentence input forms."""
    model = _tiny_context_model().eval()
    single_sentence = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    context = torch.tensor(
        [
            [[2, 3, 0, 0], [4, 5, 6, 0], [7, 8, 0, 0]],
            [[9, 10, 11, 0], [12, 13, 0, 0], [14, 15, 16, 0]],
        ]
    )

    with torch.no_grad():
        single_logits = model(single_sentence)
        context_logits = model(context)

    assert single_logits.shape == (2, 32)
    assert context_logits.shape == (2, 32)
    assert torch.isfinite(single_logits).all()
    assert torch.isfinite(context_logits).all()


def test_context_model_generate_2d_and_3d() -> None:
    """Test sampling appends tokens along the correct sequence dimension."""
    model = _tiny_context_model()
    tokens_2d = torch.tensor([[2, 3], [4, 5]])
    tokens_3d = torch.tensor([[[2, 3], [4, 5]]])

    generated_2d = model.generate(tokens_2d, max_new=2, eos_id=-1, top_k=5)
    generated_3d = model.generate(tokens_3d, max_new=2, eos_id=-1, top_k=5)

    assert generated_2d.shape == (2, 4)
    assert generated_3d.shape == (1, 2, 4)
    assert torch.equal(generated_2d[:, :2], tokens_2d)
    assert torch.equal(generated_3d[:, :, :2], tokens_3d)


def test_context_model_generate_validates_sampling_parameters() -> None:
    """Reject sampling settings that would produce invalid probabilities."""
    model = _tiny_context_model()
    tokens = torch.tensor([[2, 3]])

    for invalid_temperature in (0.0, -1.0, float("nan")):
        try:
            model.generate(tokens, temperature=invalid_temperature)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid temperature should raise ValueError")


def test_padding_mask_independence() -> None:
    """Test that changed token IDs in explicitly masked padding do not affect logits."""
    model = _tiny_context_model().eval()
    tokens = torch.tensor(
        [
            [[2, 3, 0, 0], [4, 5, 6, 0], [7, 0, 0, 0]],
            [[8, 9, 10, 0], [11, 12, 0, 0], [13, 14, 15, 0]],
        ]
    )
    mask = tokens.ne(0)
    changed_tokens = tokens.clone()
    changed_tokens[~mask] = 17

    with torch.no_grad():
        original_logits = model(tokens, sentence_mask=mask)
        changed_logits = model(changed_tokens, sentence_mask=mask)

    torch.testing.assert_close(original_logits, changed_logits)


def test_masked_single_point_context_has_finite_gradients() -> None:
    """Test backward stability when padded sentences leave one valid geometry point."""
    model = _tiny_context_model()
    tokens = torch.tensor([[[2, 0, 0, 0], [0, 0, 0, 0]]])
    logits = model(tokens)
    logits.square().mean().backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_purecloudlm_checkpoint_transfer() -> None:
    """Test compatible embedding and geometry weights transfer from PureCloudLM."""
    pure_model = PureCloudLM(
        vocab_size=32,
        d_model=16,
        d_cloud=12,
        d_cloud_dim=4,
        n_layers=1,
        pos_len=8,
        hidden=12,
        proj_hidden=12,
        point_hidden=8,
        use_pair_approx=False,
    )
    context_model = _tiny_context_model()
    expected_embedding = pure_model.embedding.weight.detach().clone()

    transfer_result = context_model.load_purecloudlm_checkpoint(
        {"model_state": pure_model.state_dict()}
    )

    assert torch.equal(context_model.sentence_encoder.embedding.weight, expected_embedding)
    assert "sentence_encoder.embedding.weight" not in transfer_result.missing_keys


def test_window_geometry_encoder_masking_and_shape() -> None:
    """Test the global window-geometry summary with padded windows."""
    encoder = WindowGeometryEncoder(window_dim=12, centroid_dim=6, hidden=8).eval()
    states = torch.randn(2, 3, 12)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    changed_states = states.clone()
    changed_states[~mask] = 100.0

    with torch.no_grad():
        geometry = encoder(states, mask)
        changed_geometry = encoder(changed_states, mask)

    assert geometry.shape == (2, 12)
    assert torch.isfinite(geometry).all()
    torch.testing.assert_close(geometry, changed_geometry)
