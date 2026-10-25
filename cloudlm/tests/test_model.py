import torch

from cloudlm import CloudLM


def _model(pool: str, cloud_attn: bool = False) -> CloudLM:
    return CloudLM(
        vocab_size=32,
        d_model=16,
        d_cloud=12,
        d_cloud_dim=8,
        n_layers=1,
        nhead=4,
        pos_len=5,
        hidden=12,
        proj_hidden=12,
        point_hidden=8,
        pool=pool,
        cloud_attn=cloud_attn,
    )


def test_forward_shapes_finite_and_padding_mask() -> None:
    tokens = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 0]])
    mask = tokens != 0

    for pool in ("max", "attn"):
        model = _model(pool).eval()
        with torch.no_grad():
            word_logits, next_point, points = model(tokens, mask)
            unmasked_outputs = model(tokens)

        assert word_logits.shape == (2, 32)
        assert next_point.shape == (2, 8)
        assert points.shape == (2, 5, 8)
        assert torch.isfinite(word_logits).all()
        assert torch.isfinite(next_point).all()
        assert torch.isfinite(points).all()
        assert all(torch.isfinite(output).all() for output in unmasked_outputs)

        # Masked point values must not affect either pooling mode's cloud state.
        changed_points = points.clone()
        changed_points[~mask] = 100.0
        with torch.no_grad():
            original_state = model.cloud(points, mask)
            changed_state = model.cloud(changed_points, mask)
        torch.testing.assert_close(original_state, changed_state)


def test_attention_and_max_pool_produce_different_cloud_outputs() -> None:
    torch.manual_seed(7)
    max_model = _model("max").eval()
    attn_model = _model("attn").eval()
    attn_model.load_state_dict(max_model.state_dict(), strict=False)
    with torch.no_grad():
        attn_model.cloud.attn_proj.weight.zero_()
        attn_model.cloud.attn_proj.bias.zero_()

    points = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, True, False, False],
                         [True, True, True, True, False]])
    with torch.no_grad():
        max_state = max_model.cloud(points, mask)
        attn_state = attn_model.cloud(points, mask)

    assert torch.isfinite(max_state).all()
    assert torch.isfinite(attn_state).all()
    assert not torch.allclose(max_state, attn_state)


def test_legacy_max_pool_checkpoint_loads_strictly(tmp_path) -> None:
    saved_model = _model("max")
    old_state = {
        key: value for key, value in saved_model.state_dict().items()
        if not key.startswith("cloud.attn_proj.")
    }
    checkpoint = tmp_path / "legacy_max_state.pt"
    torch.save(old_state, checkpoint)

    loaded_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert not any(key.startswith("cloud.attn_proj.") for key in loaded_state)

    reloaded_model = _model("max").eval()
    reloaded_model.load_state_dict(loaded_state)
    tokens = torch.tensor([[2, 3, 0, 0, 0], [4, 5, 6, 0, 0]])
    mask = tokens != 0
    with torch.no_grad():
        outputs = reloaded_model(tokens, mask)

    assert all(torch.isfinite(output).all() for output in outputs)


def test_max_pool_does_not_let_padding_zero_dominate_negative_points():
    model = _model("max").eval()
    with torch.no_grad():
        model.cloud.point_mlp[-1].weight.zero_()
        model.cloud.point_mlp[-1].bias.fill_(-1.0)
    points = torch.full((1, 3, 8), -2.0)
    mask = torch.tensor([[True, True, False]])
    with torch.no_grad():
        masked_state = model.cloud(points, mask)
        valid_only_state = model.cloud(points[:, :2])
    torch.testing.assert_close(masked_state, valid_only_state)


def test_cloud_attn_with_max_pool():
    """Test cloud self-attention with max pooling (R2 feature)."""
    model = _model("max", cloud_attn=True).eval()
    tokens = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 0]])
    mask = tokens != 0
    
    with torch.no_grad():
        word_logits, next_point, points = model(tokens, mask)
    
    assert word_logits.shape == (2, 32)
    assert next_point.shape == (2, 8)
    assert points.shape == (2, 5, 8)
    assert torch.isfinite(word_logits).all()
    assert torch.isfinite(next_point).all()
    assert torch.isfinite(points).all()


def test_cloud_attn_with_attn_pool():
    """Test cloud self-attention with attention pooling (R2 feature)."""
    model = _model("attn", cloud_attn=True).eval()
    tokens = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 0]])
    mask = tokens != 0
    
    with torch.no_grad():
        word_logits, next_point, points = model(tokens, mask)
    
    assert word_logits.shape == (2, 32)
    assert next_point.shape == (2, 8)
    assert points.shape == (2, 5, 8)
    assert torch.isfinite(word_logits).all()
    assert torch.isfinite(next_point).all()
    assert torch.isfinite(points).all()


def test_cloud_attn_produces_different_outputs():
    """Verify that cloud attention produces different outputs than max pool."""
    torch.manual_seed(42)
    max_model = _model("max", cloud_attn=False).eval()
    attn_model = _model("max", cloud_attn=True).eval()
    
    # Load max model state into attn model (missing attn layers will be random)
    attn_model.load_state_dict(max_model.state_dict(), strict=False)
    
    points = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, True, False, False],
                         [True, True, True, True, False]])
    
    with torch.no_grad():
        max_state = max_model.cloud(points, mask)
        attn_state = attn_model.cloud(points, mask)
    
    assert torch.isfinite(max_state).all()
    assert torch.isfinite(attn_state).all()
    # Cloud attention should produce different output
    assert not torch.allclose(max_state, attn_state)
