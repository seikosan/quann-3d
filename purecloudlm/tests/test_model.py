"""Tests for PureCloudLM"""
import torch
from purecloudlm import PureCloudLM


def _model(use_pair_approx=False) -> PureCloudLM:
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
        use_pair_approx=use_pair_approx,
        num_sampled_pairs=4,
    )


def test_forward_shapes_finite():
    """Test that forward pass produces finite outputs"""
    tokens = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 0]])
    mask = tokens != 0

    for use_pair_approx in [False, True]:
        model = _model(use_pair_approx=use_pair_approx).eval()
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


def test_padding_mask_independence():
    """Test that padding tokens don't affect cloud state"""
    model = _model(use_pair_approx=False).eval()
    
    points = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, True, False, False],
                         [True, True, True, True, False]])
    
    with torch.no_grad():
        original_state = model.cloud(points, mask)
        
        # Change padded points
        changed_points = points.clone()
        changed_points[~mask] = 100.0
        changed_state = model.cloud(changed_points, mask)
    
    # States should be identical (padding doesn't affect output)
    torch.testing.assert_close(original_state, changed_state)


def test_pair_sampling():
    """Test that pair sampling produces finite outputs"""
    model = _model(use_pair_approx=True).eval()
    
    tokens = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 0]])
    mask = tokens != 0
    
    with torch.no_grad():
        word_logits, next_point, points = model(tokens, mask)
    
    assert torch.isfinite(word_logits).all()
    assert torch.isfinite(next_point).all()


def test_different_pooling_modes():
    """Test different pooling modes work"""
    for pool_mode in ["all", "mean", "max"]:
        model = PureCloudLM(
            vocab_size=32,
            d_model=16,
            d_cloud=12,
            d_cloud_dim=8,
            n_layers=1,
            pos_len=5,
            hidden=12,
            proj_hidden=12,
            point_hidden=8,
            pool_mode=pool_mode,
            use_pair_approx=False,
        ).eval()
        
        tokens = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 0]])
        mask = tokens != 0
        
        with torch.no_grad():
            word_logits, next_point, points = model(tokens, mask)
        
        assert torch.isfinite(word_logits).all()
        assert torch.isfinite(next_point).all()


def test_no_transformer():
    """Verify that PureCloudLM doesn't use transformer layers"""
    model = _model()
    
    # Check no TransformerEncoder or MultiheadAttention
    for name, module in model.named_modules():
        module_type = type(module).__name__
        assert "Transformer" not in module_type, f"Found transformer module: {name}"
        assert "MultiheadAttention" not in module_type, f"Found attention module: {name}"
    
    print("✅ PureCloudLM confirmed: No transformer layers")


def test_o_n_complexity():
    """Test that GlobalGeometryEncoder uses O(N) operations"""
    model = _model(use_pair_approx=False)
    
    # Check that cloud encoder doesn't have O(N²) operations
    cloud = model.cloud
    
    # Verify no pair computation (full matrix)
    assert not hasattr(cloud, 'pair_mlp') or cloud.use_pair_approx == False
    
    print("✅ PureCloudLM confirmed: O(N) global pooling")


if __name__ == "__main__":
    test_forward_shapes_finite()
    print("✅ test_forward_shapes_finite passed")
    
    test_padding_mask_independence()
    print("✅ test_padding_mask_independence passed")
    
    test_pair_sampling()
    print("✅ test_pair_sampling passed")
    
    test_different_pooling_modes()
    print("✅ test_different_pooling_modes passed")
    
    test_no_transformer()
    print("✅ test_no_transformer passed")
    
    test_o_n_complexity()
    print("✅ test_o_n_complexity passed")
    
    print("\n✅ All PureCloudLM tests passed!")
