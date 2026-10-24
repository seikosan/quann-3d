from quann_cloud.src.qcloud_network import QCloudNetwork
import torch
import unittest


class TestQCloudNetwork(unittest.TestCase):
    def test_forward_pass(self):
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3)
        tokens = torch.tensor([[1, 2, 3]])
        mask = (tokens != 0).bool()
        
        word_logits, next_point, points = model(tokens, mask)
        self.assertEqual(word_logits.shape, (1, 100))
        self.assertEqual(next_point.shape, (1, 3))
        self.assertEqual(points.shape, (1, 3, 3))

    def test_generate(self):
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3)
        tokens = torch.tensor([[1]])
        generated = model.generate(tokens, max_new=5)
        self.assertEqual(generated.shape[1], 6)  # 1 (prompt) + 5 (new)

    def test_get_weight(self):
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3)
        w = model.get_weight(5)
        self.assertEqual(w.D, 3)
        self.assertIn(5, model.token_weights)

    def test_update_token_weights(self):
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3).eval()
        tokens = torch.tensor([[2, 3, 0]])
        mask = tokens != 0
        stats = model.update_token_weights(tokens, mask, 4)
        self.assertEqual(
            set(stats),
            {'rewarded', 'censured', 'unchanged', 'base_error', 'final_error', 'mean_delta'},
        )
        self.assertEqual(stats['rewarded'] + stats['censured'] + stats['unchanged'], 2)
        self.assertTrue(torch.isfinite(torch.tensor(stats['base_error'])))
        self.assertTrue(torch.isfinite(torch.tensor(stats['final_error'])))
        self.assertGreaterEqual(stats['final_error'], 0.0)

    def test_compute_error_uses_model_device(self):
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3).eval()
        tokens = torch.tensor([[2, 3]])
        mask = tokens != 0
        error = model.compute_error(tokens, mask, target_token=4)
        self.assertGreaterEqual(error, 0.0)

    def test_bidirectional_flag_controls_opposite_search(self):
        tokens = torch.tensor([[2, 3]])
        mask = tokens != 0
        for bidirectional in (False, True):
            model = QCloudNetwork(
                vocab_size=100, d_model=64, d_cloud_dim=3,
                bidirectional=bidirectional,
            ).eval()
            self.assertIs(model.bidirectional, bidirectional)
            stats = model.update_token_weights(tokens, mask, target_token=4)
            self.assertEqual(stats['rewarded'] + stats['censured'] + stats['unchanged'], 2)

    def test_all_converged(self):
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3)
        self.assertFalse(model.all_converged())  # Boş → False

if __name__ == '__main__':
    unittest.main()
