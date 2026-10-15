import sys
sys.path.insert(0, '/Users/serkankose/Documents/quann-3d/quann_cloud/src')

from qcloud_network import QCloudNetwork
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
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3)
        for tid in range(1, 4):
            model.get_weight(tid)
        
        tokens = torch.tensor([[1, 2, 3]])
        mask = (tokens != 0).bool()
        
        stats = model.update_token_weights(tokens, mask, 4)
        self.assertIn('rewarded', stats)
        self.assertIn('censured', stats)

    def test_all_converged(self):
        model = QCloudNetwork(vocab_size=100, d_model=64, d_cloud_dim=3)
        self.assertFalse(model.all_converged())  # Boş → False

if __name__ == '__main__':
    unittest.main()
