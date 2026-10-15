import sys
sys.path.insert(0, '/Users/serkankose/Documents/quann-3d/quann_cloud/src')

from qcloud_weight import QCloudWeight
import numpy as np
import unittest


class TestQCloudWeight(unittest.TestCase):
    def test_initialization(self):
        w = QCloudWeight(D=2)
        self.assertEqual(w.D, 2)
        self.assertTrue(np.all(-1.0 <= w.low))
        self.assertTrue(np.all(w.high <= 1.0))

    def test_narrow_reward(self):
        w = QCloudWeight(D=1)
        w.set_value(np.array([0.0]))  # Deterministic
        point = np.array([0.5])
        w.narrow(point, reward=True)
        # 0.5 > 0.0 → direction = 1
        self.assertEqual(w.direction[0], 1)
        self.assertTrue(w.low[0] <= 0.5 <= w.high[0])

    def test_narrow_censure(self):
        w = QCloudWeight(D=1)
        w.set_value(np.array([0.0]))  # Deterministic
        point = np.array([0.5])
        w.narrow(point, reward=False)
        # 0.5 > 0.0 → low = 0.5 (daralt)
        self.assertAlmostEqual(w.low[0], 0.5)
        self.assertEqual(w.direction[0], -1)

    def test_convergence(self):
        w = QCloudWeight(D=2, convergence_threshold=0.1)
        self.assertFalse(w.is_converged())

    def test_sample_within_bounds(self):
        w = QCloudWeight(D=3)
        s = w.sample()
        self.assertTrue(np.all(s >= w.low - 1e-6))
        self.assertTrue(np.all(s <= w.high + 1e-6))

if __name__ == '__main__':
    unittest.main()
