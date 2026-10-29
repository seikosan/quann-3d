from quann_cloud.src.qcloud_weight import QCloudWeight
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
        """Test FIXED narrow_search censure semantics.
        
        Per narrow_search (line 176-179):
        - Kötü deneme: value DEĞİŞMEZ; aralık, kötü noktanın dışını keser
        - 0.5 > 0.0 → high = min(high, 0.5) = 0.5 (üst kısmı kes)
        - value remains 0.0 (FIXED: old code updated value to bad point)
        """
        w = QCloudWeight(D=1)
        w.set_value(np.array([0.0]))  # Deterministic
        point = np.array([0.5])
        w.narrow(point, reward=False)
        # FIXED: value unchanged, high cut at bad point
        self.assertEqual(w.value[0], 0.0)  # value unchanged
        self.assertAlmostEqual(w.high[0], 0.5)  # upper bound cut at bad point
        self.assertEqual(w.direction[0], -1)  # direction reversed

    def test_convergence(self):
        w = QCloudWeight(D=2, convergence_threshold=0.1)
        self.assertFalse(w.is_converged())

    def test_sample_within_bounds(self):
        w = QCloudWeight(D=3)
        s = w.sample()
        self.assertTrue(np.all(s >= w.low - 1e-6))
        self.assertTrue(np.all(s <= w.high + 1e-6))

    def test_sample_uses_direction_and_sigma(self):
        w = QCloudWeight(D=1, sigma=0.5)
        w.set_value(np.array([0.0]))
        np.random.seed(5)
        positive = w.sample()
        w.direction[:] = -1
        np.random.seed(5)
        negative = w.sample()
        self.assertGreater(positive[0], 0.0)
        self.assertLess(negative[0], 0.0)

        w.direction[:] = 1
        w.sigma = 0.1
        np.random.seed(5)
        small_step = w.sample()[0]
        w.sigma = 0.8
        np.random.seed(5)
        large_step = w.sample()[0]
        self.assertGreater(large_step, small_step)

    def test_zero_convergence_threshold_is_respected(self):
        w = QCloudWeight(D=1)
        w.low[:] = 0.0
        w.high[:] = 0.0
        self.assertTrue(w.is_converged(threshold=0))

if __name__ == '__main__':
    unittest.main()
