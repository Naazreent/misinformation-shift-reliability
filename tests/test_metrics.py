import unittest

import numpy as np

from misinformation_reliability.metrics import binary_metrics, expected_calibration_error


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        labels = np.array([0, 0, 1, 1])
        predictions = labels.copy()
        probabilities = np.array([0.05, 0.10, 0.90, 0.95])
        metrics = binary_metrics(labels, predictions, probabilities)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_ece_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            expected_calibration_error(np.array([0, 1]), np.array([0.2, 1.2]))


if __name__ == "__main__":
    unittest.main()

