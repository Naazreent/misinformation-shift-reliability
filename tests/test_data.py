import unittest

import pandas as pd

from misinformation_reliability.data import deduplicate_statements, normalize_statement


class DataTests(unittest.TestCase):
    def test_normalization_is_stable(self):
        self.assertEqual(normalize_statement("  Fake—NEWS!!  "), "fake news")

    def test_conflicts_are_removed_and_same_label_duplicates_collapsed(self):
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "Statement": ["Same text", "same—text", "Other", "OTHER!!", "Unique"],
                "Image": ["x"] * 5,
                "Web": ["A", "B", "A", "B", "C"],
                "Category": ["C"] * 5,
                "Date": ["2020-01-01"] * 5,
                "Label": [0, 1, 1, 1, 0],
            }
        )
        result, stats = deduplicate_statements(frame)
        self.assertEqual(set(result["id"]), {3, 5})
        self.assertEqual(stats["conflicting_groups"], 1)
        self.assertEqual(stats["rows_removed_as_conflicting"], 2)
        self.assertEqual(stats["same_label_duplicate_rows_removed"], 1)


if __name__ == "__main__":
    unittest.main()

