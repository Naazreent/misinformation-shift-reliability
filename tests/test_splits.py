import unittest

import pandas as pd

from misinformation_reliability.data import add_text_groups
from misinformation_reliability.splits import random_stratified_split, source_disjoint_split


def synthetic_frame(rows=1200):
    sources = [f"SOURCE_{i}" for i in range(12)]
    records = []
    for index in range(rows):
        records.append(
            {
                "id": index + 1,
                "Statement": f"unique statement {index}",
                "Web": sources[index % len(sources)],
                "Label": index % 2,
            }
        )
    return add_text_groups(pd.DataFrame(records))


class SplitTests(unittest.TestCase):
    def test_random_split_is_deterministic(self):
        frame = synthetic_frame()
        first = random_stratified_split(frame, 42, 0.7, 0.15, 0.15)
        second = random_stratified_split(frame, 42, 0.7, 0.15, 0.15)
        self.assertTrue(first.assignments.equals(second.assignments))

    def test_source_split_has_no_source_overlap(self):
        frame = synthetic_frame()
        result = source_disjoint_split(frame, 42, 0.7, 0.15, 0.15)
        source_sets = {
            split: set(frame.loc[result.assignments == split, "Web"])
            for split in ("train", "validation", "test")
        }
        self.assertFalse(source_sets["train"] & source_sets["validation"])
        self.assertFalse(source_sets["train"] & source_sets["test"])
        self.assertFalse(source_sets["validation"] & source_sets["test"])

    def test_source_repetitions_use_distinct_candidate_blocks(self):
        frame = synthetic_frame()
        first = source_disjoint_split(frame, 13, 0.7, 0.15, 0.15)
        second = source_disjoint_split(frame, 42, 0.7, 0.15, 0.15)
        self.assertNotEqual(
            first.metadata["test_selection"]["selected_seed"],
            second.metadata["test_selection"]["selected_seed"],
        )
        self.assertTrue(
            set(first.metadata["sources"]["test"])
            != set(second.metadata["sources"]["test"])
            or set(first.metadata["sources"]["validation"])
            != set(second.metadata["sources"]["validation"])
        )


if __name__ == "__main__":
    unittest.main()
