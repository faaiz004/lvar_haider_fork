import unittest
from unittest.mock import patch

from lvar.dataset import CLEVRCoGenTDataset
from lvar.rewards import normalize_answer


class FakeHFDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return FakeHFDataset([self.rows[index] for index in indices])


class DatasetTests(unittest.TestCase):
    def test_answer_normalization(self):
        self.assertEqual(normalize_answer("<answer> 3 </answer>"), "3")
        self.assertEqual(normalize_answer("<answer> Yes </answer>"), "yes")

    @patch("lvar.dataset.load_dataset")
    def test_dataset_mapping(self, mock_load_dataset):
        rows = [
            {
                "id": "sample-1",
                "image": "fake-image",
                "problem": "How many objects are there?",
                "solution": "<answer> 3 </answer>",
            }
        ]
        mock_load_dataset.return_value = FakeHFDataset(rows)

        dataset = CLEVRCoGenTDataset(limit=1)
        example = dataset[0]

        self.assertEqual(example["id"], "sample-1")
        self.assertEqual(example["image"], "fake-image")
        self.assertEqual(example["question"], "How many objects are there?")
        self.assertEqual(example["solution"], "<answer> 3 </answer>")
        self.assertEqual(example["gold_answer"], "3")

    @patch("lvar.dataset.load_dataset")
    def test_train_test_partitions_are_deterministic_and_disjoint(self, mock_load_dataset):
        rows = [
            {
                "id": f"sample-{index}",
                "image": "fake-image",
                "problem": f"Question {index}",
                "solution": "<answer> yes </answer>",
            }
            for index in range(20)
        ]
        mock_load_dataset.return_value = FakeHFDataset(rows)

        train_dataset = CLEVRCoGenTDataset(partition="train", test_fraction=0.25, split_seed=7)
        test_dataset = CLEVRCoGenTDataset(partition="test", test_fraction=0.25, split_seed=7)

        train_ids = {example["id"] for example in train_dataset}
        test_ids = {example["id"] for example in test_dataset}

        self.assertEqual(len(train_ids.intersection(test_ids)), 0)
        self.assertEqual(len(train_dataset) + len(test_dataset), len(rows))

        repeated_train_dataset = CLEVRCoGenTDataset(partition="train", test_fraction=0.25, split_seed=7)
        repeated_train_ids = {example["id"] for example in repeated_train_dataset}
        self.assertEqual(train_ids, repeated_train_ids)


if __name__ == "__main__":
    unittest.main()
