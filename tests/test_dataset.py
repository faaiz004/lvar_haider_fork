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


if __name__ == "__main__":
    unittest.main()
