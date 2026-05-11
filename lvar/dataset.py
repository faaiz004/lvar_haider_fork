import random
from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from lvar.utils import extract_tagged_answer, normalize_answer_text

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised in environments without HF deps
    load_dataset = None


class CLEVRCoGenTDataset(Dataset):
    """Thin Dataset wrapper that exposes CLEVR CoGenT rows in LVAR-ready format."""

    def __init__(
        self,
        split: str = "train",
        limit: Optional[int] = None,
        dataset_name: str = "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex",
        partition: str = "all",
        test_fraction: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        """
        Load a Hugging Face split and optionally truncate it for quick experiments.

        Attributes:
            self.dataset: The underlying HF dataset split used for indexing.
        """
        if load_dataset is None:
            raise ImportError(
                "datasets is required to load MMInstruction/Clevr_CoGenT_TrainA_70K_Complex. "
                "Install the requirements first."
            )
        self.dataset = load_dataset(dataset_name, split=split)

        # Deterministic partitioning ensures reproducible, non-overlapping train/test subsets.
        if partition not in {"all", "train", "test"}:
            raise ValueError("partition must be one of: all, train, test")
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must be in the open interval (0, 1)")

        if partition != "all":
            total_size = len(self.dataset)
            all_indices = list(range(total_size))
            rng = random.Random(split_seed)
            rng.shuffle(all_indices)

            test_size = int(total_size * test_fraction)
            test_size = max(1, min(test_size, total_size - 1))
            test_indices = all_indices[:test_size]
            train_indices = all_indices[test_size:]
            selected_indices = train_indices if partition == "train" else test_indices
            self.dataset = self.dataset.select(selected_indices)

        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

    def __len__(self) -> int:
        """Return number of examples available after optional truncation."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return one example with normalized answer fields used by inference/training.

        Keys:
            id: Stable row id (or index fallback)
            image: PIL image object from the dataset
            question: CLEVR problem text
            solution: Raw solution text containing XML-like tags
            gold_answer: Normalized text extracted from <answer>...</answer>
        """
        row = self.dataset[index]
        solution = row["solution"]
        gold_answer = normalize_answer_text(extract_tagged_answer(solution))
        return {
            "id": row.get("id", index),
            "image": row["image"],
            "question": row["problem"],
            "solution": solution,
            "gold_answer": gold_answer,
        }
