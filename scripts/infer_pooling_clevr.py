import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Allow running as a script: `python scripts/infer_clevr.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import CLEVRCoGenTDataset
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import correctness_reward


def load_config(config_path: str):
    """Load YAML config file used to parameterize dataset/model/inference paths."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_jsonl(path: Path, rows):
    """Persist prediction rows as JSONL so each example is one appendable line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Run CLEVR inference comparing full-image and pooled-image decode baselines."""
    # Allow script-level limit/output overrides while defaulting to config values.
    parser = argparse.ArgumentParser(description="Compare CLEVR accuracy for full-image and pooled visual inputs.")
    parser.add_argument("--config", default="configs/qwen2vl_lvar.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})
    train_cfg = config.get("train", {})
    dataset_partition = inference_cfg.get("dataset_partition", dataset_cfg.get("eval_partition", "test"))
    split_seed = int(inference_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(inference_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))

    dataset_limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    dataset = CLEVRCoGenTDataset(
        split=dataset_cfg.get("split", "train"),
        limit=dataset_limit,
        dataset_name=dataset_cfg.get("name", "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex"),
        partition=dataset_partition,
        test_fraction=test_fraction,
        split_seed=split_seed,
    )
    model = QwenLVAR(config["model"])

    # Evaluate one example at a time (v1 design) and collect metrics for JSONL output.
    rows = []
    totals = {
        "full_image_correct": 0,
        "mean_pooled_correct": 0,
        "max_pooled_correct": 0,
    }
    for example in tqdm(dataset, total=len(dataset), desc="Inferring"):
        full_image_output = model.generate_baseline(example["image"], example["question"])
        mean_pooled_output = model.generate_pooled_baseline(example["image"], example["question"], pooling="mean")
        max_pooled_output = model.generate_pooled_baseline(example["image"], example["question"], pooling="max")

        full_image_correct = correctness_reward(full_image_output["prediction"], example["gold_answer"])
        mean_pooled_correct = correctness_reward(mean_pooled_output["prediction"], example["gold_answer"])
        max_pooled_correct = correctness_reward(max_pooled_output["prediction"], example["gold_answer"])
        totals["full_image_correct"] += int(full_image_correct)
        totals["mean_pooled_correct"] += int(mean_pooled_correct)
        totals["max_pooled_correct"] += int(max_pooled_correct)
        row = {
            "example_id": example["id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "full_image_prediction": full_image_output["prediction"],
            "mean_pooled_prediction": mean_pooled_output["prediction"],
            "max_pooled_prediction": max_pooled_output["prediction"],
            "full_image_correct": bool(full_image_correct),
            "mean_pooled_correct": bool(mean_pooled_correct),
            "max_pooled_correct": bool(max_pooled_correct),
            "full_image_generated_text": full_image_output["generated_text"],
            "mean_pooled_generated_text": mean_pooled_output["generated_text"],
            "max_pooled_generated_text": max_pooled_output["generated_text"],
        }
        rows.append(row)

    # Write all rows once at the end for a clean single output artifact.
    output_path = Path(args.output or inference_cfg.get("output_path", "outputs/clevr_predictions.jsonl"))
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")
    if rows:
        count = len(rows)
        print("Accuracy summary:")
        print(f"  full_image: {totals['full_image_correct'] / count:.4f} ({totals['full_image_correct']}/{count})")
        print(f"  mean_pooled: {totals['mean_pooled_correct'] / count:.4f} ({totals['mean_pooled_correct']}/{count})")
        print(f"  max_pooled: {totals['max_pooled_correct'] / count:.4f} ({totals['max_pooled_correct']}/{count})")


if __name__ == "__main__":
    main()
