import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from lvar.dataset import CLEVRCoGenTDataset
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import baseline_correctness_reward, correctness_reward, delta_reward


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
    """Run full-dataset inference for both LVAR and baseline and save comparison rows."""
    # Allow script-level limit/output overrides while defaulting to config values.
    parser = argparse.ArgumentParser(description="Run LVAR and baseline inference on CLEVR CoGenT.")
    parser.add_argument("--config", default="configs/qwen2vl_lvar.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    inference_cfg = config.get("inference", {})

    dataset_limit = args.limit if args.limit is not None else dataset_cfg.get("limit")
    dataset = CLEVRCoGenTDataset(
        split=dataset_cfg.get("split", "train"),
        limit=dataset_limit,
        dataset_name=dataset_cfg.get("name", "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex"),
    )
    model = QwenLVAR(config["model"])

    # Evaluate one example at a time (v1 design) and collect metrics for JSONL output.
    rows = []
    for example in tqdm(dataset, total=len(dataset), desc="Inferring"):
        lvar_output = model.generate_lvar(example["image"], example["question"])
        baseline_output = model.generate_baseline(example["image"], example["question"])

        # Compute correctness and delta reward used later by GRPO analysis/training.
        lvar_correct = correctness_reward(lvar_output["prediction"], example["gold_answer"])
        base_correct = baseline_correctness_reward(baseline_output["prediction"], example["gold_answer"])
        row = {
            "example_id": example["id"],
            "question": example["question"],
            "gold_answer": example["gold_answer"],
            "lvar_prediction": lvar_output["prediction"],
            "baseline_prediction": baseline_output["prediction"],
            "lvar_correct": bool(lvar_correct),
            "baseline_correct": bool(base_correct),
            "delta_reward": delta_reward(
                lvar_output["prediction"],
                baseline_output["prediction"],
                example["gold_answer"],
            ),
            "reasoning_trace": lvar_output["trace"],
            "num_reasoning_steps": lvar_output["num_steps"],
        }
        rows.append(row)

    # Write all rows once at the end for a clean single output artifact.
    output_path = Path(args.output or inference_cfg.get("output_path", "outputs/clevr_predictions.jsonl"))
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} predictions to {output_path}")


if __name__ == "__main__":
    main()
