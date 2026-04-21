import argparse
from pathlib import Path

import yaml

from lvar.dataset import CLEVRCoGenTDataset
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import format_trace_step


def load_config(config_path: str):
    """Load YAML config file used by all script entrypoints."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    """Run one example and print a human-readable LVAR reasoning trace."""
    # CLI inputs: config path plus optional explicit example index override.
    parser = argparse.ArgumentParser(description="Run a single LVAR debug example.")
    parser.add_argument("--config", default="configs/qwen2vl_lvar.yaml")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    debug_cfg = config.get("debug", {})
    index = args.index if args.index is not None else int(debug_cfg.get("index", 0))

    # Build dataset and model exactly like main inference path so debug reflects real behavior.
    dataset = CLEVRCoGenTDataset(
        split=dataset_cfg.get("split", "train"),
        limit=dataset_cfg.get("limit"),
        dataset_name=dataset_cfg.get("name", "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex"),
    )
    example = dataset[index]

    model = QwenLVAR(config["model"])
    lvar_output = model.generate_lvar(example["image"], example["question"])
    baseline_output = model.generate_baseline(example["image"], example["question"])

    # Print structured trace for quick inspection of action choices and loop length.
    print(f"Example ID: {example['id']}")
    print(f"Question: {example['question']}")
    print(f"Gold answer: {example['gold_answer']}")
    print("Reasoning trace:")
    for step in lvar_output["trace"]:
        print(f"  {format_trace_step(step)}")
    print(f"LVAR answer: {lvar_output['prediction']}")
    print(f"Baseline answer: {baseline_output['prediction']}")
    print(f"Reasoning steps: {lvar_output['num_steps']}")


if __name__ == "__main__":
    main()
