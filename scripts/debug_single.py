import argparse
import random
import sys
from pathlib import Path

import yaml

# Allow running as a script: `python scripts/debug_single.py ...`.
# Direct execution otherwise adds only `scripts/` to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    # Build dataset and model exactly like main inference path so debug reflects real behavior.
    dataset = CLEVRCoGenTDataset(
        split=dataset_cfg.get("split", "train"),
        limit=dataset_cfg.get("limit"),
        dataset_name=dataset_cfg.get("name", "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex"),
    )
    index = args.index if args.index is not None else random.randrange(len(dataset))
    example = dataset[index]

    model = QwenLVAR(config["model"])
    uses_sampling = model._inference_uses_sampling()
    lvar_output = model.generate_lvar(example["image"], example["question"])
    # baseline_output = model.generate_baseline(example["image"], example["question"])

    # Print structured trace for quick inspection of action choices and loop length.
    print(f"Example ID: {example['id']}")
    print(
        "Sampling strategy: "
        f"{model.action_selection} "
        f"({'stochastic sampling' if uses_sampling else 'deterministic argmax'})"
    )
    print(f"Question: {example['question']}")
    print(f"Gold answer: {example['gold_answer']}")
    print("Reasoning trace:")
    for step in lvar_output["trace"]:
        print(f"  {format_trace_step(step)}")
    print(f"LVAR answer: {lvar_output['prediction']}")
    print(f"Generated token IDs: {lvar_output['generated_ids']}")
    # print(f"Baseline answer: {baseline_output['prediction']}")
    # print(f"Reasoning steps: {lvar_output['num_steps']}")


if __name__ == "__main__":
    main()
