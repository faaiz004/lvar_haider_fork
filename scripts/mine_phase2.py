import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Allow running as a script: `python scripts/mine_phase2.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.oracle_mining import OracleTraceMiner
from lvar.qwen_lvar import QwenLVAR
from lvar.utils import add_model_loading_args, apply_model_loading_overrides


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_jsonl_row(handle, row) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def collect_example_ids(dataset) -> list:
    ids = []
    for index in range(len(dataset)):
        ids.append(dataset[index].get("id", index))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine Phase 2 oracle traces from M3CoT.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    add_model_loading_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    phase2_cfg = config.get("phase2", {})
    dataset_cfg = config["dataset"]

    seed = int(args.seed if args.seed is not None else phase2_cfg.get("seed", config.get("train", {}).get("seed", 42)))
    set_seed(seed)

    limit = args.limit if args.limit is not None else phase2_cfg.get("limit", dataset_cfg.get("limit"))
    dataset = build_dataset(dataset_cfg, limit=limit, partition=phase2_cfg.get("dataset_partition"))
    output_path = Path(args.output or phase2_cfg.get("output_path", "outputs/phase2_m3cot_traces.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = QwenLVAR(config["model"])
    model.eval()
    miner = OracleTraceMiner(
        model=model,
        selection_delta=float(phase2_cfg.get("selection_delta", 0.03)),
        patch_k_choices=phase2_cfg.get("patch_k_choices", [1, 2, 3, 4]),
        max_steps=int(phase2_cfg.get("max_steps", config["model"].get("max_steps", 8))),
        rng=random.Random(seed),
        initial_visual_mode=str(phase2_cfg.get("initial_visual_mode", "global_mean")),
        image_size=phase2_cfg.get("image_size", 280),
    )

    example_ids = collect_example_ids(dataset)
    with open(output_path, "w", encoding="utf-8") as handle:
        for example in tqdm(dataset, total=len(dataset), desc="Mining Phase 2"):
            row = miner.mine_example(example, negative_global_example_ids=example_ids)
            write_jsonl_row(handle, row)

    summary = miner.get_summary()
    summary.update(
        {
            "output_path": str(output_path),
            "dataset_type": dataset_cfg.get("type"),
            "dataset_name": dataset_cfg.get("name"),
            "num_examples": len(dataset),
            "seed": seed,
        }
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    write_json(summary_path, summary)
    print(f"Wrote Phase 2 traces to {output_path}")
    print(f"Wrote Phase 2 summary to {summary_path}")


if __name__ == "__main__":
    main()
