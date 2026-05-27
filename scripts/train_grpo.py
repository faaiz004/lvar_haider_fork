import argparse
import random
import sys
from pathlib import Path

import torch
import yaml

# Allow running as a script: `python scripts/train_grpo.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvar.dataset import build_dataset
from lvar.qwen_lvar import QwenLVAR
from lvar.rewards import correctness_reward
from lvar.utils import add_model_loading_args, apply_model_loading_overrides

try:
    from accelerate import Accelerator
except ImportError:  # pragma: no cover - exercised in environments without HF deps
    Accelerator = None


def load_config(config_path: str):
    """Load YAML config values shared across scripts."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set Python/Torch seeds for reproducible rollouts and optimization."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_group_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """
    Group-wise reward normalization used by GRPO-style advantage estimation.

    This matches the "relative within prompt group" intuition while guarding
    against zero-variance groups.
    """
    if rewards.numel() == 1:
        return rewards - rewards.mean()
    std = rewards.std(unbiased=False)
    if float(std.item()) == 0.0:
        return rewards - rewards.mean()
    return (rewards - rewards.mean()) / (std + 1e-8)

def asymmetric_baseline_weight(
        baseline_score: float,
        rollout_score: float,
        improve_weight: float = 1.5,
        miss_weight: float = 1.0,
        already_correct_weight: float = 0.5,
        regression_weight: float = 1.5,
    ) -> float:
        baseline_correct = baseline_score > 0.5
        rollout_correct = rollout_score > 0.5

        if not baseline_correct and rollout_correct:
            return improve_weight

        if not baseline_correct and not rollout_correct:
            return miss_weight

        if baseline_correct and rollout_correct:
            return already_correct_weight

        # baseline correct, rollout wrong
        return regression_weight


def trainable_state_dict(model: torch.nn.Module) -> dict:
    """Return only trainable LVAR/controller-facing parameters, excluding frozen backbone weights."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_controller_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Save a controller-only checkpoint to the requested path."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state_dict(model), checkpoint_path)
    print(f"Saved controller checkpoint to {checkpoint_path}")


def compute_grpo_policy_loss(advantages: torch.Tensor, rollout_outputs: list) -> torch.Tensor | None:
    """Build the policy-gradient loss from rollout action log-prob tensors."""
    loss_terms = []
    for advantage, rollout in zip(advantages, rollout_outputs):
        action_log_prob_sum = rollout.get("action_log_prob_sum")

        if action_log_prob_sum is None:
            continue

        action_loss = action_log_prob_sum / max(1, len(rollout["action_log_probs"]))
        loss_terms.append(-advantage.detach() * action_loss)

    if not loss_terms:
        return None
    return torch.stack(loss_terms).mean()


def main() -> None:
    """Train controller-facing LVAR parameters using custom grouped rollouts."""
    parser = argparse.ArgumentParser(description="Train the LVAR controller with custom GRPO-style updates.")
    parser.add_argument("--config", default="configs/qwen2vl_m3cot.yaml")
    add_model_loading_args(parser)
    args = parser.parse_args()

    if Accelerator is None:
        raise ImportError("accelerate is required for train_grpo.py. Install the requirements first.")

    config = load_config(args.config)
    config["model"] = apply_model_loading_overrides(config["model"], args)
    train_cfg = config["train"]
    dataset_cfg = config["dataset"]
    dataset_partition = train_cfg.get("dataset_partition", dataset_cfg.get("train_partition", "train"))
    split_seed = int(train_cfg.get("split_seed", dataset_cfg.get("split_seed", train_cfg.get("seed", 42))))
    test_fraction = float(train_cfg.get("test_fraction", dataset_cfg.get("test_fraction", 0.1)))

    set_seed(int(train_cfg.get("seed", 42)))
    accelerator = Accelerator()
    model = QwenLVAR(config["model"]).to(accelerator.device)
    model.train()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    dataset_options = dict(dataset_cfg)
    dataset_options["test_fraction"] = test_fraction
    dataset_options["split_seed"] = split_seed
    dataset = build_dataset(
        dataset_options,
        limit=train_cfg.get("max_examples", dataset_cfg.get("limit")),
        partition=dataset_partition,
    )

    output_dir = Path(train_cfg.get("output_dir", "outputs/train"))
    output_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = int(train_cfg.get("num_epochs", 1))
    group_size = int(train_cfg.get("group_size", 4))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 10))

    global_step = 0

    for epoch in range(num_epochs):
        for example in dataset:
            # ------------------------------------------------------------
            # 1. Compute no-latent baseline correctness once for this prompt.
            # ------------------------------------------------------------
            with torch.no_grad():
                baseline_output = model.baseline_forward(
                    example["image"],
                    example["question"],
                )
                baseline_score = correctness_reward(
                    baseline_output["answer"],
                    example["gold_answer"],
                )

            # Convert to a scalar Python float for prompt-level weighting.
            baseline_score_float = float(baseline_score)

            # ------------------------------------------------------------
            # 2. Sample grouped LVAR trajectories for the same prompt.
            # ------------------------------------------------------------
            rollout_outputs = []
            rewards = []

            for _ in range(group_size):
                rollout = model.forward(
                    example["image"],
                    example["question"],
                    sample_actions=True,
                )
                rollout_outputs.append(rollout)

                rollout_score = correctness_reward(
                    rollout["answer"],
                    example["gold_answer"],
                )
                rewards.append(float(rollout_score))

            # ------------------------------------------------------------
            # 3. Convert rewards into group-normalized advantages.
            #    Do NOT subtract baseline here because it cancels under
            #    per-prompt group normalization.
            # ------------------------------------------------------------
            reward_tensor = torch.tensor(
                rewards,
                device=accelerator.device,
                dtype=torch.float32,
            )

            advantages = normalize_group_rewards(reward_tensor)

            # Apply no-latent baseline as prompt-level advantage weight.
            # If baseline is wrong, amplify the preference signal.
            # If baseline is correct, dampen the preference signal.
            weights = torch.tensor([
                    asymmetric_baseline_weight(
                        baseline_score=baseline_score_float,
                        rollout_score=float(r),
                        improve_weight=float(train_cfg.get("improve_weight", 1.5)),
                        miss_weight=float(train_cfg.get("miss_weight", 1.0)),
                        already_correct_weight=float(train_cfg.get("already_correct_weight", 0.5)),
                        regression_weight=float(train_cfg.get("regression_weight", 1.5)),
                    )
                    for r in rewards
                ],
                device=accelerator.device,
                dtype=torch.float32,
            )
            advantages = advantages * weights

            # ------------------------------------------------------------
            # 4. Policy-gradient objective over sampled action log-probs.
            # ------------------------------------------------------------
            # action_log_prob_sum must be a differentiable tensor produced by
            # the controller during sampled rollout.
            loss = compute_grpo_policy_loss(advantages, rollout_outputs)
            if loss is None:
                continue

            loss_value = float(loss.detach().item())

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                grad_clip_norm,
            )

            optimizer.step()

            del rollout_outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            global_step += 1

            if global_step % log_every == 0 and accelerator.is_local_main_process:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss_value:.4f} "
                    f"reward_mean={float(reward_tensor.mean().item()):.4f} "
                    f"reward_std={float(reward_tensor.std(unbiased=False).item()):.4f} "
                    f"baseline_score={baseline_score_float:.1f} "
                )
        if accelerator.is_local_main_process:
            epoch_checkpoint_path = output_dir / f"lvar_controller_epoch_{epoch + 1}.pt"
            save_controller_checkpoint(model, epoch_checkpoint_path)

    # Save final weights from the main process only.
    if accelerator.is_local_main_process:
        checkpoint_path = output_dir / "lvar_controller.pt"
        save_controller_checkpoint(model, checkpoint_path)


if __name__ == "__main__":
    main()
