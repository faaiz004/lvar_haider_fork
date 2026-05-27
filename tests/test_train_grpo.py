import unittest

import torch

from lvar.utils import ACTION_REGION
from scripts.train_grpo import compute_grpo_policy_loss
from test_model import build_model


class GRPOTrainingTests(unittest.TestCase):
    def test_region_rollout_with_raw_patches_produces_differentiable_policy_loss(self):
        model = build_model(controller_context_window=1, max_steps=1, region_window=2)
        model.train()

        with torch.no_grad():
            for parameter in model.controller.parameters():
                parameter.zero_()
            model.controller.type_head.bias.fill_(-10.0)
            model.controller.type_head.bias[ACTION_REGION] = 10.0

        def fake_decode(state, labels=None):
            del labels
            return {
                "answer": "yes",
                "generated_text": "<answer>yes</answer>",
                "generated_ids": [1],
                "decode_prefix_length": state["inputs_embeds"].size(1),
                "final_sequence_length": state["inputs_embeds"].size(1),
            }

        model.decode_answer = fake_decode
        rollout = model.forward("image", "question", sample_actions=True)

        self.assertEqual(rollout["trace"][0]["action_id"], ACTION_REGION)
        self.assertEqual(
            rollout["trace"][0]["sequence_length_after"] - rollout["trace"][0]["sequence_length_before"],
            4,
        )
        self.assertTrue(rollout["action_log_prob_sum"].requires_grad)

        loss = compute_grpo_policy_loss(torch.tensor([1.0]), [rollout])
        self.assertIsNotNone(loss)
        loss.backward()

        self.assertIsNotNone(model.controller.type_head.bias.grad)
        self.assertTrue(torch.isfinite(model.controller.type_head.bias.grad).all())

    def test_grpo_policy_loss_returns_none_without_action_log_probs(self):
        loss = compute_grpo_policy_loss(torch.tensor([1.0]), [{"action_log_prob_sum": None}])

        self.assertIsNone(loss)


if __name__ == "__main__":
    unittest.main()
