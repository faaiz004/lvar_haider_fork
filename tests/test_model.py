import types
import unittest

import torch

from lvar.qwen_lvar import QwenLVAR
from lvar.utils import (
    ACTION_GLOBAL,
    ACTION_PATCH,
    ACTION_REGION,
    ACTION_STOP,
    ACTION_THINK,
)


class DummyTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        pieces = []
        for token_id in token_ids:
            if token_id == 1:
                pieces.append("<answer>yes</answer>")
            elif token_id == 0:
                pieces.append("</s>")
            else:
                pieces.append(f"tok{token_id}")
        return "".join(pieces)


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict, return_tensors):
        del messages, add_generation_prompt, tokenize, return_dict, return_tensors
        return {
            "input_ids": torch.tensor([[5, 999, 999, 999, 999, 6]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            "pixel_values": torch.tensor([[0.0]], dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        del skip_special_tokens
        return [self.tokenizer.decode(sequence) for sequence in sequences]


class DummyVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "image_tokens",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, pixel_values, grid_thw=None):
        del pixel_values, grid_thw
        return self.image_tokens.clone()


class DummyMergedGridProcessor(DummyProcessor):
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, return_dict, return_tensors):
        batch = super().apply_chat_template(messages, add_generation_prompt, tokenize, return_dict, return_tensors)
        batch["image_grid_thw"] = torch.tensor([[1, 4, 4]], dtype=torch.long)
        return batch


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4, image_token_id=999, eos_token_id=0)
        self.embedding = torch.nn.Embedding(1100, 4)
        self.visual = DummyVision()
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[5] = torch.tensor([0.5, 0.5, 0.0, 0.0])
            self.embedding.weight[6] = torch.tensor([0.0, 0.5, 0.5, 0.0])
            self.embedding.weight[1] = torch.tensor([0.0, 0.0, 0.5, 0.5])

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
        use_cache=False,
        **kwargs,
    ):
        del input_ids, attention_mask, output_hidden_states, return_dict, use_cache, kwargs
        if inputs_embeds is None:
            raise ValueError("DummyBackbone expects inputs_embeds in these tests.")
        seq_len = inputs_embeds.size(1)
        positions = torch.arange(seq_len, device=inputs_embeds.device, dtype=inputs_embeds.dtype).view(1, seq_len, 1)
        hidden = inputs_embeds + positions * 0.01
        logits = torch.zeros((1, seq_len, 4), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        logits[:, -1, 1] = 10.0
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=[inputs_embeds, hidden],
            last_hidden_state=hidden,
        )


def build_model():
    cfg = {
        "device": "cpu",
        "dtype": "float32",
        "max_steps": 3,
        "region_window": 1,
        "max_answer_tokens": 1,
        "action_selection": "argmax",
    }
    return QwenLVAR(cfg, backbone=DummyBackbone(), processor=DummyProcessor())


def build_merged_grid_model():
    cfg = {
        "device": "cpu",
        "dtype": "float32",
        "max_steps": 3,
        "region_window": 1,
        "max_answer_tokens": 1,
        "action_selection": "argmax",
    }
    return QwenLVAR(cfg, backbone=DummyBackbone(), processor=DummyMergedGridProcessor())


class QwenLVARTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        prepared = self.model.prepare_inputs("image", "question")
        projected = self.model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        self.prepared = prepared
        self.bank = self.model.build_visual_bank(projected)

    def _set_controller_action(self, action_id):
        def forward(latent_hidden, act_hidden, step_hidden, bank):
            del latent_hidden, act_hidden, step_hidden
            type_logits = torch.full((1, 5), -10.0)
            type_logits[0, action_id] = 10.0
            region_logits = torch.arange(bank["regions"].size(0), dtype=torch.float32).unsqueeze(0)
            patch_logits = torch.arange(bank["patches"].size(0), dtype=torch.float32).unsqueeze(0)
            return type_logits, region_logits, patch_logits

        self.model.controller.forward = forward

    def test_build_visual_bank_shapes(self):
        self.assertEqual(tuple(self.bank["patches"].shape), (4, 4))
        self.assertEqual(tuple(self.bank["regions"].shape), (4, 4))
        self.assertEqual(tuple(self.bank["global"].shape), (1, 4))

    def test_build_visual_bank_infers_merged_patch_grid(self):
        model = build_merged_grid_model()
        prepared = model.prepare_inputs("image", "question")
        projected = model.get_projected_image_tokens(prepared)
        bank = model.build_visual_bank(projected)
        self.assertEqual(tuple(model._current_image_grid), (2, 2))
        self.assertEqual(tuple(bank["patches"].shape), (4, 4))

    def test_build_visual_bank_pads_non_divisible_grid(self):
        model = build_model()
        model.region_window = 2
        model._current_image_grid = (3, 5)
        projected = torch.arange(60, dtype=torch.float32).view(15, 4)

        bank = model.build_visual_bank(projected)

        self.assertEqual(tuple(bank["patches"].shape), (15, 4))
        self.assertEqual(tuple(bank["regions"].shape), (6, 4))
        self.assertEqual(tuple(bank["global"].shape), (1, 4))

    def test_forward_reasoning_actions(self):
        for action_id in [ACTION_THINK, ACTION_STOP, ACTION_GLOBAL, ACTION_REGION, ACTION_PATCH]:
            with self.subTest(action_id=action_id):
                state = self.model.build_initial_state(self.prepared)
                initial_length = state["inputs_embeds"].size(1)
                initial_latent_pos = state["latent_pos"]
                initial_act_pos = state["act_pos"]
                initial_latent_embed = state["inputs_embeds"][:, initial_latent_pos, :].clone()
                initial_act_embed = state["inputs_embeds"][:, initial_act_pos, :].clone()
                self._set_controller_action(action_id)
                updated_state, selected_action, should_stop, step_trace = self.model.forward_reasoning_step(
                    state, self.bank, 0
                )
                self.assertEqual(selected_action, action_id)
                if action_id in [ACTION_GLOBAL, ACTION_REGION, ACTION_PATCH]:
                    self.assertEqual(updated_state["inputs_embeds"].size(1), initial_length + 1)
                    self.assertEqual(step_trace["sequence_length_after"], initial_length + 1)
                    self.assertEqual(updated_state["latent_pos"], initial_latent_pos + 1)
                    self.assertTrue(
                        torch.allclose(
                            updated_state["inputs_embeds"][:, updated_state["latent_pos"], :], initial_latent_embed
                        )
                    )
                    self.assertTrue(
                        torch.allclose(
                            updated_state["inputs_embeds"][:, updated_state["act_pos"], :], initial_act_embed
                        )
                    )
                elif action_id == ACTION_THINK:
                    self.assertEqual(step_trace["sequence_length_after"], initial_length)
                    self.assertFalse(
                        torch.allclose(updated_state["inputs_embeds"][:, initial_latent_pos, :], initial_latent_embed)
                    )
                    self.assertFalse(
                        torch.allclose(updated_state["inputs_embeds"][:, initial_act_pos, :], initial_act_embed)
                    )
                else:
                    self.assertEqual(step_trace["sequence_length_after"], initial_length)
                    self.assertTrue(
                        torch.allclose(updated_state["inputs_embeds"][:, initial_latent_pos, :], initial_latent_embed)
                    )
                    self.assertTrue(
                        torch.allclose(updated_state["inputs_embeds"][:, initial_act_pos, :], initial_act_embed)
                    )
                if action_id == ACTION_STOP:
                    self.assertTrue(should_stop)
                else:
                    self.assertFalse(should_stop)

    def test_drop_act_token_keeps_graph(self):
        state = self.model.build_initial_state(self.prepared)
        dropped = self.model.drop_act_token(state)
        self.assertTrue(dropped["inputs_embeds"].requires_grad)
        self.assertIsNotNone(dropped["inputs_embeds"].grad_fn)
        self.assertIsNone(dropped["act_pos"])

    def test_baseline_excludes_latent_and_act_tokens(self):
        baseline = self.model.baseline_forward("image", "question")
        self.assertEqual(baseline["decode_prefix_length"], self.prepared["input_ids"].size(1))
        self.assertEqual(baseline["num_steps"], 0)
        self.assertEqual(baseline["trace"], [])

    def test_generate_lvar_respects_action_selection(self):
        model = build_model()
        captured = {}

        def fake_forward(images, questions, labels=None, sample_actions=None):
            del images, questions, labels
            captured["sample_actions"] = sample_actions
            return {
                "answer": "yes",
                "trace": [],
                "num_steps": 0,
                "generated_text": "<answer>yes</answer>",
                "generated_ids": [1],
            }

        model.forward = fake_forward

        model.action_selection = "sample"
        model.generate_lvar("image", "question")
        self.assertTrue(captured["sample_actions"])

        model.action_selection = "argmax"
        model.generate_lvar("image", "question")
        self.assertFalse(captured["sample_actions"])


if __name__ == "__main__":
    unittest.main()
