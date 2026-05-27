import random
import unittest

from lvar.oracle_mining import (
    Candidate,
    OracleTraceMiner,
    build_step_target,
    group_steps_to_max,
    preprocess_reasoning_steps,
    split_rationale_into_sentences,
)
from test_model import build_model


class OracleMiningTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        prepared = self.model.prepare_inputs("image", "question")
        projected = self.model.get_projected_image_tokens(prepared)
        prepared["projected_image_tokens"] = projected
        self.bank = self.model.build_visual_bank(projected)

    def test_step_target_excludes_current_step(self):
        target = build_step_target(["look at the object", "identify the part", "choose answer"], 0, "C")

        self.assertNotIn("look at the object", target)
        self.assertIn("identify the part", target)
        self.assertTrue(target.endswith("Therefore, the answer is C"))

    def test_reasoning_preprocessing_merges_long_rationales(self):
        rationale = " ".join(f"Sentence {idx}." for idx in range(12))
        steps = preprocess_reasoning_steps({"rationale": rationale}, max_steps=8)

        self.assertEqual(len(steps), 8)
        self.assertIn("Sentence 0.", steps[0])
        self.assertIn("Sentence 11.", steps[-1])

    def test_sentence_split_and_group_match_training_preprocess_shape(self):
        sentences = split_rationale_into_sentences("First step. Second step? Third step!")
        grouped = group_steps_to_max(sentences, 2)

        self.assertEqual(sentences, ["First step.", "Second step?", "Third step!"])
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0], "First step.")
        self.assertEqual(grouped[1], "Second step? Third step!")

    def test_selection_threshold_keeps_noop_below_delta(self):
        miner = OracleTraceMiner(self.model, selection_delta=0.03)
        below = {
            "NO_OP": Candidate("NO_OP", [], 1.0),
            "THINK": Candidate("THINK", [{"type": "THINK"}], 0.98),
        }
        above = {
            "NO_OP": Candidate("NO_OP", [], 1.0),
            "THINK": Candidate("THINK", [{"type": "THINK"}], 0.96),
        }

        self.assertEqual(miner.select_candidate(below).name, "NO_OP")
        self.assertEqual(miner.select_candidate(above).name, "THINK")

    def test_patch_candidate_preserves_order_and_optional_think(self):
        miner = OracleTraceMiner(self.model, patch_k_choices=[2], rng=random.Random(0))
        state = self.model.build_coarse_initial_state(self.model.prepare_inputs("image", "question"), self.bank)

        def fake_score_actions(state, bank, target_text, name, actions):
            del state, bank, target_text
            ce = 10.0 - len(actions)
            if name == "PATCH_SEQ_THINK":
                ce = 0.0
            return Candidate(name, [dict(action) for action in actions], ce)

        miner.score_actions = fake_score_actions
        candidate = miner._best_patch_candidate(state, self.bank, "target")

        self.assertEqual(candidate.name, "PATCH_SEQ_THINK")
        self.assertEqual(candidate.actions[0], {"type": "PATCH", "patch_idx": 0})
        self.assertEqual(candidate.actions[1], {"type": "PATCH", "patch_idx": 1})
        self.assertEqual(candidate.actions[2], {"type": "THINK"})

    def test_counterfactual_preserves_structure_and_swaps_patch_ids(self):
        miner = OracleTraceMiner(self.model, rng=random.Random(0))
        positive = [
            {"type": "PATCH", "patch_idx": 0},
            {"type": "PATCH", "patch_idx": 1},
            {"type": "THINK"},
        ]

        pair = miner.build_counterfactual_pair(
            "ex-1",
            0,
            [],
            positive,
            "target",
            self.bank,
            negative_global_example_ids=["ex-1", "ex-2"],
        )

        self.assertIsNotNone(pair)
        self.assertEqual([action["type"] for action in pair["negative_actions"]], ["PATCH", "PATCH", "THINK"])
        self.assertNotIn(pair["negative_actions"][0]["patch_idx"], {0, 1})
        self.assertNotIn(pair["negative_actions"][1]["patch_idx"], {0, 1})

    def test_mined_trace_omits_noop_and_appends_stop(self):
        miner = OracleTraceMiner(self.model, selection_delta=0.03)

        def fake_candidates(state, bank, target_text):
            del state, bank, target_text
            return {
                "NO_OP": Candidate("NO_OP", [], 1.0),
                "THINK": Candidate("THINK", [{"type": "THINK"}], 0.99),
                "GLOBAL": Candidate("GLOBAL", [{"type": "GLOBAL"}], 0.99),
                "REGION": Candidate("REGION", [{"type": "REGION", "region_idx": 0}], 0.99),
                "PATCH": Candidate("PATCH_SEQ", [{"type": "PATCH", "patch_idx": 0}], 0.99),
            }

        miner.score_step_candidates = fake_candidates
        row = miner.mine_example(
            {
                "id": "ex-1",
                "image": "image",
                "question": "question",
                "solution": "First. Second.\n<answer>A</answer>",
            },
            negative_global_example_ids=["ex-1", "ex-2"],
        )

        self.assertEqual(row["question"], "question")
        self.assertEqual(row["answer"], "A")
        self.assertEqual(row["trace"], [{"type": "STOP"}])
        self.assertEqual([decision["selected"] for decision in row["decisions"]], ["NO_OP", "NO_OP"])

    def test_target_tokenization_appends_eos_like_training_collator(self):
        miner = OracleTraceMiner(self.model)
        token_ids = miner._tokenize_target("Therefore, the answer is A")

        self.assertEqual(int(token_ids[0, -1].item()), self.model.processor.tokenizer.eos_token_id)


if __name__ == "__main__":
    unittest.main()
