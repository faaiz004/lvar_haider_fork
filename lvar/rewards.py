from lvar.utils import normalize_answer_text


def normalize_answer(answer: str) -> str:
    """Shared normalization entrypoint used by all reward calculations."""
    return normalize_answer_text(answer)


def correctness_reward(prediction: str, gold_answer: str) -> float:
    """Return 1.0 when normalized prediction equals normalized gold, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold_answer))


def baseline_correctness_reward(prediction: str, gold_answer: str) -> float:
    """Baseline correctness mirror kept separate for clarity in delta reward code."""
    return correctness_reward(prediction, gold_answer)


def delta_reward(lvar_prediction: str, baseline_prediction: str, gold_answer: str) -> float:
    """Compute R_delta = R_lvar - R_base used by controller policy optimization."""
    lvar_score = correctness_reward(lvar_prediction, gold_answer)
    baseline_score = baseline_correctness_reward(baseline_prediction, gold_answer)
    return lvar_score - baseline_score
