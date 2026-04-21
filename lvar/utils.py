import re
from typing import Dict, Iterable, List, Optional

# Discrete action ids used by the controller and trace utilities.
ACTION_THINK = 0
ACTION_STOP = 1
ACTION_GLOBAL = 2
ACTION_REGION = 3
ACTION_PATCH = 4

# Stable id->name and name->id maps used by model code and debug output.
ACTION_NAMES: Dict[int, str] = {
    ACTION_THINK: "THINK",
    ACTION_STOP: "STOP",
    ACTION_GLOBAL: "GLOBAL",
    ACTION_REGION: "REGION",
    ACTION_PATCH: "PATCH",
}
ACTION_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in ACTION_NAMES.items()}

# Regex used everywhere to extract the final tagged answer from model text.
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def extract_tagged_answer(text: Optional[str]) -> str:
    """Extract the content inside <answer>...</answer>; fallback to stripped raw text."""
    if not text:
        return ""
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_answer_text(text: Optional[str]) -> str:
    """Normalize answers for reward/eval by lowercasing and collapsing whitespace."""
    extracted = extract_tagged_answer(text)
    normalized = re.sub(r"\s+", " ", extracted).strip().lower()
    return normalized


def format_trace_step(step_trace: Dict[str, object]) -> str:
    """Render one reasoning step dict into a compact human-readable debug line."""
    action = step_trace.get("action", "UNKNOWN")
    parts: List[str] = [
        f"step={step_trace.get('step_idx', '?')}",
        f"action={action}",
        f"stop={step_trace.get('should_stop', False)}",
        f"seq_len={step_trace.get('sequence_length_after', step_trace.get('sequence_length_before', '?'))}",
    ]
    if step_trace.get("region_index") is not None:
        parts.append(f"region={step_trace['region_index']}")
    if step_trace.get("patch_index") is not None:
        parts.append(f"patch={step_trace['patch_index']}")
    return " | ".join(parts)


def format_trace(trace: Iterable[Dict[str, object]]) -> str:
    """Render an entire action trace as multi-line text for CLI debugging."""
    return "\n".join(format_trace_step(step) for step in trace)
