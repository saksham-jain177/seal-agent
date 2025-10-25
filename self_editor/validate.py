# self_editor/validate.py
import re
from typing import Dict, Any

# Limits you can tune
MAX_QUESTION_LEN = 240
MAX_ANSWER_LEN = 4000

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")  # control chars

def _clean_text(s: str) -> str:
    """Trim, collapse whitespace, strip control characters."""
    if s is None:
        return ""
    s = str(s)
    s = _CONTROL_RE.sub("", s)
    # collapse multiple spaces/newlines into single space, but keep some newlines
    s = re.sub(r"\s+", " ", s).strip()
    return s

def validate_self_edit(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize a self-edit dict.
    Expected keys: 'question' and 'answer'. 'source' optional.
    Returns a cleaned dict with keys: question, answer, source.
    Raises ValueError for unrecoverable issues.
    """
    if not isinstance(obj, dict):
        raise ValueError("self-edit must be a dict")

    # Basic presence
    q = obj.get("question") or obj.get("q") or obj.get("Question")
    a = obj.get("answer") or obj.get("a") or obj.get("Answer")
    source = obj.get("source") or obj.get("src") or obj.get("Source") or ""

    if q is None or a is None:
        raise ValueError("self-edit missing 'question' or 'answer' fields")

    # Clean
    q_clean = _clean_text(q)
    a_clean = _clean_text(a)
    source_clean = _clean_text(source) or "unknown"

    # Length checks
    if len(q_clean) == 0:
        raise ValueError("question is empty after cleaning")
    if len(a_clean) == 0:
        raise ValueError("answer is empty after cleaning")

    if len(q_clean) > MAX_QUESTION_LEN:
        # truncate politely at whitespace boundary
        q_clean = q_clean[:MAX_QUESTION_LEN].rsplit(" ", 1)[0]

    if len(a_clean) > MAX_ANSWER_LEN:
        a_clean = a_clean[:MAX_ANSWER_LEN].rsplit(" ", 1)[0]

    cleaned = {
        "question": q_clean,
        "answer": a_clean,
        "source": source_clean
    }

    # Optional: simple heuristics to reject low-quality edits
    # e.g., if answer is just "Yes"/"No" or question is a duplicate-like junk
    if len(cleaned["question"].split()) < 3:
        # Too short question to be useful
        raise ValueError("question too short / not informative")

    if len(cleaned["answer"].split()) < 2:
        raise ValueError("answer too short / not informative")

    return cleaned

# quick manual test
if __name__ == "__main__":
    sample = {
        "question": " What is the Hubble constant? ",
        "answer": " ~ 67.4 km/s/Mpc (Planck) ",
        "source": "summary of searches"
    }
    print("Validated:", validate_self_edit(sample))
