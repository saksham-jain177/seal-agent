import os
import json
import re
from typing import Dict, Any, Optional

# Local LLM
from langchain_ollama.chat_models import ChatOllama

# configurable paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "self_edits.jsonl")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "self_edits_reviewed.jsonl")

# threshold for automatic approval
APPROVAL_THRESHOLD = 0.70

def _extract_json(text: str):
    """Extract first valid JSON object from text output."""
    import json
    start = text.find("{")
    if start == -1:
        return None
    for end in range(len(text), start, -1):
        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except Exception:
            continue
    return None

def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to find and parse the first JSON object in `text`.
    Returns dict on success, otherwise None.
    """
    if not text or not isinstance(text, str):
        return None
    # direct parse attempt
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # fallback: find first {...} block
    matches = _extract_json(text)
    if not matches:
        return None
    return matches[0]

def model_review_prompt(entry: Dict[str, Any]) -> str:
    """
    Build a stable, deterministic prompt for scoring a saved self-edit entry.
    """
    q = entry.get("question", "")
    a = entry.get("answer", "")
    src = entry.get("source", "") or "unknown"
    created = entry.get("created_at", "")

    prompt = f"""
You are a strict technical reviewer. Evaluate the following Question / Answer pair for quality as a training example.
Respond ONLY with a single JSON object (no extra text) with these keys:
  - accuracy: float between 0.0 and 1.0 (how factually correct the answer is)
  - clarity: float between 0.0 and 1.0 (how clear and unambiguous the Q/A are)
  - novelty: float between 0.0 and 1.0 (how much new factual content this Q/A provides; not the same as wording)
  - score: float between 0.0 and 1.0 (weighted aggregate; weights: accuracy 0.5, clarity 0.3, novelty 0.2)
  - approved: boolean (true if score >= {APPROVAL_THRESHOLD})
  - remarks: one-sentence justification (max 120 chars)

Important constraints:
  - Produce numeric values with two decimal places (e.g., 0.75).
  - Do NOT include markdown, lists, or explanatory text — only the JSON object.
  - If you cannot judge accuracy due to missing evidence, put accuracy=0.50 and explain briefly in remarks.

Here is the item to evaluate (do not access the web; use the provided data only):
Question: {q}
Answer: {a}
Source: {src}
Created at: {created}

Output:
"""
    return prompt.strip()

def review_entry_with_llm(llm: ChatOllama, entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the local LLM to review a single entry and return a structured review dict.
    If parsing fails, include `debug_output` for manual inspection.
    """
    prompt = model_review_prompt(entry)

    # invoke model. Using llm.invoke(...) which returns an object with .content in this environment.
    try:
        resp = llm.invoke(prompt)
        # attempt to read textual content
        text = getattr(resp, "content", None)
        if text is None:
            # fallback: maybe resp is string-like
            text = str(resp)
    except Exception as e:
        return {"error": f"LLM invocation failed: {e}", "debug_output": None}

    parsed = extract_first_json(text)
    if parsed is None:
        # Save debug output for manual inspection
        return {"error": "failed to parse JSON from model response", "debug_output": text}

    # Normalize numeric fields and boolean
    try:
        accuracy = float(parsed.get("accuracy", 0.0))
        clarity = float(parsed.get("clarity", 0.0))
        novelty = float(parsed.get("novelty", 0.0))
    except Exception:
        # If types are wrong, reject with debug
        return {"error": "parsed JSON missing numeric fields or wrong type", "parsed": parsed, "debug_output": text}

    # Clip to [0.0, 1.0]
    def clip(v):
        v = float(v)
        return max(0.0, min(1.0, v))
    accuracy = round(clip(accuracy), 2)
    clarity = round(clip(clarity), 2)
    novelty = round(clip(novelty), 2)
    score = round(accuracy * 0.5 + clarity * 0.3 + novelty * 0.2, 2)
    approved = bool(parsed.get("approved", score >= APPROVAL_THRESHOLD))
    remarks = str(parsed.get("remarks", "")).strip()
    if len(remarks) > 200:
        remarks = remarks[:197].rstrip() + "..."

    return {
        "accuracy": accuracy,
        "clarity": clarity,
        "novelty": novelty,
        "score": score,
        "approved": approved,
        "remarks": remarks,
        "raw_model_output": text if parsed is None else None,
        "parsed_model_json": parsed
    }

def load_existing_reviewed(output_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load already-reviewed entries into an index keyed by a deterministic hash of question+answer.
    Returns dict: hash -> reviewed_record
    """
    index = {}
    if not os.path.exists(output_path):
        return index
    try:
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec.get("question","") + "\n" + rec.get("answer","")).strip()
                    index[key] = rec
                except Exception:
                    continue
    except Exception:
        pass
    return index

def append_reviewed_record(output_path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def main(input_path: str = DEFAULT_INPUT_PATH, output_path: str = DEFAULT_OUTPUT_PATH):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # prepare LLM (single instance)
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)

    # load index of already-reviewed to skip duplicates
    reviewed_index = load_existing_reviewed(output_path)

    processed = 0
    appended = 0
    skipped = 0
    failed = 0

    with open(input_path, "r", encoding="utf-8") as fh:
        for line in fh:
            processed += 1
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                failed += 1
                print(f"[WARN] Skipping invalid JSON line #{processed}")
                continue

            key = (entry.get("question","") + "\n" + entry.get("answer","")).strip()
            if key in reviewed_index:
                skipped += 1
                continue

            # run review
            review_result = review_entry_with_llm(llm, entry)

            # build output record
            output_record = {
                "question": entry.get("question"),
                "answer": entry.get("answer"),
                "source": entry.get("source", "unknown"),
                "created_at": entry.get("created_at"),
                "reviewed_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "review": review_result
            }

            # if review_result contains an 'error' key, mark as failed but still save for manual triage
            if isinstance(review_result, dict) and "error" in review_result:
                failed += 1
                output_record["review_status"] = "error"
            else:
                appended += 1
                output_record["review_status"] = "ok"

            append_reviewed_record(output_path, output_record)
            reviewed_index[key] = output_record

    print(f"Processed: {processed}, Appended: {appended}, Skipped(already reviewed): {skipped}, Failed: {failed}")
    print(f"Reviewed file: {output_path}")

if __name__ == "__main__":
    inp = os.getenv("REVIEW_INPUT", DEFAULT_INPUT_PATH)
    outp = os.getenv("REVIEW_OUTPUT", DEFAULT_OUTPUT_PATH)
    main(input_path=inp, output_path=outp)
