import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUT_PATH = os.path.join(DATA_DIR, "self_edits.jsonl")
PROPOSALS_PATH = os.path.join(DATA_DIR, "edit_proposals.jsonl")
LEDGER_PATH = os.path.join(DATA_DIR, "edit_ledger.json")
INDEX_PATH = os.path.join(DATA_DIR, "self_edits_index.json")

def _hash_edit(edit: Dict[str, str]) -> str:
    """Deterministic hash of question+answer for simple dedupe."""
    key = (edit.get("question","") + "\n" + edit.get("answer","")).encode("utf-8")
    return hashlib.sha256(key).hexdigest()

def _load_index() -> Dict[str, bool]:
    """Load or create in-memory index of hashes to avoid duplicates."""
    if not os.path.exists(INDEX_PATH):
        return {}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_index(index: Dict[str, bool]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def append_edit_proposal(proposal: Dict) -> Tuple[str, bool]:
    """
    Appends a new EditProposal to the ledger file after checking for duplicates.
    """
    os.makedirs(os.path.dirname(PROPOSALS_PATH), exist_ok=True)
    
    # Simple deduplication based on Q/A content hash
    content = f"{proposal['proposed_q']}|{proposal['proposed_a']}".encode('utf-8')
    proposal_id = hashlib.sha256(content).hexdigest()
    
    # Check for existing
    if os.path.exists(PROPOSALS_PATH):
        with open(PROPOSALS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing = json.loads(line)
                        existing_content = f"{existing['proposed_q']}|{existing['proposed_a']}".encode('utf-8')
                        if hashlib.sha256(existing_content).hexdigest() == proposal_id:
                            return PROPOSALS_PATH, False
                    except:
                        continue

    with open(PROPOSALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(proposal) + "\n")
    
    # Update ledger count
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        ledger["total_edits_proposed"] = ledger.get("total_edits_proposed", 0) + 1
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
            
    return PROPOSALS_PATH, True

def update_ledger_applied_count(count: int):
    """Updates the ledger with the number of edits applied during a training run."""
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        ledger["total_edits_applied"] = ledger.get("total_edits_applied", 0) + count
        ledger["last_training_timestamp"] = datetime.utcnow().isoformat() + "Z"
        ledger["adapter_version"] = ledger.get("adapter_version", 1) + 1
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

def append_self_edit(edit: Dict[str, str]) -> Tuple[str, bool]:
    """
    Append a validated self-edit to the JSONL file.
    Returns (path, appended_bool). If duplicate found, it's skipped (returns False).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    idx = _load_index()
    h = _hash_edit(edit)

    if h in idx:
        return OUT_PATH, False  # duplicate, skip

    # prepare object to save: include minimal metadata
    save_obj = {
        "question": edit["question"],
        "answer": edit["answer"],
        "source": edit.get("source", "unknown"),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    # Append line
    with open(OUT_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(save_obj, ensure_ascii=False) + "\n")

    # Update index
    idx[h] = True
    _save_index(idx)

    return OUT_PATH, True

# quick manual test
if __name__ == "__main__":
    test = {"question": "What is X?", "answer": "X is Y.", "source": "test"}
    path, appended = append_self_edit(test)
    print("Saved to:", path, "appended?", appended)
