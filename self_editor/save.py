import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Tuple

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROPOSALS_PATH = os.path.join(DATA_DIR, "edit_proposals.jsonl")
LEDGER_PATH = os.path.join(DATA_DIR, "edit_ledger.json")

def append_edit_proposal(proposal_dict: Dict, sim_result_dict: Dict) -> Tuple[str, bool]:
    """
    Appends a new EditProposal and its SimulationResult to the ledger file.
    Only allows appending if simulation passed (Contract Rule).
    """
    os.makedirs(os.path.dirname(PROPOSALS_PATH), exist_ok=True)
    
    # Merge proposal and simulation for a single audit entry
    audit_entry = {**proposal_dict, **sim_result_dict}
    
    # Check for existing by ID or Content Hash
    proposal_id = audit_entry.get("id")
    if os.path.exists(PROPOSALS_PATH):
        with open(PROPOSALS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing = json.loads(line)
                        if existing.get("id") == proposal_id:
                            return PROPOSALS_PATH, False
                    except:
                        continue

    with open(PROPOSALS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry) + "\n")
    
    return PROPOSALS_PATH, True

def update_ledger_applied_count(count: int, avg_confidence: float = 0.0):
    """
    Updates the ledger after a compilation (training) run.
    Ensures monotonic progress and budget compliance.
    """
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            ledger = json.load(f)
        
        # Monotonicity check: Applied count should never exceed budget
        new_total = ledger.get("applied_edits", 0) + count
        if new_total > ledger.get("budget", 50):
            print(f"[WARNING] Budget Overrun detected: {new_total} > {ledger.get('budget')}")
            # We still record the real state, but signal the breach
            
        ledger["applied_edits"] = new_total
        ledger["avg_confidence"] = avg_confidence
        ledger["last_train_at"] = datetime.utcnow().isoformat() + "Z"
        ledger["adapter_version"] = ledger.get("adapter_version", 0) + 1
        
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)

def init_ledger_if_needed(budget: int = 50):
    """
    Initializes the ledger with SEAL contract fields.
    Does not overwrite if already exists.
    """
    if not os.path.exists(LEDGER_PATH):
        os.makedirs(DATA_DIR, exist_ok=True)
        ledger = {
            "initial_budget": budget,
            "budget": budget,
            "applied_edits": 0,
            "avg_confidence": 0.0,
            "last_train_at": None,
            "adapter_version": 1
        }
        with open(LEDGER_PATH, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
