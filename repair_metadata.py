import json
import os
from datetime import datetime

ADAPTER_DIR = "adapters/lora_adapter"
LEDGER_PATH = "data/edit_ledger.json"
META_PATH = os.path.join(ADAPTER_DIR, "adapter_meta.json")

def repair():
    if not os.path.exists(ADAPTER_DIR):
        print("Adapter directory not found.")
        return

    # Try to reconstruct from ledger
    num_applied = 0
    version = 1
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r") as f:
            ledger = json.load(f)
            num_applied = ledger.get("total_edits_applied", 0)
            version = ledger.get("adapter_version", 1)

    metadata = {
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "epochs": 1,
        "max_steps": 40,
        "num_edits_used": num_applied,
        "adapter_dir": ADAPTER_DIR,
        "timestamp": datetime.now().isoformat(),
        "adapter_version": version
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Successfully repaired metadata at {META_PATH}")

if __name__ == "__main__":
    repair()
