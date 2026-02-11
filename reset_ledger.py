import json
import os

DATA_PATH = "data/edit_proposals.jsonl"
LEDGER_PATH = "data/edit_ledger.json"

def reset():
    if not os.path.exists(DATA_PATH):
        print("No audit log found.")
        return

    # 1. Reset Audit Log
    temp_data = []
    reset_count = 0
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            e = json.loads(line)
            if e.get("simulation_passed") and e.get("applied"):
                e["applied"] = False
                reset_count += 1
            temp_data.append(e)
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        for e in temp_data:
            f.write(json.dumps(e) + "\n")
            
    print(f"[Reset] Un-applied {reset_count} edits in audit log.")

    # 2. Reset Ledger
    if os.path.exists(LEDGER_PATH):
        with open(LEDGER_PATH, "r") as f:
            ledger = json.load(f)
        
        ledger["applied_edits"] = 0
        ledger["adapter_version"] = ledger.get("adapter_version", 1) + 1
        
        with open(LEDGER_PATH, "w") as f:
            json.dump(ledger, f, indent=2)
        print(f"[Reset] Ledger applied_edits reset to 0.")

if __name__ == "__main__":
    reset()
