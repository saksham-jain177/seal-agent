import json
from inference_adapter import generate_with_adapter

DATA_PATH = "data/edit_proposals.jsonl"

def verify():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    
    # Filter for applied edits
    applied = [e for e in entries if e.get("simulation_passed")]
    
    print(f"--- SEAL Phase 9 Bulk Verification ({len(applied)} checks) ---")
    
    for i, e in enumerate(applied):
        q = e.get("proposed_q")
        target = e.get("proposed_a")
        
        print(f"\n[{i+1}] Query: {q}")
        print(f"Target: {target[:70]}...")
        
        response = generate_with_adapter(
            f"Question: {q}\nAnswer:",
            max_length=150,
            temperature=0.0
        )
        
        print(f"Actual: {response}")
        print("-" * 50)

if __name__ == "__main__":
    verify()
