"""
SEAL Evaluation Harness
Benchmarking tool to verify that compiled edits are correctly reflected in model behavior.
"""
import json
import os
from inference_adapter import generate_with_adapter, get_learning_state

DATA_PATH = "data/edit_proposals.jsonl"

def run_evaluation():
    print("--- SEAL Evaluation Harness ---")
    
    state = get_learning_state()
    if state["status"] == "base_model":
        print("[SKIP] No fine-tuned adapter found. Cannot evaluate.")
        return

    print(f"[INFO] Evaluating Adapter Version: {state.get('version')}")
    print(f"[INFO] Total Applied Edits to Verify: {state.get('applied_edits')}")

    if not os.path.exists(DATA_PATH):
        print("[ERROR] Audit Log (edit_proposals.jsonl) not found.")
        return

    passed = 0
    total = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            e = json.loads(line)
            
            # Only test applied edits
            if not e.get("applied") or not e.get("simulation_passed"):
                continue
            
            total += 1
            q = e.get("proposed_q")
            target = e.get("proposed_a")
            
            print(f"\n[{total}] Testing: {q}")
            response = generate_with_adapter(f"Question: {q}\nAnswer:", max_length=100, temperature=0.0)
            
            print(f"   Target: {target[:70]}...")
            print(f"   Model:  {response[:70]}...")

            # Simple heuristic: is the target knowledge present?
            # In a real harness, we'd use an 8B judge.
            if any(word.lower() in response.lower() for word in target.split()[:5]):
                print("   Result: ✅ LIKELY CORRECT")
                passed += 1
            else:
                print("   Result: ❌ POTENTIAL REGRESSION")

    if total > 0:
        accuracy = (passed / total) * 100
        print(f"\n--- FINAL RESULTS ---")
        print(f"Accuracy: {accuracy:.1f}% ({passed}/{total})")
    else:
        print("\n[INFO] No applied edits found to evaluate.")

if __name__ == "__main__":
    run_evaluation()
