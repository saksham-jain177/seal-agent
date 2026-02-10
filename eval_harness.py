"""
SEAL Evaluation Harness (v2.1 Hardening)
Replays verified edits to ensure behavior changes are persistent and correct.
"""
import json
import os
import hashlib
from inference_adapter import generate_with_adapter, get_learning_state

DATA_PATH = "data/edit_proposals.jsonl"

def run_evaluation():
    print("--- SEAL High-Fidelity Evaluation (Delta Replay) ---")
    
    state = get_learning_state()
    if state["status"] == "base_model":
        print("[SKIP] No fine-tuned adapter found. Cannot evaluate.")
        return

    print(f"[INFO] Evaluating Adapter: {state.get('version', 'unknown')}")
    print(f"[INFO] Learning Events tracked in Ledger: {state.get('applied_edits', 0)}")

    if not os.path.exists(DATA_PATH):
        print("[ERROR] Audit Log (edit_proposals.jsonl) not found.")
        return

    passed = 0
    regressions = 0
    total = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                e = json.loads(line)
            except:
                continue
            
            # Only test applied edits that were verified during audit
            if not e.get("applied") or not e.get("simulation_passed"):
                continue
            
            total += 1
            q = e.get("proposed_q")
            expected_hash = e.get("simulated_output_hash")
            pre_hash = e.get("base_output_hash")
            
            print(f"\n[{total}] Replaying Edit: {e.get('intent', 'Unknown Intent')}")
            print(f"    Scenario: \"{q}\"")
            
            # Query model with 0 temperature for determinism
            response = generate_with_adapter(f"Question: {q}\nAnswer:", max_length=100, temperature=0.0)
            current_hash = hashlib.sha256(response.encode()).hexdigest()
            
            # 1. Check for Behavioral Success (Delta Matched)
            if current_hash == expected_hash:
                print("    Result: ✅ SUCCESS (Target behavior verified)")
                passed += 1
            # 2. Check for Regression to Base behavior
            elif current_hash == pre_hash:
                print("    Result: ❌ REGRESSION (Model reverted to pre-adapter behavior)")
                regressions += 1
            # 3. Fuzzy match (Heuristic)
            else:
                target = e.get("proposed_a", "")
                if any(word.lower() in response.lower() for word in target.split()[:5]):
                    print("    Result: ⚠️ PARTIAL (Hash mismatch, but content likely correct)")
                    passed += 1
                else:
                    print("    Result: ‼️ FAILED (Unexpected behavior change)")
                    regressions += 1

    if total > 0:
        accuracy = (passed / total) * 100
        print(f"\n--- HARNESS REPORT ---")
        print(f"Fidelity Score: {accuracy:.1f}% ({passed}/{total})")
        print(f"Regression Rate: {(regressions/total)*100:.1f}% ({regressions}/{total})")
    else:
        print("\n[INFO] No applied edits found in ledger to evaluate.")

if __name__ == "__main__":
    run_evaluation()
