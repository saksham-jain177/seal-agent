"""
Talk to the SEAL Learner Model

This script allows you to chat directly with your fine-tuned 3B model.
It automatically loads any available LoRA adapters from adapters/lora_adapter.
"""
import sys
from inference_adapter import generate_with_adapter, is_adapter_available, get_learning_state

def main():
    print("--- SEAL Model Interaction ---")
    
    if not is_adapter_available():
        print("[WARNING] No fine-tuned adapter found. You are talking to the RAW base model.")
    else:
        state = get_learning_state()
        print(f"[INFO] Fine-tuned weights detected (Version: {state.get('adapter_version', '1')})")
        print(f"[INFO] Knowledge Base: {state.get('num_edits_used', 0)} impactful edits applied.")

    print("\nType 'exit' or 'quit' to stop.")
    
    while True:
        try:
            query = input("\nUser: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue

            print("Assistant: ", end="", flush=True)
            
            # Format prompt same as training
            prompt = f"Question: {query}\nAnswer:"
            
            response = generate_with_adapter(
                prompt,
                max_length=512,
                temperature=0.7 # Slight creativity for chat
            )
            
            if response:
                print(response)
            else:
                print("[ERROR] Model failed to generate response.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
