import json
import time
from self_editor.generate_selfedit import generate_self_edit
from self_editor.validate import validate_self_edit
from self_editor.save import append_self_edit

def populate_data():
    # List of topics to generate data for
    topics = [
        "What is the capital of France?",
        "Explain quantum entanglement",
        "How does photosynthesis work?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "Define machine learning",
        "What are black holes?",
        "How do vaccines work?",
        "What is the tallest mountain?",
        "Explain the theory of relativity",
        "What is DNA?",
        "How do airplanes fly?",
        "What is the difference between AI and ML?",
        "Who discovered penicillin?",
        "What causes earthquakes?"
    ]

    print(f"Starting batch generation for {len(topics)} topics...")

    for i, topic in enumerate(topics):
        print(f"\n[{i+1}/{len(topics)}] Processing: {topic}")
        
        # Generate
        raw_edit = generate_self_edit(topic=topic)
        
        if raw_edit:
            try:
                # Validate
                cleaned = validate_self_edit(raw_edit)
                
                # Save
                out_path, appended = append_self_edit(cleaned)
                if appended:
                    print(f"  ✅ Saved to {out_path}")
                else:
                    print("  ⚠️ Duplicate - skipped")
            except Exception as e:
                print(f"  ❌ Validation failed: {e}")
        else:
            print("  ❌ Generation failed")
            
        # Small pause to be nice to the system
        time.sleep(1)

    print("\nBatch generation complete! You can now run 'python tuner.py'")

if __name__ == "__main__":
    populate_data()
