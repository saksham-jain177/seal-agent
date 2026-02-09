import os
import json
import torch
import warnings
import logging
from typing import Tuple
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

import argparse
from datetime import datetime

# ---- Silence Noisy Logs ----
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# ---- Configuration ----
# Default to Llama-3.2-3B for 6GB VRAM compatibility
DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct" 
DATA_PATH = "data/edit_proposals.jsonl"
LEDGER_PATH = "data/edit_ledger.json"
ADAPTER_OUTPUT_DIR = "adapters/lora_adapter"
# Optimized for faster training (reduced from 100 steps)
MAX_STEPS = 40        # Reduced for faster iterations (was 100)
BATCH_SIZE = 2
EPOCHS = 1            # Reduced from 2 for speed
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Incremental training: only train when we have enough new data
MIN_NEW_EDITS_FOR_TRAINING = 10  # Minimum new Q&A pairs before training

# ---- Dataset loader ----
def load_dataset(path):
    if not os.path.exists(LEDGER_PATH):
        return Dataset.from_list([])
        
    with open(LEDGER_PATH, "r") as f:
        ledger = json.load(f)
    
    risk_threshold = ledger.get("risk_threshold", 0.4)
    min_confidence = ledger.get("min_confidence", 0.7)
    budget = ledger.get("current_budget", 50)
    
    with open(path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    
    pairs = []
    applied_count = 0
    
    for e in entries:
        if applied_count >= budget:
            break
            
        # SEAL Filtering: Only impactful, low-risk, high-confidence, un-applied edits
        if (e.get("simulation_passed") and 
            not e.get("applied") and
            e.get("risk_score", 1.0) <= risk_threshold and
            e.get("confidence", 0.0) >= min_confidence):
            
            q = e.get("proposed_q")
            a = e.get("proposed_a")
            if q and a:
                pairs.append({"prompt": f"Question: {q}\nAnswer:", "response": a})
                applied_count += 1
                
    return Dataset.from_list(pairs)

# ---- Tokenization ----
def tokenize(examples, tokenizer):
    prompts = examples["prompt"]
    responses = examples["response"]
    
    inputs = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding=False # We will pad manually/via collator if needed, but here we do it simple
    )
    
    # Now tokenized the FULL thing (Prompt + Response)
    combined_texts = [p + " " + r + tokenizer.eos_token for p, r in zip(prompts, responses)]
    full_tokens = tokenizer(
        combined_texts,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    labels = []
    for i, prompt in enumerate(prompts):
        # Find where the prompt ends
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        label = full_tokens["input_ids"][i].copy()
        # Mask the prompt part with -100
        label[:prompt_len] = [-100] * prompt_len
        # Also mask padding
        padding_start = (full_tokens["attention_mask"][i] == 0).nonzero()
        if len(padding_start) > 0:
            idx = padding_start[0].item()
            label[idx:] = [-100] * (512 - idx)
        labels.append(label)
        
    full_tokens["labels"] = labels
    return full_tokens

# ---- Incremental Training Check ----
def should_train(min_new_edits: int = MIN_NEW_EDITS_FOR_TRAINING) -> Tuple[bool, int]:
    """
    Check if we have enough approved/simulate-passed edits to justify training.
    """
    if not os.path.exists(DATA_PATH) or not os.path.exists(LEDGER_PATH):
        return False, 0
    
    with open(LEDGER_PATH, "r") as f:
        ledger = json.load(f)
    
    risk_threshold = ledger.get("risk_threshold", 0.4)
    min_confidence = ledger.get("min_confidence", 0.7)
    
    ready_count = 0
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if (e.get("simulation_passed") and 
                    not e.get("applied") and
                    e.get("risk_score", 1.0) <= risk_threshold and
                    e.get("confidence", 0.0) >= min_confidence):
                    ready_count += 1
    
    should = ready_count >= min_new_edits
    return should, ready_count

# ---- Main ----
def main():
    parser = argparse.ArgumentParser(description="Fine-tune the SEAL agent model.")
    parser.add_argument("--force", "-f", action="store_true", help="Force training regardless of data count.")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Base model to use for training.")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        print(f"[SKIP] No dataset found at {DATA_PATH}. Run 'python research_agent.py' to collect impactful edits first.")
        return
    
    # Check if we should train
    if not args.force:
        should, new_count = should_train()
        if not should:
            print(f"[SKIP] Only {new_count} new edits. Need {MIN_NEW_EDITS_FOR_TRAINING} minimum.")
            print(f"[SKIP] Use 'python tuner.py --force' to train anyway, or collect more data.")
            return
        print(f"[INFO] Found {new_count} new edits. Proceeding with training...")

    os.makedirs(ADAPTER_OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, 
            use_fast=True, # Fast tokenizer is usually better for Llama-3
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[ERROR] AutoTokenizer.from_pretrained failed: {e}")
        # Fallback to local files if any, or raise
        return

    if tokenizer is None or isinstance(tokenizer, bool):
        print(f"[ERROR] AutoTokenizer returned invalid object: {tokenizer}")
        return

    # Standard Llama-3 padding setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # Check if adapter exists - if so, load it for continued training
    adapter_exists = os.path.exists(os.path.join(ADAPTER_OUTPUT_DIR, "adapter_model.safetensors"))
    
    if adapter_exists:
        try:
            with open(os.path.join(ADAPTER_OUTPUT_DIR, "adapter_config.json"), "r") as f:
                config = json.load(f)
            
            # Check for model mismatch
            saved_base = config.get("base_model_name_or_path", "")
            if saved_base and saved_base != args.model:
                print(f"[WARNING] Existing adapter was trained on {saved_base}, but you are using {args.model}.")
                print(f"[ACTION] Please delete the '{ADAPTER_OUTPUT_DIR}' folder to start fresh with the new architecture.")
                return

            print(f"[INFO] Loading existing adapter from {ADAPTER_OUTPUT_DIR} for continued training...")
            model = prepare_model_for_kbit_training(model)
            model = PeftModel.from_pretrained(model, ADAPTER_OUTPUT_DIR)
            print("[INFO] Adapter loaded. Continuing training with new data...")
        except Exception as e:
            print(f"[ERROR] Could not load existing adapter: {e}")
            print(f"[ACTION] The adapter might be corrupted or from a different version. Delete '{ADAPTER_OUTPUT_DIR}' to reset.")
            return
    else:
        print("[INFO] No existing adapter found. Starting fresh training...")
        model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=8,  # Reduced from 16 for faster training (less parameters)
            lora_alpha=16,  # Reduced proportionally
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_cfg)

    dataset = load_dataset(DATA_PATH)
    if len(dataset) == 0:
        print("[SKIP] No new edits to train.")
        return

    tokenized = dataset.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=dataset.column_names)

    # Optimization: If dataset is smaller than the accumulation target, lower it
    # to ensure the model actually takes optimizer steps.
    grad_acc = 4
    if len(dataset) < 8:
        grad_acc = 1
        print(f"[INFO] Small dataset ({len(dataset)} examples). Reducing gradient accumulation to 1.")

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=grad_acc,
        warmup_steps=2, # Reduced for small data
        max_steps=MAX_STEPS,
        num_train_epochs=5, # Higher epochs for very small data to ensure convergence
        learning_rate=LR,
        logging_steps=1, # More frequent logging for small data
        output_dir=ADAPTER_OUTPUT_DIR,
        save_strategy="no",  # Save manually at the end
        report_to="none",
        dataloader_num_workers=0  # Deterministic data loading on Windows
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)

    # 4. Update Ledger & Metadata
    from self_editor.save import update_ledger_applied_count
    num_applied = len(dataset)
    update_ledger_applied_count(num_applied)
    
    # Mark edits as applied in PROPOSALS_PATH (rewriting for simplicity in this minimal scale)
    if os.path.exists(DATA_PATH):
        temp_data = []
        with open(DATA_PATH, "r") as f:
            for line in f:
                e = json.loads(line)
                # This is a bit simplistic; in a larger system we'd use IDs
                # But here we'll just mark the ones that passed the filter
                if (e.get("simulation_passed") and 
                    not e.get("applied") and
                    e.get("risk_score", 1.0) <= 0.4 and # Hardcoded to match ledger default for now
                    e.get("confidence", 0.0) >= 0.7):
                    e["applied"] = True
                temp_data.append(e)
        
        with open(DATA_PATH, "w") as f:
            for e in temp_data:
                f.write(json.dumps(e) + "\n")

    metadata = {
        "base_model": args.model,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "num_edits_used": num_applied,
        "adapter_dir": ADAPTER_OUTPUT_DIR,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(ADAPTER_OUTPUT_DIR, "adapter_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Adapter trained and saved to {ADAPTER_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
