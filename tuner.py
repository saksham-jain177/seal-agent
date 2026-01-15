import os
import json
import torch
from typing import Tuple
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

import argparse

# ---- Configuration ----
# Default to Llama-3.2-3B for 6GB VRAM compatibility
DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct" 
DATA_PATH = "data/self_edits.jsonl"
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
    with open(path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    pairs = []
    for e in entries:
        q = e.get("question") or e.get("topic") or ""
        a = e.get("answer") or e.get("content") or ""
        if q and a:
            pairs.append({"prompt": f"Question: {q}\nAnswer:", "response": a})
    return Dataset.from_list(pairs)

# ---- Tokenization ----
def tokenize(examples, tokenizer):
    combined = [p + " " + r for p, r in zip(examples["prompt"], examples["response"])]
    tokens = tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    return tokens

# ---- Incremental Training Check ----
def should_train(min_new_edits: int = MIN_NEW_EDITS_FOR_TRAINING) -> Tuple[bool, int]:
    """
    Check if we have enough new edits to justify training.
    Returns (should_train: bool, num_edits: int)
    """
    if not os.path.exists(DATA_PATH):
        return False, 0
    
    # Count total edits
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        total_edits = sum(1 for line in f if line.strip())
    
    # Check last training metadata
    metadata_path = os.path.join(ADAPTER_OUTPUT_DIR, "adapter_meta.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            last_trained_count = metadata.get("num_edits_used", 0)
            new_edits = total_edits - last_trained_count
            should = new_edits >= min_new_edits
            return should, new_edits
        except Exception:
            pass
    
    # No previous training or metadata corrupted
    should = total_edits >= min_new_edits
    return should, total_edits

# ---- Main ----
def main():
    parser = argparse.ArgumentParser(description="Fine-tune the SEAL agent model.")
    parser.add_argument("--force", "-f", action="store_true", help="Force training regardless of data count.")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Base model to use for training.")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
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
        print(f"[INFO] Loading existing adapter from {ADAPTER_OUTPUT_DIR} for continued training...")
        model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, ADAPTER_OUTPUT_DIR)
        print("[INFO] Adapter loaded. Continuing training with new data...")
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
    tokenized = dataset.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        logging_steps=5,
        output_dir=ADAPTER_OUTPUT_DIR,
        save_strategy="no",  # Save manually at the end
        report_to="none",
        dataloader_num_workers=0  # Deterministic data loading on Windows
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)

    metadata = {
        "base_model": args.model,
        "epochs": EPOCHS,
        "max_steps": MAX_STEPS,
        "num_edits_used": len(dataset),
        "adapter_dir": ADAPTER_OUTPUT_DIR,
        "timestamp": str(torch.utils.benchmark.utils.common.datetime.datetime.now())
    }
    with open(os.path.join(ADAPTER_OUTPUT_DIR, "adapter_meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Adapter trained and saved to {ADAPTER_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
