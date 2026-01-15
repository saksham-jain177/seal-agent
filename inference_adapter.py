"""
Adapter Inference Module

Loads fine-tuned LoRA adapter and uses it for text generation.
This allows the research agent to benefit from continual learning.
"""
import os
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration (matches tuner.py)
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_PATH = "adapters/lora_adapter"

# Global variables for lazy loading
_model = None
_tokenizer = None


def load_adapter_model(adapter_path: str = ADAPTER_PATH) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
    """
    Load the fine-tuned LoRA adapter for inference.
    Returns (model, tokenizer) or (None, None) if adapter not found.
    
    Uses lazy loading - only loads once, reuses on subsequent calls.
    """
    global _model, _tokenizer
    
    # Check if adapter exists
    if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        return None, None
    
    # Return cached model if already loaded
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        print(f"[Adapter] Loading fine-tuned adapter from {adapter_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load base model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        
        # Cache for reuse
        _model = model
        _tokenizer = tokenizer
        
        print("[Adapter] Fine-tuned adapter loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"[Adapter] Failed to load adapter: {e}")
        return None, None


def generate_with_adapter(
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    adapter_path: str = ADAPTER_PATH
) -> Optional[str]:
    """
    Generate text using the fine-tuned adapter.
    
    Args:
        prompt: Input prompt text
        max_length: Maximum generation length
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        top_p: Nucleus sampling parameter
        adapter_path: Path to adapter directory
    
    Returns:
        Generated text or None if adapter not available
    """
    model, tokenizer = load_adapter_model(adapter_path)
    
    if model is None or tokenizer is None:
        return None
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        print(f"[Adapter] Generation failed: {e}")
        return None


def is_adapter_available(adapter_path: str = ADAPTER_PATH) -> bool:
    """Check if adapter is available for use."""
    return os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors"))


if __name__ == "__main__":
    # Test adapter loading
    print("Testing adapter inference...")
    if is_adapter_available():
        test_prompt = "Question: What are vision-language models?\nAnswer:"
        result = generate_with_adapter(test_prompt, max_length=200, temperature=0.7)
        if result:
            print(f"\nGenerated:\n{result}")
        else:
            print("Generation failed")
    else:
        print("Adapter not found. Train an adapter first using tuner.py")

