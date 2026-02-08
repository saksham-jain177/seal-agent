# SEAL Agent

A minimal, edge-device–oriented self-adapting research agent inspired by the SEAL framework. The agent performs web research, generates high-quality Q&A self-edits, and periodically internalizes knowledge via local LoRA fine-tuning.

## Core Principles

- **Minimal Surface Area**: No overengineering, no noise.
- **Local-First**: Inference and data generation happen on-device (RTX 4050 6GB compatible).
- **Manual Control**: Training is an explicit, manual batch-process—no hidden background loops.
- **Reproducible State**: Deterministic training and predictable adapter loading.

## Architecture

- **Research LLM**: Llama 3.1 8B (via Ollama)
- **Training Base Model**: Llama 3.2 3B Instruct (via Unsloth/HuggingFace)
- **Fine-Tuning**: LoRA (4-bit quantization)
- **Web Search**: Tavily Search API

## Repository Structure

```
seal-agent/
├── research_agent.py      # Entry point: Web search + Answer synthesis
├── tuner.py               # LoRA fine-tuning script (Manual entry)
├── inference_adapter.py   # Logic for loading fine-tuned adapters
├── self_editor/           # Core logic: generate, validate, review edits
├── data/                  # Local training data (JSONL)
└── adapters/              # Saved LoRA checkpoints
```

## Setup

1. **Environment**:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env  # Add TAVILY_API_KEY
   ```
2. **Local Models**:
   - Install [Ollama](https://ollama.ai/)
   - `ollama pull llama3.1:8b-instruct-q4_K_M`

## Usage Workflow

1. **Research & Collect**:
   Run the agent to collect knowledge for your dataset.
   ```bash
   python research_agent.py
   ```
2. **Train (Manual Batch)**:
   Once you have enough data (threshold: 10 edits), update the model weights.
   ```bash
   python tuner.py
   ```
3. **Internalize**:
   The agent automatically detects and loads the latest adapter from `adapters/lora_adapter` on next run.
