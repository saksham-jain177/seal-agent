# SEAL Agent (Continual-Intelligence)

A structurally faithful implementation of the [SEAL paper](https://arxiv.org/abs/2506.10943) (Self-Adapting Language Models). This agent performs web research and "self-adapts" its knowledge weights using a rigorous Edit Proposal and Simulation pipeline.

## Core Philosophical Alignment

- **Edit vs Data**: We do not ingest raw data. We propose **Intent-Driven Edits**.
- **Shadow Inference**: Every edit is **Simulated** (base vs proposed) to ensure impact.
- **Budgeted Learning**: weight updates are throttled by a **Learning Ledger**.
- **Sparse Updates**: Only impactful, low-risk, high-confidence knowledge reaches the model weights.

## System Architecture

- **Research LLM**: Llama 3.1 8B (via Ollama)
- **Base Model (Weights)**: Llama 3.2 3B Instruct
- **Adaptation**: PEFT LoRA (4-bit quantization)
- **Edit Simulation**: Shadow inference verifies knowledge impact before persistence.

## Repository Structure

```
seal-agent/
├── research_agent.py      # Research flow -> Edit Proposal -> Simulation
├── tuner.py               # Managed Training: respect budget/simulation gates
├── inference_adapter.py   # State-aware loading of fine-tuned knowledge
├── self_editor/
│   ├── propose_edit.py    # SEAL EditProposal abstraction
│   ├── simulate_edit.py   # Shadow Inference verifying impact
│   └── save.py            # Persistence & Ledger management
└── data/
    ├── edit_ledger.json    # Learning budget, risk thresholds, and stats
    └── edit_proposals.jsonl # Structured edits (pending/applied)
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

1. **Research & Propose**:

   ```bash
   python research_agent.py
   ```

   The agent will identify knowledge gaps, propose an edit, and run a **Shadow Inference Simulation**. Only impactful edits are saved to the ledger.

2. **Managed Training**:

   ```bash
   python tuner.py
   ```

   Filters and trains only on approved proposals while respecting the `risk_threshold` and `current_budget` in `data/edit_ledger.json`.

3. **Inference**:
   The agent automatically detects and loads the latest adapter weights.

## Learning Ledger (`data/edit_ledger.json`)

You can manually adjust the following constraints:

- `current_budget`: Max number of edits allowed in the current cycle.
- `risk_threshold`: Max risk score for an edit to be trained.
- `min_confidence`: Min confidence score for an edit to be accepted.
- `total_edits_applied`: Tracks cumulative learning progress.
