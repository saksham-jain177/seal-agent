# SEAL Agent

A Python implementation of [SEAL](https://arxiv.org/abs/2506.10943) (Self-Adapting Language Models). It researches the web and updates its own knowledge using a simulation pipeline.

## How it works

1.  **Research**: Queries Tavily, DuckDuckGo, or arXiv.
2.  **Edit Proposals**: Identifies a knowledge gap and proposes a Q&A pair.
3.  **Audit (Simulation)**: Compares the proposal against its current knowledge. Rejects if it's already known or low quality.
4.  **Compilation**: Validated edits are trained into a LoRA adapter.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Add TAVILY_API_KEY
ollama pull llama3.1:8b-instruct # Research model
```

## Usage

### 1. Research a topic

```bash
python research_agent.py
```

This saves verified edits to `data/edit_proposals.jsonl`.

### 2. Update model weight

```bash
python tuner.py
```

Trains on pending edits while respecting the budget in `data/edit_ledger.json`.

### 3. Talk to the model

```bash
python talk_to_model.py
```

## Project Layout

- `research_agent.py`: Web search -> Edits -> Simulation.
- `tuner.py`: The "compiler" that trains the model.
- `eval_harness.py`: Checks if the model actually learned the edits.
- `data/edit_ledger.json`: Budget, risk checks, and training stats.
