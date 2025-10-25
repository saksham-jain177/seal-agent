# Seal Agent

A self-improving research agent that generates, validates, and reviews Q&A pairs for continuous learning and model enhancement.

## Overview

Seal Agent is an intelligent research assistant that combines web search capabilities with a self-editing mechanism to create high-quality training data. The system automatically generates Q&A pairs from research queries, validates their quality, and maintains a curated knowledge base for continuous improvement.

## Features

- **Web Research**: Uses Tavily Search API for real-time web information retrieval
- **Local LLM Integration**: Powered by Ollama with Llama 3.1 8B model
- **Self-Edit Generation**: Automatically creates structured Q&A pairs from research results
- **Quality Validation**: Multi-criteria validation system for generated content
- **Duplicate Prevention**: Hash-based deduplication to avoid redundant entries
- **Automated Review**: LLM-powered quality scoring and approval system
- **Persistent Storage**: JSONL-based data storage with indexing

## Project Structure

```
seal-agent/
├── research_agent.py          # Main application entry point
├── new.py                     # Simple LLM test script
├── self_editor/               # Self-editing module
│   ├── generate_selfedit.py   # Q&A pair generation
│   ├── validate.py            # Content validation and sanitization
│   ├── save.py                # Data persistence and deduplication
│   └── review_selfedits.py    # Quality review and scoring
├── data/                      # Data storage directory
│   ├── self_edits.jsonl       # Generated Q&A pairs
│   └── self_edits_index.json  # Deduplication index
└── venv/                      # Virtual environment
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/saksham-jain177/seal-agent
   cd seal-agent
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install langchain-tavily langchain-ollama python-dotenv
   ```

4. **Set up Ollama**
   - Install [Ollama](https://ollama.ai/)
   - Pull the required model: `ollama pull llama3.1:8b-instruct-q4_K_M`

5. **Configure environment**
   - Create a `.env` file in the project root
   - Add your Tavily API key: `TAVILY_API_KEY=your_api_key_here`

## Usage

### Basic Research Query
```bash
python research_agent.py
```
Enter your research question when prompted. The system will:
1. Search the web for relevant information
2. Generate a comprehensive answer
3. Create a structured Q&A pair
4. Validate and save the content
5. Review the quality automatically

### Standalone Components

**Generate Self-Edit Only**
```bash
python self_editor/generate_selfedit.py
```

**Review Existing Entries**
```bash
python self_editor/review_selfedits.py
```

**Test LLM Connection**
```bash
python new.py
```

## Data Format

Generated Q&A pairs are stored in JSONL format with the following structure:

```json
{
  "question": "What are vision-language models (VLMs) and how do they work?",
  "answer": "Vision-language models (VLMs) are multimodal architectures that simultaneously comprehend image and text data modalities...",
  "source": "https://example.com/source",
  "created_at": "2025-10-24T11:19:08.371877Z"
}
```

## Quality Control

The system implements a multi-layered quality control system:

1. **Validation**: Content length, format, and basic quality checks
2. **Deduplication**: SHA-256 hash-based duplicate detection
3. **Review Scoring**: LLM-powered evaluation on:
   - Accuracy (50% weight)
   - Clarity (30% weight) 
   - Novelty (20% weight)
4. **Approval Threshold**: Automatic approval for scores ≥ 0.70

## Configuration

Key parameters can be adjusted in the respective modules:

- **Content Limits**: `MAX_QUESTION_LEN`, `MAX_ANSWER_LEN` in `validate.py`
- **Approval Threshold**: `APPROVAL_THRESHOLD` in `review_selfedits.py`
- **Search Results**: `max_results` in `research_agent.py`
- **LLM Model**: Model name in all LLM initialization calls

## Requirements

- Python 3.11+
- Ollama with Llama 3.1 8B model
- Tavily Search API key
- Internet connection for web search

## Dependencies

- `langchain-tavily`: Web search integration
- `langchain-ollama`: Local LLM integration
- `python-dotenv`: Environment variable management

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes and test thoroughly
4. Commit with clear messages
5. Push to your fork and create a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.