# ASIR-RAG: Adaptive Strategy Intelligent Retrieval for Legal Documents

ASIR-RAG is a multi-strategy Retrieval-Augmented Generation (RAG) project for legal and policy documents (for example, DPDP Act and IT Act resources).

It combines:
- Query classification (factual / relational / comparative / exploratory / no_retrieval)
- Multiple retrieval modes (vector, keyword BM25, hierarchical, graph)
- Cross-encoder re-ranking
- Grounded answer generation with LLM fallback
- Feedback logging for future model improvement

---

## Project Structure

Main folders:
- `classifier/` - Query preprocessing and classification pipeline
- `retriever/` - Vector, BM25 keyword, and hierarchical retrieval
- `graph_builder/`, `graph_retriever/`, `extraction/` - Knowledge graph build + graph retrieval
- `generation/` - Prompt building and final answer generation
- `ingestion/` - Chunking and index construction
- `feedback/` - Logging, feedback capture, and training data utilities
- `config/` - Project-wide configs
- `shared/` - Shared models and dataclasses

Main scripts:
- `ingestion.py` - Build vector/keyword/hierarchical indexes
- `ingestion_graph.py` - Build graph index from PDFs
- `main.py` - Interactive CLI RAG pipeline
- `classifier_main.py` - Classifier demo
- `reranker_main.py` - Re-ranker demo
- `app.py` - App entry (if using web/UI flow)

---

## Setup

### 1) Clone and enter project
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2) Create virtual environment (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Install spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5) Prepare environment variables
Create a local `.env` file in project root and set API keys:
```env
# Gemini key used by generator
BACKUP_LLM_API_KEY=your_gemini_api_key

# Optional compatibility key (some error messages refer to this name)
GEMINI_API_KEY=your_gemini_api_key
```

### 6) Start Ollama (for local classifier/fallback flows)
```bash
ollama pull llama3
ollama serve
```

---

## Data Preparation

Place your source PDFs in:
```text
data_file/
```

Current local dataset includes legal documents (DPDP, IT Act, etc.).

---

## Build Indexes

### A) Build vector + keyword + hierarchical indexes
```bash
python ingestion.py
```

### B) Build knowledge graph index
```bash
python ingestion_graph.py
```

---

## Run

### Interactive RAG CLI
```bash
python main.py
```

### Classifier demo
```bash
python classifier_main.py
```

### Re-ranker demo
```bash
python reranker_main.py
```

---

## Notes for GitHub Upload

- This repo includes a `.gitignore` configured for:
  - Python cache/venv files
  - `.env` secrets
  - local databases and generated model files
  - large local data/index folders
  - local WhatsApp image artifact
- If you want to version-control sample data, remove specific entries from `.gitignore` before commit.

---

## Suggested First Commit

```bash
git init
git add .
git commit -m "Initial commit: ASIR-RAG project setup"
```

