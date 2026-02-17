# Intelligent Complaint Analysis (RAG Chatbot)

A small end-to-end Retrieval-Augmented Generation (RAG) project for exploring financial service complaints.

## What’s included

- **Streamlit app** for interactive Q&A over complaint excerpts
- **Vector store builder** that chunks complaint narratives, embeds them, and builds a FAISS index
- **RAG pipeline utilities** (`src/rag_pipeline.py`) for retrieval + evaluation

## Repository layout

- `app.py` — Streamlit UI (retrieval + response generation)
- `src/build_vector_store.py` — builds FAISS index + metadata from processed CSV
- `src/rag_pipeline.py` — reusable retrieval helpers + evaluation runner
- `data/` — raw + processed datasets (ignored in git by default)
- `vector_store/` — FAISS artifacts (ignored in git by default)
- `notebooks/` — EDA / preprocessing notebook(s)
- `tests/` — unit tests

## Prerequisites

- Python 3.10+ (3.11/3.12 OK)
- (Optional) An OpenAI API key in `OPENAI_API_KEY`

## Setup (Windows / PowerShell)

```powershell
cd rag-complaint-chatbot

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

## Configure your API key (optional)

The app reads the key from either:

1) Environment variable:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

2) Streamlit secrets file (recommended for Streamlit):

Create `.streamlit/secrets.toml` (note: this file is ignored by git):

```toml
OPENAI_API_KEY = "sk-..."
```

You can copy the example:

```powershell
Copy-Item .streamlit/secrets.toml.example .streamlit/secrets.toml
```

## Build the vector store

Input expected at:

- `data/processed/filtered_complaints.csv`

Then run:

```powershell
python src/build_vector_store.py
```

This creates:

- `vector_store/faiss_index.bin`
- `vector_store/metadata.pkl`

`src/build_vector_store.py` auto-detects the complaint text column from:

- `cleaned_narrative`
- `Consumer complaint narrative`
- `complaint_text`

## Run the Streamlit app

```powershell
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

## OpenAI quota / 429 errors

If OpenAI returns `429 insufficient_quota` (no credit/billing/quota), the app is designed to **keep working**:

- Retrieval falls back to **offline keyword search** (no embeddings call)
- Answer falls back to an **extractive response** (shows the most relevant excerpts)

Note: building a fresh vector store still requires embeddings. If you don’t have OpenAI quota, you’ll need either an existing `vector_store/` already built, or to swap embeddings to another provider.

## Tests

```powershell
pytest -q
```

## Troubleshooting

- **Vector store files not found**: run `python src/build_vector_store.py`
- **Embedding dimension mismatch**: rebuild the vector store with the same embedding model used by the app
- **API key not detected**: set `OPENAI_API_KEY` or create `.streamlit/secrets.toml`
