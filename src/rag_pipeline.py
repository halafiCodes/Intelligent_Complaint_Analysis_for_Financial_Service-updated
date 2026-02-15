from __future__ import annotations

from pathlib import Path
import pickle
import os
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.pkl"
EVAL_OUTPUT = ROOT_DIR / "data" / "rag_evaluation.csv"


PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer,
state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""


def _require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    question = question.strip()
    if not question:
        raise ValueError("question must be non-empty")
    return question


def _validate_top_k(top_k: int) -> int:
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an int")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    return top_k


def load_vector_store(
    index_path: Path = INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
):
    _require_file(index_path, "FAISS index")
    _require_file(metadata_path, "metadata")

    try:
        import faiss  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: faiss. Install with: pip install faiss-cpu "
            "(Windows may require conda-forge: conda install -c conda-forge faiss-cpu)."
        ) from exc

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    if not hasattr(index, "ntotal") or int(index.ntotal) <= 0:
        raise ValueError("FAISS index contains no vectors")
    if not isinstance(metadata, list):
        raise TypeError("metadata.pkl must contain a list")
    if len(metadata) != int(index.ntotal):
        raise ValueError(
            f"Vector store mismatch: index.ntotal={int(index.ntotal)} but metadata has {len(metadata)} rows"
        )

    return index, metadata


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: sentence-transformers. Install with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def retrieve_chunks(
    question: str,
    *,
    index,
    metadata: list[Any],
    embedding_model,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    question = _validate_question(question)
    top_k = _validate_top_k(top_k)

    ntotal = int(getattr(index, "ntotal", 0))
    if ntotal <= 0:
        raise ValueError("FAISS index contains no vectors")
    top_k = min(top_k, ntotal)

    query_vec = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    _distances, indices = index.search(query_vec, top_k)
    results: list[dict[str, Any]] = []
    for idx in indices[0]:
        if int(idx) < 0:
            continue
        item = metadata[int(idx)]
        if not isinstance(item, dict):
            raise TypeError("metadata entries must be dicts")
        results.append(item)
    return results


def build_generator(
    model_name: str | None = None,
    device: str = "cpu",
    max_new_tokens: int = 256,
):
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = os.environ.get("RAG_GENERATOR_MODEL", "distilgpt2").strip()
    if not model_name:
        raise ValueError("model_name must be a non-empty string")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps")
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive int")

    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Missing dependency: transformers. Install with: pip install transformers"
        ) from exc

    try:
        return pipeline(
            "text-generation",
            model=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to load generator model '{model_name}'. "
            "If this is a gated model, authenticate with Hugging Face (e.g., set HF_TOKEN or run 'huggingface-cli login'), "
            "or set RAG_GENERATOR_MODEL to an open model such as 'distilgpt2'."
        ) from exc


def generate_answer(
    question: str,
    *,
    index,
    metadata: list[Any],
    embedding_model,
    generator,
    top_k: int = 5,
    temperature: float = 0.3,
) -> str:
    question = _validate_question(question)
    top_k = _validate_top_k(top_k)
    if not isinstance(temperature, (int, float)) or float(temperature) < 0:
        raise ValueError("temperature must be >= 0")

    chunks = retrieve_chunks(
        question,
        index=index,
        metadata=metadata,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    if chunks and "text" not in chunks[0]:
        raise ValueError(
            "Retrieved items do not include 'text'. Your metadata.pkl appears to contain only metadata fields. "
            "Rebuild the vector store and persist chunk text alongside metadata (e.g., save full documents)."
        )

    context_text = "\n\n".join([c.get("text", "") for c in chunks])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    output = generator(prompt, do_sample=True, temperature=float(temperature))[0]["generated_text"]
    return str(output)


def run_evaluation() -> None:
    index, metadata = load_vector_store()
    embedding_model = load_embedding_model()

    probe = retrieve_chunks(
        "probe",
        index=index,
        metadata=metadata,
        embedding_model=embedding_model,
        top_k=1,
    )
    if probe and "text" not in probe[0]:
        raise ValueError(
            "Vector store metadata does not include chunk 'text'. "
            "Update the vector store builder to persist chunk text alongside metadata (or store a separate documents list)."
        )

    generator = build_generator()

    questions = [
        "What is the most common issue with credit cards?",
        "Which company received the highest complaints for personal loans?",
        "What are typical complaints about money transfers?",
        "Are savings account complaints mostly about fees?",
        "Which state has the most complaints?",
    ]

    evaluation_results: list[dict[str, Any]] = []

    for q in questions:
        retrieved = retrieve_chunks(q, index=index, metadata=metadata, embedding_model=embedding_model)
        answer = generate_answer(
            q,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            generator=generator,
        )
        sources = "\n".join([f"{r.get('company')} | {r.get('product')}" for r in retrieved[:2]])
        evaluation_results.append(
            {
                "Question": q,
                "Generated Answer": answer,
                "Retrieved Sources": sources,
                "Quality Score": None,
                "Comments/Analysis": "",
            }
        )

    eval_df = pd.DataFrame(evaluation_results)
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(EVAL_OUTPUT, index=False)
    print(f"Saved evaluation table to {EVAL_OUTPUT}")


if __name__ == "__main__":
    run_evaluation()
