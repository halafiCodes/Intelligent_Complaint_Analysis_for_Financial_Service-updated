import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import openai
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
FAISS_PATH = BASE_DIR / "vector_store" / "faiss_index.bin"
METADATA_PATH = BASE_DIR / "vector_store" / "metadata.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5


def get_openai_client() -> OpenAI:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "Missing OPENAI_API_KEY.\n\n"
            "Fix option A (recommended): set an environment variable named OPENAI_API_KEY.\n"
            "Fix option B: create .streamlit/secrets.toml with:\n\n"
            "OPENAI_API_KEY = \"your_key_here\"\n"
        )
        st.stop()

    return OpenAI(api_key=api_key)


def embed_texts_openai(texts: list[str]) -> np.ndarray:
    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=np.float32)


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens: list[str] = []
    current = []
    for ch in text:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


@st.cache_resource
def _metadata_token_cache(metadata: list[dict]) -> list[set[str]]:
    cached: list[set[str]] = []
    for item in metadata:
        chunk = ""
        if isinstance(item, dict):
            chunk = str(item.get("chunk_text", ""))
        cached.append(set(_tokenize(chunk)))
    return cached


def keyword_retrieve(query: str, *, top_k: int = TOP_K) -> list[dict]:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    token_sets = _metadata_token_cache(metadata)
    scored: list[tuple[int, int]] = []  # (score, idx)
    for idx, token_set in enumerate(token_sets):
        score = len(query_tokens & token_set)
        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    results: list[dict] = []
    for _score, idx in scored[:top_k]:
        item = metadata[idx]
        if isinstance(item, dict):
            results.append(item)
    return results


def _is_quota_error(exc: Exception) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    # new SDK exceptions often carry an error payload in the message
    msg = str(exc).lower()
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg) or ("error code: 429" in msg)


@st.cache_resource
def load_resources():
    if not FAISS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store files not found. Build them first by running: python src/build_vector_store.py (from the rag-complaint-chatbot folder)."
        )

    index = faiss.read_index(str(FAISS_PATH))

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

try:
    index, metadata = load_resources()
except Exception as exc:
    st.error(str(exc))
    st.stop()


def retrieve(query):
    try:
        query_embedding = embed_texts_openai([query])
        index_dim = int(getattr(index, "d", 0))
        if index_dim and int(query_embedding.shape[1]) != index_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: FAISS index dim={index_dim} but '{EMBEDDING_MODEL}' returned dim={int(query_embedding.shape[1])}. "
                "Rebuild the vector store using the same embedding model as the app (run: python src/build_vector_store.py)."
            )

        _distances, indices = index.search(query_embedding, TOP_K)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            if idx < len(metadata):
                results.append(metadata[idx])
        return results
    except Exception as exc:
        if _is_quota_error(exc):
            st.warning(
                "OpenAI quota is unavailable (HTTP 429 / insufficient_quota). "
                "Falling back to offline keyword search for retrieval."
            )
            return keyword_retrieve(query, top_k=TOP_K)
        raise


def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "I don't have enough information in the indexed complaints to answer that question."

    context = "\n\n".join([c["chunk_text"] for c in retrieved_chunks])

    prompt = f"""
You are a financial complaint analyst.

Use ONLY the complaint excerpts below to answer.

Complaint Excerpts:
{context}

Question:
{query}
"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as exc:
        if _is_quota_error(exc):
            st.warning(
                "OpenAI quota is unavailable (HTTP 429 / insufficient_quota). "
                "Showing an extractive answer from the most relevant complaint excerpts instead."
            )
            return "\n\n".join([c.get("chunk_text", "") for c in retrieved_chunks])
        raise


st.title("📊 Financial Complaint RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about complaints:")

col1, col2 = st.columns(2)
ask = col1.button("Ask")
clear = col2.button("Clear")

if clear:
    st.session_state.history = []
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

if ask and query:
    with st.spinner("Analyzing complaints..."):
        retrieved = retrieve(query)
        answer = generate_answer(query, retrieved)
        st.session_state.history.append((query, answer, retrieved))



for q, a, sources in reversed(st.session_state.history):
    st.markdown("### ❓ Question")
    st.write(q)

    st.markdown("### 🤖 Answer")
    st.write(a)

    st.markdown("### 📚 Sources Used")
    for i, src in enumerate(sources):
        with st.expander(f"Source {i+1} - {src['product']}"):
            st.write(src["chunk_text"])
