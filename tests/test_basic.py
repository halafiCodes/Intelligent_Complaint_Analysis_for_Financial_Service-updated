import numpy as np
import pandas as pd
import pytest

from src.build_vector_store import _word_chunks, create_documents, stratified_sample
from src import rag_pipeline


def test_stratified_sample_empty_df_returns_empty_df():
    df = pd.DataFrame(columns=["Product", "Company", "State", "cleaned_narrative"])
    sampled = stratified_sample(df, sample_size=100)
    assert isinstance(sampled, pd.DataFrame)
    assert sampled.shape[0] == 0
    assert list(sampled.columns) == list(df.columns)


def test_stratified_sample_missing_group_col_raises():
    df = pd.DataFrame({"X": ["a", "b"]})
    with pytest.raises(KeyError):
        stratified_sample(df, group_col="Product")


def test_word_chunks_overlap_behavior():
    text = "one two three four five six seven eight nine ten"
    chunks = _word_chunks(text, chunk_size=4, chunk_overlap=1)
    assert chunks[0] == "one two three four"
    assert chunks[1].startswith("four")
    assert len(chunks) >= 3


def test_create_documents_requires_text_col():
    df = pd.DataFrame({"Product": ["A"], "cleaned_narrative": ["hello world"]})
    with pytest.raises(KeyError):
        create_documents(df, text_col="missing")


class DummyEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        assert isinstance(texts, list)
        return np.array([[0.1, 0.2]], dtype=np.float32)


class DummyIndex:
    def __init__(self, ntotal=3):
        self.ntotal = ntotal

    def search(self, query_vec, top_k):
        distances = np.zeros((1, top_k), dtype=np.float32)
        indices = np.arange(top_k, dtype=np.int64).reshape(1, -1)
        return distances, indices


def test_rag_retrieve_chunks_validates_and_returns_dicts():
    index = DummyIndex(ntotal=3)
    metadata = [
        {"text": "a", "company": "c1", "product": "p1"},
        {"text": "b", "company": "c2", "product": "p2"},
        {"text": "c", "company": "c3", "product": "p3"},
    ]
    results = rag_pipeline.retrieve_chunks(
        "What happened?",
        index=index,
        metadata=metadata,
        embedding_model=DummyEmbeddingModel(),
        top_k=2,
    )
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)


def test_rag_retrieve_chunks_rejects_empty_question():
    index = DummyIndex(ntotal=1)
    metadata = [{"text": "a"}]
    with pytest.raises(ValueError):
        rag_pipeline.retrieve_chunks(
            "   ",
            index=index,
            metadata=metadata,
            embedding_model=DummyEmbeddingModel(),
            top_k=1,
        )
