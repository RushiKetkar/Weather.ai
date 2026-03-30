import math
import ollama
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

EMBEDDING_MODEL = "mxbai-embed-large:335m"
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None
_vector_db: list[tuple[str, np.ndarray]] = []
_documents: pd.DataFrame | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker: {RERANKING_MODEL} …")
        _reranker = CrossEncoder(RERANKING_MODEL)
    return _reranker


def _get_embedding(text: str) -> np.ndarray:
    return np.array(ollama.embed(model=EMBEDDING_MODEL, input=text)["embeddings"][0])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot   = float(np.dot(a, b))
    denom = (math.sqrt(float(np.dot(a, a))) *
             math.sqrt(float(np.dot(b, b))))
    return dot / denom if denom else 0.0

def load_and_embed(filepath: str = "weather_facts.txt") -> None:
    """Read facts file, embed every non-empty line, and populate the vector DB."""
    global _vector_db, _documents

    with open(filepath, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    print(f"Embedding {len(lines)} facts from '{filepath}' …")
    _vector_db = []
    for i, text in enumerate(lines):
        emb = _get_embedding(text)
        _vector_db.append((text, emb))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(lines)} embedded")

    _documents = pd.DataFrame({
        "doc_id":   range(len(lines)),
        "text":     lines,
        "metadata": [{}] * len(lines),
    })
    print("RAG vector DB ready.")


def _init_retrieve(query: str, top_k: int) -> pd.DataFrame:
    """Return top_k candidates ranked by embedding cosine similarity."""
    assert _documents is not None, "Call load_and_embed() first."

    query_emb   = _get_embedding(query)
    similarities = []
    text_arr     = []

    for text, emb in _vector_db:
        similarities.append(_cosine_similarity(query_emb, emb))
        text_arr.append(text)

    res = _documents.copy()
    res["_sim"]  = similarities
    res["_text"] = text_arr
    top_k = min(top_k, len(res))
    res  = res.nlargest(top_k, "_sim")
    sims = res["_sim"].tolist()
    txts = res["_text"].tolist()
    res  = res.drop(columns=["_sim", "_text"])
    res["embedding_score"] = list(zip(txts, sims))
    return res.reset_index(drop=True)


def _rerank(query: str, candidates: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Rerank candidates with the cross-encoder; return top_k rows."""
    pairs         = [[query, doc] for doc in candidates["text"]]
    rerank_scores = _get_reranker().predict(pairs)

    res = candidates.copy()
    res["rerank_score"] = rerank_scores
    top_k = min(top_k, len(res))
    res  = res.nlargest(top_k, "rerank_score")
    return res.reset_index(drop=True)


def search(query: str, initial_k: int = 20, final_k: int = 5) -> pd.DataFrame:
    """
    Two-stage retrieval:
      1. Cosine-similarity recall (initial_k)
      2. Cross-encoder rerank    (final_k)
    Returns a DataFrame with columns: doc_id, text, metadata,
    embedding_score, rerank_score.
    """
    candidates = _init_retrieve(query, top_k=initial_k)
    return _rerank(query, candidates, top_k=final_k)
