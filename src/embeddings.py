import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from src.db import load_concepts_dict, load_metadata
from src.nlp import preprocess_query

model = SentenceTransformer("BlackKakapo/stsb-xlm-r-multilingual-ro")

def generate_embeddings(texts: list) -> np.ndarray:
    """
    Encode a list of texts using the Sentence-BERT model.
    Returns a NumPy array of shape (n_texts, embedding_dim).
    """
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index from the given embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    return index

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)

def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

def load_embedding_metadata(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_search_with_concepts(
        query: str,
        db_path: str = "data/traffic_code.db",
        index_path: str = "data/faiss_index.index",
        k: int = 10,
        top_k: int = 5,
        alpha: float = 0.7
) -> list:
    """
    Performs semantic + concept-based search using FAISS and your SQLite DB.

    Params:
        - query: natural language user query (in Romanian)
        - db_path: path to the SQLite DB file
        - index_path: path to FAISS .index file
        - k: number of FAISS top results to retrieve
        - top_k: how many results to return after reranking
        - alpha: weight between semantic (FAISS) and concept overlap (0–1)

    Returns:
        - List of result dicts with scores and matched metadata
    """
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Generate normalized embedding for the query
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_embedding, k)

    # Load metadata and concepts from DB
    metadata_by_id = load_metadata(db_path)
    concepts_by_id = load_concepts_dict(db_path)
    query_lemmas = preprocess_query(query)

    # Build result list
    all_ids = list(metadata_by_id.keys())
    results = []

    for faiss_idx, score in zip(indices[0], scores[0]):
        try:
            result_id = all_ids[faiss_idx]
            meta = metadata_by_id[result_id]
            paragraph_concepts = concepts_by_id.get(result_id, [])

            # Compute concept overlap score
            matched = [c for c in paragraph_concepts if any(q in c.lower() for q in query_lemmas)]
            overlap_score = len(matched) / max(len(paragraph_concepts), 1)

            # Combined score
            final_score = alpha * score + (1 - alpha) * overlap_score

            results.append({
                "id": result_id,
                "article": meta["article"],
                "paragraph": meta["paragraph"],
                "text": meta["text"],
                "semantic_score": float(score),
                "overlap_score": float(overlap_score),
                "final_score": float(final_score),
                "matched_concepts": matched
            })
        except IndexError:
            continue  # In case FAISS returns an index out of bounds

    # Sort by final combined score
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:top_k]

