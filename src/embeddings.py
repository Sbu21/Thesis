import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import logging
import os
from src.db import load_concepts_dict, load_metadata
from src.nlp import preprocess_query

logger = logging.getLogger(__name__)
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
        db_path: str = "data/traffic_code.db",  # Default, but will be passed from Flask
        index_path: str = "data/faiss_index.index",  # Default, but will be passed from Flask
        k_faiss_retrieval: int = 100,  # How many results to fetch from FAISS initially
        top_k_final: int = 5,  # How many results to return after reranking
        alpha: float = 0.3  # Weight for semantic_score vs overlap_score
) -> list:
    """
    Performs semantic + concept-based search using FAISS and your SQLite DB.

    Params:
        - query: natural language user query (in Romanian)
        - db_path: path to the SQLite DB file
        - index_path: path to FAISS .index file
        - k_faiss_retrieval: number of FAISS top results to retrieve
        - top_k_final: how many results to return after reranking
        - alpha: weight between semantic (FAISS) and concept overlap (0–1)

    Returns:
        - List of result dicts with scores and matched metadata
    """
    if model is None:
        logger.error("SentenceTransformer model not loaded. Cannot perform cosine search.")
        return []
    if not os.path.exists(index_path):
        logger.error(f"FAISS index not found at {index_path}")
        return []
    if not os.path.exists(db_path):
        logger.error(f"SQLite DB not found at {db_path}")
        return []

    logger.info(
        f"Cosine search with concepts: query='{query}', k_faiss={k_faiss_retrieval}, top_k_final={top_k_final}, alpha={alpha}")

    try:
        index = faiss.read_index(index_path)
    except RuntimeError as e:
        logger.error(f"Failed to read FAISS index from {index_path}: {e}", exc_info=True)
        return []

    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    # FAISS returns L2 distances. For normalized embeddings, D^2 = 2 - 2*cos_sim.
    # So, cos_sim = 1 - (D^2 / 2). Higher cos_sim is better.
    # index.search returns distances (D) and indices (I).
    distances, indices = index.search(query_embedding, k_faiss_retrieval)

    metadata_by_id = load_metadata(db_path)
    concepts_by_id = load_concepts_dict(db_path)
    query_lemmas = preprocess_query(query)

    all_doc_ids_in_db = list(metadata_by_id.keys())  # Assuming these correspond to FAISS index order
    # This needs to be robust, using the faiss_row_ids.json mapping

    faiss_ids_mapping_path = os.path.join(os.path.dirname(index_path), "faiss_row_ids.json")
    db_ids_for_faiss_indices = []
    if os.path.exists(faiss_ids_mapping_path):
        with open(faiss_ids_mapping_path, "r", encoding="utf-8") as f_map:
            db_ids_for_faiss_indices = json.load(f_map)
    else:
        logger.warning(
            f"FAISS ID mapping file not found at {faiss_ids_mapping_path}. Assuming direct mapping or using all_doc_ids_in_db if lengths match.")
        if index.ntotal == len(all_doc_ids_in_db):
            db_ids_for_faiss_indices = all_doc_ids_in_db
        else:
            logger.error("FAISS ID mapping missing and DB ID list size mismatch. Cannot reliably map FAISS results.")
            return []

    results = []
    min_distance_for_norm = float('inf')
    max_distance_for_norm = 0.0

    raw_candidates = []
    for i in range(len(indices[0])):
        faiss_idx = indices[0][i]
        if faiss_idx == -1:
            continue

        distance = distances[0][i]
        try:
            result_id = db_ids_for_faiss_indices[faiss_idx]
            meta = metadata_by_id.get(result_id)
            if not meta:
                logger.warning(f"Metadata not found for db_id {result_id} (FAISS index {faiss_idx}). Skipping.")
                continue

            paragraph_concepts = concepts_by_id.get(result_id, [])

            matched_c = [c for c in paragraph_concepts if any(q_lemma in c.lower() for q_lemma in query_lemmas)]
            overlap_score = len(matched_c) / max(len(paragraph_concepts), 1) if paragraph_concepts else 0.0

            raw_candidates.append({
                "id": result_id,
                "meta": meta,
                "distance": float(distance),
                "overlap_score": float(overlap_score),
                "matched_concepts": matched_c,
                "paragraph_concepts": paragraph_concepts
            })
            if distance != -1:  # Valid distance
                min_distance_for_norm = min(min_distance_for_norm, distance)
                max_distance_for_norm = max(max_distance_for_norm, distance)

        except IndexError:
            logger.warning(f"FAISS index {faiss_idx} out of bounds for db_ids_for_faiss_indices. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Error processing candidate for db_id (from FAISS index {faiss_idx}): {e}", exc_info=True)

    for cand in raw_candidates:
        semantic_score_from_distance = 0.0
        if max_distance_for_norm > min_distance_for_norm:
            semantic_score_from_distance = 1.0 - (
                        (cand["distance"] - min_distance_for_norm) / (max_distance_for_norm - min_distance_for_norm))
        elif len(raw_candidates) == 1 and cand["distance"] != -1:
            semantic_score_from_distance = 0.5
            if cand["distance"] < 1.0:
                semantic_score_from_distance = 1.0
            elif cand["distance"] > 1.5:
                semantic_score_from_distance = 0.1


        final_score = alpha * semantic_score_from_distance + (1 - alpha) * cand["overlap_score"]

        results.append({
            "id": cand["id"],
            "article": cand["meta"]["article"],
            "paragraph": cand["meta"]["paragraph"],
            "text": cand["meta"]["text"],
            "semantic_score": semantic_score_from_distance,
            "raw_distance": cand["distance"],
            "overlap_score": cand["overlap_score"],
            "final_score": float(final_score),
            "matched_concepts": cand["matched_concepts"]
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    logger.info(f"Cosine search returning {len(results[:top_k_final])} results. Alpha={alpha}")
    return results[:top_k_final]

