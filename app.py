from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import sys
import os
from src.db import (
        create_connection,
        get_distinct_article_headers_from_db,
        get_paragraph_identifiers_for_article_from_db,
        get_content_by_article_and_paragraph_from_db )
from src.embeddings import cosine_search_with_concepts
from src.graph_query import graph_semantic_search

app = Flask(__name__)
CORS(app)

logger = logging.getLogger("traffic_code_api_main_graph")
if not app.debug or not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

DB_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "traffic_code.db")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faiss_index.index")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

logger.info(f"Database file path configured to: {DB_FILE_PATH}")
logger.info(f"FAISS index path configured to: {FAISS_INDEX_PATH}")
logger.info(f"Neo4j URI configured to: {NEO4J_URI}")


# --- Helper to get DB connection (keep as is) ---
def get_db_connection():
    if not os.path.exists(DB_FILE_PATH):
        logger.error(f"DATABASE FILE NOT FOUND: {DB_FILE_PATH}. API endpoints requiring DB will fail.")
        return None
    conn = create_connection(DB_FILE_PATH)
    return conn


# --- API Endpoints for Article Number Search
@app.route('/api/articles', methods=['GET'])
def list_articles_endpoint():
    logger.info("API call: /api/articles")
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        headers = get_distinct_article_headers_from_db(conn)
        return jsonify({"articles": headers})
    except Exception as e:
        logger.error(f"Error in /api/articles: {e}", exc_info=True)
        return jsonify({"error": "Failed to retrieve article headers", "details": str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/articles/<path:article_header>/paragraphs', methods=['GET'])
def list_paragraphs_for_article_endpoint(article_header):
    logger.info(f"API call: /api/articles/{article_header}/paragraphs")
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        identifiers = get_paragraph_identifiers_for_article_from_db(conn, article_header)
        return jsonify({"article_header": article_header, "paragraphs": identifiers})
    except Exception as e:
        logger.error(f"Error in /api/articles/{article_header}/paragraphs: {e}", exc_info=True)
        return jsonify({"error": f"Failed to retrieve paragraphs for {article_header}", "details": str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/search/article-content', methods=['GET'])
def search_specific_article_content_endpoint():
    article_header = request.args.get('article_header')
    paragraph_identifier = request.args.get('paragraph_identifier')
    logger.info(
        f"API call: /api/search/article-content - article_header='{article_header}', paragraph_identifier='{paragraph_identifier}'")
    if not article_header:
        return jsonify({"error": "'article_header' query parameter is required"}), 400
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    try:
        results = get_content_by_article_and_paragraph_from_db(
            conn, article_header_str=article_header, paragraph_identifier_str=paragraph_identifier
        )
        return jsonify({"query": {"article_header": article_header, "paragraph_identifier": paragraph_identifier},
                        "results": results})
    except Exception as e:
        logger.error(f"Error in /api/search/article-content: {e}", exc_info=True)
        return jsonify({"error": "Search operation failed", "details": str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/search/semantic', methods=['GET'])
def semantic_search_endpoint():
    query = request.args.get('q')
    k_results = request.args.get('k', default=5, type=int)
    alpha_param = request.args.get('alpha', default=0.3, type=float)
    k_faiss_param = request.args.get('k_faiss', default=100, type=int)

    logger.info(
        f"API call: /api/search/semantic - query='{query}', k={k_results}, alpha={alpha_param}, k_faiss={k_faiss_param}")

    if not query:
        return jsonify({"error": "'q' query parameter (the search query) is required"}), 400

    if not os.path.exists(DB_FILE_PATH) or not os.path.exists(FAISS_INDEX_PATH):
        missing_files = []
        if not os.path.exists(DB_FILE_PATH): missing_files.append(f"Database file {DB_FILE_PATH}")
        if not os.path.exists(FAISS_INDEX_PATH): missing_files.append(f"FAISS index file {FAISS_INDEX_PATH}")
        logger.error(f"Semantic search cannot proceed: {', '.join(missing_files)} not found.")
        return jsonify({"error": f"{', '.join(missing_files)} not found, cannot perform search."}), 500

    try:
        results = cosine_search_with_concepts(
            query=query,
            db_path=DB_FILE_PATH,
            index_path=FAISS_INDEX_PATH,
            k_faiss_retrieval=k_faiss_param,
            top_k_final=k_results,
            alpha=alpha_param
        )
        return jsonify(
            {"query": query, "alpha_used": alpha_param, "k_faiss_retrieval_used": k_faiss_param, "results": results})
    except FileNotFoundError as e:
        logger.error(f"File not found error during semantic search: {e}", exc_info=True)
        return jsonify({"error": "A required file for search was not found.", "details": str(e)}), 500
    except Exception as e:
        logger.error(f"Error during semantic search for query '{query}': {e}", exc_info=True)
        return jsonify({"error": "Semantic search operation failed", "details": str(e)}), 500


@app.route('/api/search/graph', methods=['GET'])
def graph_search_endpoint():
    query = request.args.get('q')
    k_results = request.args.get('k', default=5, type=int)  # Default to 5 results

    logger.info(f"API call: /api/search/graph - query='{query}', k={k_results}")

    if not query:
        return jsonify({"error": "'q' query parameter (the search query) is required"}), 400

    if not os.path.exists(DB_FILE_PATH):
        logger.error(f"Graph search cannot proceed: Database file {DB_FILE_PATH} not found.")
        return jsonify({"error": "Database file not found, cannot perform graph search."}), 500

    try:
        results = graph_semantic_search(
            query_text=query,
            k=k_results,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            db_path=DB_FILE_PATH
        )
        return jsonify({"query": query, "results": results})
    except Exception as e:
        logger.error(f"Error during graph semantic search for query '{query}': {e}", exc_info=True)
        return jsonify({"error": "Graph semantic search operation failed", "details": str(e)}), 500


@app.route('/api/search/combined', methods=['GET'])
def combined_search_endpoint():
    query = request.args.get('q')
    k_final_results = request.args.get('k', default=5, type=int)  # Final number of results to return
    k_candidates = request.args.get('k_candidates', default=10,
                                    type=int)  # Number of candidates to fetch from each search method
    rrf_k = request.args.get('rrf_k', default=60, type=int)  # RRF k parameter

    logger.info(f"API call: /api/search/combined - query='{query}', k_final={k_final_results}")

    if not query:
        return jsonify({"error": "'q' query parameter (the search query) is required"}), 400

    try:
        logger.info(f"Combined Search: Getting semantic results for query '{query}'")
        semantic_results = cosine_search_with_concepts(
            query=query,
            db_path=DB_FILE_PATH,
            index_path=FAISS_INDEX_PATH,
            top_k_final=k_candidates,
            alpha=0.3
        )

        logger.info(f"Combined Search: Getting graph results for query '{query}'")
        graph_results = graph_semantic_search(
            query_text=query,
            k=k_candidates,
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            db_path=DB_FILE_PATH
        )

        fused_scores = {}
        for rank, doc in enumerate(semantic_results):
            doc_id = doc['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rrf_k + rank + 1)

        for rank, doc in enumerate(graph_results):
            doc_id = doc['id']
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rrf_k + rank + 1)

        all_unique_docs_by_id = {doc['id']: doc for doc in semantic_results}
        for doc in graph_results:
            if doc['id'] not in all_unique_docs_by_id:
                all_unique_docs_by_id[doc['id']] = doc

        combined_results = []
        for doc_id, rrf_score in fused_scores.items():
            doc_details = all_unique_docs_by_id.get(doc_id)
            if doc_details:
                doc_details['rrf_score'] = rrf_score
                doc_details['found_by'] = []
                if any(d['id'] == doc_id for d in semantic_results):
                    doc_details['found_by'].append('semantic')
                if any(d['id'] == doc_id for d in graph_results):
                    doc_details['found_by'].append('graph')
                combined_results.append(doc_details)

        combined_results.sort(key=lambda x: x['rrf_score'], reverse=True)

        logger.info(
            f"Combined search fused {len(fused_scores)} unique documents and is returning top {k_final_results}.")

        return jsonify({"query": query, "results": combined_results[:k_final_results]})

    except Exception as e:
        logger.error(f"Error during combined search for query '{query}': {e}", exc_info=True)
        return jsonify({"error": "Combined search operation failed", "details": str(e)}), 500


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory: {data_dir}")
    if not os.path.exists(DB_FILE_PATH):
        logger.critical(f"CRITICAL: Database file {DB_FILE_PATH} not found.")
    else:
        logger.info(f"Using database file: {DB_FILE_PATH}")
    if not os.path.exists(FAISS_INDEX_PATH):
        logger.warning(f"WARNING: FAISS index file {FAISS_INDEX_PATH} not found. Semantic search will fail.")

    logger.info("Starting Flask API server for Romanian Traffic Code...")
    app.run(debug=True, host='0.0.0.0', port=5001)