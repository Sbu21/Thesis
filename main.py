import os
import json
import sqlite3

import pandas as pd
import logging
import sys
from src.db import (
    create_connection, create_table, insert_articles,
    update_nlp_data, update_extractions, update_svo_data,
    update_ngram_data, update_concepts_data
)
from src.nlp import process_articles, get_nlp
from src.extract import process_keywords
from src.ngrams import process_ngrams
from src.preprocess import process_traffic_code
from src.dependency import extract_svo_triples, extract_prepositional_triples, extract_modal_constructions
from src.concepts import merge_concepts
from src.embeddings import generate_embeddings, build_faiss_index, save_index
from src.graph_builder import build_legal_graph
from src.neo4j_exporter import export_graph_to_neo4j

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/traffic_code_processing.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the Romanian Traffic Code processing pipeline.")
    txt_file_path = 'data/traffic_code.txt'
    db_file_path = 'data/traffic_code.db'
    faiss_index_path = "data/faiss_index.index"
    faiss_ids_path = "data/faiss_row_ids.json"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

    conn = None

    try:
        # Step 1: Initialize DB
        logger.info("Step 1: Initializing Database...")
        if os.path.exists(db_file_path):
            logger.info(f"Database '{db_file_path}' found. Loading...")
            conn = create_connection(db_file_path)
        else:
            logger.info(f"Database '{db_file_path}' not found. Creating fresh database...")
            if not os.path.exists(txt_file_path):
                logger.error(f"Raw data file not found: {txt_file_path}. Cannot create database.")
                return
            processed_articles = process_traffic_code(txt_file_path)
            if not processed_articles:
                logger.error("No articles processed from the text file. Aborting.")
                return
            conn = create_connection(db_file_path)
            create_table(conn)
            insert_articles(conn, processed_articles)
            logger.info(f"Inserted {len(processed_articles)} articles into the new database.")
        logger.info("Database initialization complete.")

        # Step 2: NLP Preprocessing
        logger.info("Step 2: NLP Preprocessing...")
        df = pd.read_sql_query("SELECT * FROM articles", conn)
        if "tokens" not in df.columns or df["tokens"].isnull().all():
            logger.info("Performing NLP tokenization, lemmatization, POS tagging, and NER...")
            df_nlp = process_articles(df.copy()) # Use .copy() to avoid SettingWithCopyWarning
            update_nlp_data(conn, df_nlp)
            logger.info("NLP data updated in the database.")
        else:
            logger.info("NLP data already exists in DB. Skipping NLP preprocessing.")
        logger.info("NLP Preprocessing step complete.")

        # Step 3: Keyword Extraction
        logger.info("Step 3: Keyword Extraction...")
        df = pd.read_sql_query("SELECT * FROM articles", conn) # Refresh DataFrame
        if "keywords" not in df.columns or df["keywords"].isnull().all():
            logger.info("Extracting keywords...")
            df_keywords = process_keywords(df.copy())
            update_extractions(conn, df_keywords)
            logger.info("Keywords updated in the database.")
        else:
            logger.info("Keywords already extracted. Skipping keyword extraction.")
        logger.info("Keyword Extraction step complete.")

        # Step 4: Dependency Parsing (SVO, Prepositional, Modal Triples)
        logger.info("Step 4: Dependency Parsing...")
        df = pd.read_sql_query("SELECT * FROM articles", conn) # Refresh DataFrame
        if "svo_triples" not in df.columns or df["svo_triples"].isnull().all():
            logger.info("Extracting SVO, prepositional, and modal triples...")
            all_rows_svo = []
            nlp_instance = get_nlp() # Load spaCy model once
            for index, row in df.iterrows():
                try:
                    text = row["text"]
                    doc = nlp_instance(text)
                    svo = extract_svo_triples(doc)
                    prep = extract_prepositional_triples(doc)
                    modal = extract_modal_constructions(doc)
                    combined_triples = svo + prep + modal
                    # Create a new dictionary or update a copy to avoid modifying iterator
                    updated_row = row.to_dict()
                    updated_row["svo_triples"] = json.dumps(combined_triples, ensure_ascii=False)
                    all_rows_svo.append(updated_row)
                except Exception as e:
                    logger.error(f"Error processing dependencies for row ID {row.get('id', 'N/A')}: {e}", exc_info=True)
                    # Add original row if processing fails to maintain DataFrame structure
                    all_rows_svo.append(row.to_dict())


            if all_rows_svo:
                df_svo = pd.DataFrame(all_rows_svo)
                update_svo_data(conn, df_svo)
                logger.info("Dependency triples (SVO, prepositional, modal) updated in the database.")
            else:
                logger.warning("No rows processed for SVO extraction.")
        else:
            logger.info("Dependency triples already extracted. Skipping dependency parsing.")
        logger.info("Dependency Parsing step complete.")

        # Step 5: N-Gram Extraction
        logger.info("Step 5: N-Gram Extraction...")
        df = pd.read_sql_query("SELECT * FROM articles", conn) # Refresh DataFrame
        if "ngram_phrases" not in df.columns or df["ngram_phrases"].isnull().all():
            logger.info("Extracting n-grams...")
            df_ngrams = process_ngrams(df.copy())
            update_ngram_data(conn, df_ngrams)
            logger.info("N-grams updated in the database.")
        else:
            logger.info("N-grams already exist. Skipping n-gram extraction.")
        logger.info("N-Gram Extraction step complete.")

        # Step 6: Merge Concepts
        logger.info("Step 6: Merging Concepts...")
        df = pd.read_sql_query("SELECT * FROM articles", conn) # Refresh DataFrame
        if "concepts" not in df.columns or df["concepts"].isnull().all():
            logger.info("Merging keywords, n-grams, and entities into concepts...")
            df_concepts = merge_concepts(df.copy())
            update_concepts_data(conn, df_concepts)
            logger.info("Concepts updated in the database.")
        else:
            logger.info("Concepts already merged. Skipping concept merging.")
        logger.info("Concept Merging step complete.")

        # Step 7: Embeddings + FAISS Index
        logger.info("Step 7: Embeddings and FAISS Index Creation...")
        df = pd.read_sql_query("SELECT * FROM articles ORDER BY id", conn) # Refresh DataFrame
        if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_ids_path):
            logger.info("FAISS index or ID mapping not found. Generating sentence embeddings and building FAISS index...")
            if df["text"].isnull().all():
                logger.error("No text data available in 'text' column to generate embeddings. Aborting FAISS step.")
            else:
                texts = df["text"].tolist()
                ids = df["id"].tolist()
                embeddings = generate_embeddings(texts)

                logger.info("Building FAISS index...")
                index = build_faiss_index(embeddings)
                save_index(index, faiss_index_path)
                logger.info(f"FAISS index saved to {faiss_index_path}")

                with open(faiss_ids_path, "w", encoding="utf-8") as f:
                    json.dump(ids, f, ensure_ascii=False)
                logger.info(f"FAISS row ID mapping saved to {faiss_ids_path}.")
        else:
            logger.info(f"FAISS index '{faiss_index_path}' and ID mapping '{faiss_ids_path}' exist. Skipping embedding generation and index building.")
        logger.info("Embeddings and FAISS Index Creation step complete.")


        # Step 8: Graph generation and export to Neo4j
        logger.info("Step 8: Graph Generation and Export to Neo4j...")
        try:
            G = build_legal_graph(db_file_path) # build_legal_graph needs the db_path
            logger.info(f"Legal graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
            export_graph_to_neo4j(G, uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        except Exception as e:
            logger.error(f"Failed during graph generation or Neo4j export: {e}", exc_info=True)
        logger.info("Graph Generation and Export to Neo4j step complete.")

    except pd.errors.DatabaseError as e: # More specific pandas error for SQL queries
        logger.error(f"A database error occurred with Pandas: {e}", exc_info=True)
    except sqlite3.Error as e:
        logger.error(f"An SQLite error occurred: {e}", exc_info=True)
    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main pipeline: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
        logger.info("Romanian Traffic Code processing pipeline finished.")

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
        logger.info("Created 'data' directory.")
    main()