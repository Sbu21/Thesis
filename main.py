import os
import json
import pandas as pd

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
from src.embeddings import generate_embeddings, build_faiss_index, save_index, cosine_search_with_concepts
from src.graph_builder import build_legal_graph
from src.neo4j_exporter import export_graph_to_neo4j

def main():
    txt_file_path = 'data/traffic_code.txt'
    db_file_path = 'data/traffic_code.db'

    # Step 1: Initialize DB
    if os.path.exists(db_file_path):
        print(f"Database '{db_file_path}' found. Loading...")
        conn = create_connection(db_file_path)
    else:
        print(f"Database '{db_file_path}' not found. Creating fresh database...")
        processed = process_traffic_code(txt_file_path)
        conn = create_connection(db_file_path)
        create_table(conn)
        insert_articles(conn, processed)
        print(f"Inserted {len(processed)} articles.")

    # Step 2: NLP Preprocessing
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    if "tokens" not in df.columns or df["tokens"].isnull().all():
        print("Performing NLP tokenization...")
        df_nlp = process_articles(df)
        update_nlp_data(conn, df_nlp)
    else:
        print("NLP already exists. Skipping...")

    # Step 3: Keyword Extraction
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    if "keywords" not in df.columns or df["keywords"].isnull().all():
        print("Extracting keywords...")
        df_keywords = process_keywords(df)
        update_extractions(conn, df_keywords)
    else:
        print("Keywords already extracted. Skipping...")

    # Step 4: Dependency Parsing
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    if "svo_triples" not in df.columns or df["svo_triples"].isnull().all():
        print("Extracting dependency...")
        all_rows = []

        for _, row in df.iterrows():
            text = row["text"]
            nlp = get_nlp()
            doc = nlp(text)
            svo = extract_svo_triples(doc)
            prep = extract_prepositional_triples(doc)
            modal = extract_modal_constructions(doc)
            combined = svo + prep + modal
            row["svo_triples"] = json.dumps(combined, ensure_ascii=False)
            all_rows.append(row)

        df_svo = pd.DataFrame(all_rows)
        update_svo_data(conn, df_svo)
    else:
        print("SVO triples already extracted. Skipping...")

    # Step 5: N-Gram Extraction
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    if "ngram_phrases" not in df.columns or df["ngram_phrases"].isnull().all():
        print("Extracting n-grams...")
        df_ngrams = process_ngrams(df)
        update_ngram_data(conn, df_ngrams)
    else:
        print("N-grams already exist. Skipping...")

    # Step 6: Merge Concepts
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    if "concepts" not in df.columns or df["concepts"].isnull().all():
        print("Merging concepts...")
        df_concepts = merge_concepts(df)
        update_concepts_data(conn, df_concepts)
    else:
        print("Concepts already merged. Skipping...")

    # Step 7: Embeddings + FAISS Index
    df = pd.read_sql_query("SELECT * FROM articles ORDER BY id", conn)
    if not os.path.exists("data/faiss_index.index"):
        print("Generating sentence embeddings...")
        texts = df["text"].tolist()
        ids = df["id"].tolist()
        embeddings = generate_embeddings(texts)

        print("Building FAISS index...")
        index = build_faiss_index(embeddings)
        save_index(index, "data/faiss_index.index")

        with open("data/faiss_row_ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)

        print("FAISS index and row mapping saved.")
    else:
        print("FAISS index exists. Skipping embedding step.")

    conn.close()
    print("All processing complete. System ready.")

    #Step 8: Graph generation and export to Neo4j
    G = build_legal_graph("data/traffic_code.db")
    print(f"✅ Legal graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    # Export to Neo4j
    export_graph_to_neo4j(G, uri="bolt://localhost:7687", user="neo4j", password="password")

if __name__ == '__main__':
    main()
