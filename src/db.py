import sqlite3
import json
from typing import List, Dict

def create_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    return conn

def create_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article TEXT,
            paragraph TEXT,
            text TEXT,
            tokens TEXT,
            lemmas TEXT,
            pos_tags TEXT,
            keywords TEXT,
            entities TEXT,
            svo_triples TEXT,
            ngram_phrases TEXT,
            concepts TEXT
        )
    ''')
    conn.commit()

def insert_articles(conn: sqlite3.Connection, articles: List[Dict]):
    cursor = conn.cursor()
    for item in articles:
        cursor.execute('''
            INSERT INTO articles (article, paragraph, text)
            VALUES (?, ?, ?)
        ''', (item['article'], item['paragraph'], item['text']))
    conn.commit()

def reset_database(db_path: str):
    import os
    if os.path.exists(db_path):
        os.remove(db_path)

def update_nlp_data(conn: sqlite3.Connection, df):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
            UPDATE articles
            SET tokens = ?, lemmas = ?, pos_tags = ?, entities = ?
            WHERE id = ?
        ''', (
            json.dumps(row['tokens'], ensure_ascii=False),
            json.dumps(row['lemmas'], ensure_ascii=False),
            json.dumps(row['pos_tags'], ensure_ascii=False),
            json.dumps(row['entities'], ensure_ascii=False),
            row['id']
        ))
    conn.commit()

def update_extractions(conn: sqlite3.Connection, df):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
            UPDATE articles
            SET keywords = ?
            WHERE id = ?
        ''', (
            json.dumps(row['keywords'], ensure_ascii=False),
            row['id']
        ))
    conn.commit()


def update_svo_data(conn: sqlite3.Connection, df):
    """
    Update articles table with extracted SVO triples.
    """
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
            UPDATE articles
            SET svo_triples = ?
            WHERE id = ?
        ''', (
            row['svo_triples'],
            row['id']
        ))
    conn.commit()

def update_ngram_data(conn: sqlite3.Connection, df):
    """
    Update articles table with extracted n-gram phrases.
    """
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
            UPDATE articles
            SET ngram_phrases = ?
            WHERE id = ?
        ''', (
            json.dumps(row['ngram_phrases'], ensure_ascii=False),
            row['id']
        ))
    conn.commit()


def update_concepts_data(conn: sqlite3.Connection, df):
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute('''
            UPDATE articles
            SET concepts = ?
            WHERE id = ?
        ''', (
            json.dumps(row['concepts'], ensure_ascii=False),
            row['id']
        ))
    conn.commit()

def load_concepts_dict(db_path: str) -> dict:
    """
    Returns a dictionary: {article_id: [concepts]} from the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, concepts FROM articles")
    results = cursor.fetchall()
    conn.close()

    return {
        row[0]: json.loads(row[1]) if row[1] else [] for row in results
    }

def load_metadata(db_path: str) -> dict:
    """
    Load article metadata: {id: {"article": ..., "paragraph": ..., "text": ...}}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, article, paragraph, text FROM articles")
    rows = cursor.fetchall()
    conn.close()

    return {
        row[0]: {"article": row[1], "paragraph": row[2], "text": row[3]}
        for row in rows
    }

def load_svo_triples_from_db(db_path: str = "data/traffic_code.db") -> list:
    import sqlite3
    import json

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT svo_triples FROM articles WHERE svo_triples IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    triples = []
    for row in rows:
        try:
            parsed = json.loads(row[0])
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    if all(isinstance(t, str) and t.strip() for t in item):
                        triples.append(tuple(item))
        except Exception as e:
            print("Failed to parse row:", row[0], "\nError:", e)

    return triples

