import sqlite3
import json
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


def create_connection(db_path: str) -> sqlite3.Connection:
    """ Create a database connection to the SQLite database specified by db_path """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Successfully connected to SQLite database: {db_path}")
    except sqlite3.Error as e:
        logger.error(f"Error connecting to SQLite database {db_path}: {e}", exc_info=True)
        raise  # Re-raise the exception if connection fails
    return conn


def create_table(conn: sqlite3.Connection):
    """ Create articles table if it doesn't exist """
    try:
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
        logger.info("Table 'articles' checked/created successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error creating table 'articles': {e}", exc_info=True)
        raise


def insert_articles(conn: sqlite3.Connection, articles: List[Dict]):
    """ Insert new articles into the articles table """
    if not articles:
        logger.warning("No articles provided for insertion.")
        return
    sql = ''' INSERT INTO articles(article, paragraph, text)
              VALUES(?,?,?) '''
    try:
        cursor = conn.cursor()
        # Corrected: Each item in 'articles' is a dict, so extract values
        data_to_insert = [(item['article'], item['paragraph'], item['text']) for item in articles]
        cursor.executemany(sql, data_to_insert)
        conn.commit()
        logger.info(f"Successfully inserted {len(articles)} new articles.")
    except sqlite3.Error as e:
        logger.error(f"Error inserting articles: {e}", exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"Missing key in article data for insertion: {e}", exc_info=True)
        raise


def reset_database(db_path: str):
    """ Resets the database by deleting the file. """
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Database '{db_path}' has been reset (deleted).")
        else:
            logger.info(f"Database '{db_path}' not found. No reset needed.")
    except OSError as e:
        logger.error(f"Error resetting database file {db_path}: {e}", exc_info=True)
        raise


def update_nlp_data(conn: sqlite3.Connection, df):
    """ Update articles table with NLP processed data """
    sql = ''' UPDATE articles
              SET tokens = ?,
                  lemmas = ?,
                  pos_tags = ?,
                  entities = ?
              WHERE id = ?'''
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute(sql, (
                json.dumps(row.get('tokens', []), ensure_ascii=False),
                json.dumps(row.get('lemmas', []), ensure_ascii=False),
                json.dumps(row.get('pos_tags', []), ensure_ascii=False),
                json.dumps(row.get('entities', []), ensure_ascii=False),
                row['id']
            ))
        conn.commit()
        logger.info(f"NLP data updated for {len(df)} rows.")
    except sqlite3.Error as e:
        logger.error(f"Error updating NLP data: {e}", exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"Missing 'id' or other NLP columns in DataFrame for update_nlp_data: {e}", exc_info=True)
        raise


def update_extractions(conn: sqlite3.Connection, df):
    """ Update articles table with extracted keywords """
    sql = ''' UPDATE articles
              SET keywords = ?
              WHERE id = ?'''
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute(sql, (
                json.dumps(row.get('keywords', []), ensure_ascii=False),
                row['id']
            ))
        conn.commit()
        logger.info(f"Keyword extractions updated for {len(df)} rows.")
    except sqlite3.Error as e:
        logger.error(f"Error updating keyword extractions: {e}", exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"Missing 'id' or 'keywords' column in DataFrame for update_extractions: {e}", exc_info=True)
        raise


def update_svo_data(conn: sqlite3.Connection, df):
    """ Update articles table with extracted SVO triples. """
    sql = ''' UPDATE articles
              SET svo_triples = ?
              WHERE id = ?'''
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            # Ensure 'svo_triples' exists and handle potential non-JSON serializable data
            svo_data = row.get('svo_triples')
            if isinstance(svo_data, str):  # If it's already a JSON string from main.py
                json_svo_data = svo_data
            else:  # If it's a list/dict, dump it
                json_svo_data = json.dumps(svo_data if svo_data is not None else [], ensure_ascii=False)

            cursor.execute(sql, (
                json_svo_data,
                row['id']
            ))
        conn.commit()
        logger.info(f"SVO triples updated for {len(df)} rows.")
    except sqlite3.Error as e:
        logger.error(f"Error updating SVO triples: {e}", exc_info=True)
        raise
    except (TypeError, OverflowError) as e:  # Catches errors if svo_data is not serializable
        logger.error(f"Error serializing SVO data for row ID {row.get('id', 'N/A')}: {e}", exc_info=True)
        # Decide if you want to skip this row or raise
    except KeyError as e:
        logger.error(f"Missing 'id' or 'svo_triples' column in DataFrame for update_svo_data: {e}", exc_info=True)
        raise


def update_ngram_data(conn: sqlite3.Connection, df):
    """ Update articles table with extracted n-gram phrases. """
    sql = ''' UPDATE articles
              SET ngram_phrases = ?
              WHERE id = ?'''
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute(sql, (
                json.dumps(row.get('ngram_phrases', []), ensure_ascii=False),
                row['id']
            ))
        conn.commit()
        logger.info(f"N-gram phrases updated for {len(df)} rows.")
    except sqlite3.Error as e:
        logger.error(f"Error updating n-gram phrases: {e}", exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"Missing 'id' or 'ngram_phrases' column in DataFrame for update_ngram_data: {e}", exc_info=True)
        raise


def update_concepts_data(conn: sqlite3.Connection, df):
    """ Update articles table with merged concepts. """
    sql = ''' UPDATE articles
              SET concepts = ?
              WHERE id = ?'''
    try:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute(sql, (
                json.dumps(row.get('concepts', []), ensure_ascii=False),
                row['id']
            ))
        conn.commit()
        logger.info(f"Concepts data updated for {len(df)} rows.")
    except sqlite3.Error as e:
        logger.error(f"Error updating concepts data: {e}", exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"Missing 'id' or 'concepts' column in DataFrame for update_concepts_data: {e}", exc_info=True)
        raise


def load_concepts_dict(db_path: str) -> dict:
    """ Returns a dictionary: {article_id: [concepts]} from the SQLite database. """
    conn = None
    try:
        conn = create_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, concepts FROM articles")
        results = cursor.fetchall()
        logger.info(f"Loaded concepts for {len(results)} articles from DB.")

        concept_dict = {}
        for row_data in results:
            try:
                concept_dict[row_data[0]] = json.loads(row_data[1]) if row_data[1] else []
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse concepts JSON for ID {row_data[0]}: {row_data[1]}. Error: {e}. Using empty list.")
                concept_dict[row_data[0]] = []
        return concept_dict
    except sqlite3.Error as e:
        logger.error(f"Error loading concepts dictionary from {db_path}: {e}", exc_info=True)
        return {}  # Return empty dict on error
    finally:
        if conn:
            conn.close()


def load_metadata(db_path: str) -> dict:
    """ Load article metadata: {id: {"article": ..., "paragraph": ..., "text": ...}} """
    conn = None
    try:
        conn = create_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, article, paragraph, text FROM articles")
        rows = cursor.fetchall()
        logger.info(f"Loaded metadata for {len(rows)} articles from DB.")
        return {
            row_data[0]: {"article": row_data[1], "paragraph": row_data[2], "text": row_data[3]}
            for row_data in rows
        }
    except sqlite3.Error as e:
        logger.error(f"Error loading metadata from {db_path}: {e}", exc_info=True)
        return {}  # Return empty dict on error
    finally:
        if conn:
            conn.close()


def load_svo_triples_from_db(db_path: str = "data/traffic_code.db") -> list:
    """ Load SVO triples from the database. """
    conn = None
    triples = []
    try:
        conn = create_connection(db_path)  # Use create_connection for consistent logging
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, svo_triples FROM articles WHERE svo_triples IS NOT NULL AND svo_triples != '' AND svo_triples != '[]'")
        rows = cursor.fetchall()

        for row_data in rows:  # Changed 'row' to 'row_data' to avoid conflict
            try:
                parsed_triples = json.loads(row_data[1])  # svo_triples is at index 1
                for item in parsed_triples:
                    if isinstance(item, (list, tuple)) and len(item) == 3:
                        # Ensure all elements are strings and not empty after stripping
                        if all(isinstance(t, str) and t.strip() for t in item):
                            triples.append(tuple(t.strip() for t in item))  # Also strip here
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse SVO triples JSON for ID {row_data[0]}: {row_data[1]}. Error: {e}")
            except Exception as e:  # Catch other potential errors during processing
                logger.warning(f"Generic error processing SVO triples for ID {row_data[0]}. Error: {e}", exc_info=True)
        logger.info(f"Loaded {len(triples)} valid SVO triples from {len(rows)} rows.")
        return triples
    except sqlite3.Error as e:
        logger.error(f"Database error loading SVO triples from {db_path}: {e}", exc_info=True)
        return []  # Return empty list on error
    finally:
        if conn:
            conn.close()

def get_paragraph_details_by_db_id(conn: sqlite3.Connection, db_id: int) -> Optional[Dict]:
    """
    Retrieves full details (article, paragraph, text) for a given database ID.
    """
    sql = "SELECT id, article, paragraph, text FROM articles WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (db_id,))
        row_data = cursor.fetchone()
        if row_data:
            logger.debug(f"Details fetched for db_id {db_id}: Article '{row_data[1]}', Paragraph '{row_data[2]}'")
            return {"id": row_data[0], "article": row_data[1], "paragraph": row_data[2], "text": row_data[3]}
        else:
            logger.warning(f"No paragraph found with db_id: {db_id}")
            return None
    except sqlite3.Error as e:
        logger.error(f"Error fetching paragraph details for db_id {db_id}: {e}", exc_info=True)
        return None


def get_distinct_article_headers_from_db(conn: sqlite3.Connection) -> List[str]:
    """
    Fetches a sorted list of unique article headers from the articles table.
    """
    logger.debug("Fetching distinct article headers from SQLite.")
    headers = []
    sql = "SELECT DISTINCT article FROM articles WHERE article IS NOT NULL AND article <> '' ORDER BY article"
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        headers = [row[0] for row in results]
        logger.info(f"Fetched {len(headers)} distinct article headers from SQLite.")
    except sqlite3.Error as e:
        logger.error(f"SQLite error fetching distinct article headers: {e}", exc_info=True)
    return headers


def get_paragraph_identifiers_for_article_from_db(
        conn: sqlite3.Connection, article_header_str: str
) -> List[str]:
    """
    Fetches a sorted list of unique, non-empty paragraph identifiers for a given article header.
    """
    logger.debug(f"Fetching paragraph identifiers for article: '{article_header_str}' from SQLite.")
    identifiers = []
    if not article_header_str:
        logger.warning("Article header string is required to fetch paragraph identifiers.")
        return []

    sql = """
        SELECT DISTINCT paragraph 
        FROM articles 
        WHERE article = ? AND paragraph IS NOT NULL AND paragraph <> '' 
        ORDER BY paragraph 
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (article_header_str,))
        results = cursor.fetchall()
        identifiers = [row[0] for row in results]
        logger.info(f"Fetched {len(identifiers)} paragraph identifiers for article '{article_header_str}' from SQLite.")
    except sqlite3.Error as e:
        logger.error(f"SQLite error fetching paragraph ids for '{article_header_str}': {e}", exc_info=True)
    return identifiers


def get_content_by_article_and_paragraph_from_db(
        conn: sqlite3.Connection,
        article_header_str: str,
        paragraph_identifier_str: Optional[str] = None
) -> List[Dict]:
    """
    Retrieves content (id, article, paragraph, text) from SQLite
    based on article header and optionally paragraph identifier.
    """
    logger.debug(
        f"Fetching content for article: '{article_header_str}', paragraph: '{paragraph_identifier_str if paragraph_identifier_str else 'ALL'}' from SQLite.")
    results_list = []
    if not article_header_str:
        logger.warning("Article header string cannot be empty.")
        return []

    sql_base = "SELECT id, article, paragraph, text FROM articles WHERE article = ?"
    params: List[str] = [article_header_str]

    if paragraph_identifier_str:
        sql_base += " AND paragraph = ?"
        params.append(paragraph_identifier_str)

    sql_base += " ORDER BY id"

    try:
        cursor = conn.cursor()
        cursor.execute(sql_base, tuple(params))
        rows = cursor.fetchall()
        for row_data in rows:
            results_list.append({
                "id": row_data[0],
                "article": row_data[1],
                "paragraph": row_data[2],
                "text": row_data[3]
            })
        logger.info(f"Fetched {len(results_list)} records for article '{article_header_str}'"
                    f"{f', paragraph_identifier_str {paragraph_identifier_str}' if paragraph_identifier_str else ''} from SQLite.")
    except sqlite3.Error as e:
        logger.error(f"SQLite error fetching content for article '{article_header_str}': {e}", exc_info=True)

    return results_list