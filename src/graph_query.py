import sqlite3
import networkx as nx
from typing import List, Tuple, Optional, Dict
import logging
from py2neo import Graph
from py2neo.errors import DatabaseError as Py2neoDatabaseError

try:
    from .nlp import preprocess_query
    from .db import create_connection as create_sqlite_connection, get_paragraph_details_by_db_id
except ImportError:
    from nlp import preprocess_query
    from db import create_connection as create_sqlite_connection, get_paragraph_details_by_db_id

logger = logging.getLogger(__name__)


def get_edges_from_node(
        G: nx.DiGraph,
        node: str,
        relation_category: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    if node not in G:
        logger.warning(f"Node '{node}' not found in the graph.")
        return []

    edges = []
    try:
        for target in G.successors(node):
            edge_data_dict = G.get_edge_data(node, target)
            if edge_data_dict:
                edge_category = edge_data_dict.get("category", "unknown_category")
                edge_predicate = edge_data_dict.get("predicate", "unknown_predicate")
                if relation_category is None or edge_category == relation_category:
                    edges.append((node, edge_predicate, target))
    except Exception as e:
        logger.error(f"Error getting edges from node '{node}': {e}", exc_info=True)

    return edges


def get_nodes_connected_to(
        G: nx.DiGraph,
        node: str,
        direction: str = 'out'
) -> List[str]:
    if node not in G:
        logger.warning(f"Node '{node}' not found in the graph for get_nodes_connected_to.")
        return []

    try:
        if direction == 'out':
            return list(G.successors(node))
        elif direction == 'in':
            return list(G.predecessors(node))
        else:
            logger.error(f"Invalid direction '{direction}' specified. Must be 'in' or 'out'.")
            return []
    except Exception as e:
        logger.error(f"Error getting connected nodes for '{node}' (direction '{direction}'): {e}", exc_info=True)
        return []


def print_subgraph(
        G: nx.DiGraph,
        start_node: str,
        depth: int = 1
):
    if start_node not in G:
        print(f"Node '{start_node}' not found.")
        return

    visited = set()
    queue: List[Tuple[str, int]] = [(start_node, 0)]

    while queue:
        current, current_depth = queue.pop(0)

        if current in visited or current_depth > depth:
            continue
        visited.add(current)

        indent = '  ' * current_depth
        print(f"{indent}- {current} (Depth {current_depth})")

        if current_depth < depth:
            for neighbor in G.successors(current):
                if neighbor not in visited:
                    queue.append((neighbor, current_depth + 1))


def find_paths_between_entities(
        G: nx.DiGraph,
        source: str,
        target: str,
        max_depth: int = 3
) -> List[List[Tuple[str, str, str]]]:
    if source not in G or target not in G:
        return []

    results = []
    try:
        simple_paths_edges = nx.all_simple_edge_paths(G, source=source, target=target, cutoff=max_depth)
        for edge_path in simple_paths_edges:
            path_with_relations = []
            for u, v in edge_path:
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    predicate = edge_data.get("predicate", "RELATED_TO")
                    path_with_relations.append((u, predicate, v))
                else:
                    path_with_relations.append((u, "UNKNOWN_RELATION", v))
            results.append(path_with_relations)
    except nx.NetworkXNoPath:
        logger.info(f"No path found between '{source}' and '{target}' within max_depth={max_depth}.")
        return []
    except Exception as e:
        logger.error(f"Error finding paths between '{source}' and '{target}': {e}", exc_info=True)
        return []
    return results


def search_by_article_number(
        article_header_str: str,
        paragraph_identifier_str: Optional[str] = None,
        *,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        db_path: str
) -> List[Dict]:
    logger.info(
        f"Searching by article: '{article_header_str}', paragraph: '{paragraph_identifier_str if paragraph_identifier_str else 'Any'}'")

    results = []
    graph_db_conn = None
    sqlite_conn = None

    if not article_header_str:
        logger.warning("Article header string cannot be empty for search_by_article_number.")
        return []

    try:
        graph_db_conn = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

        cypher_query = """
            MATCH (p:Paragraph)
            WHERE p.article_header = $article_header
        """
        params = {"article_header": article_header_str}

        if paragraph_identifier_str:
            cypher_query += " AND p.paragraph_identifier = $paragraph_identifier"
            params["paragraph_identifier"] = paragraph_identifier_str

        cypher_query += " RETURN p.db_id AS db_id ORDER BY p.db_id"

        neo4j_results = graph_db_conn.run(cypher_query, **params)
        found_records = list(neo4j_results)

        if not found_records:
            logger.info(
                f"No paragraphs found in Neo4j matching: Art: '{article_header_str}', Para: '{paragraph_identifier_str}'")
            return []

        logger.info(f"Found {len(found_records)} matching paragraph(s) in Neo4j. Fetching details from SQLite.")

        sqlite_conn = create_sqlite_connection(db_path)
        if not sqlite_conn:
            return []

        for record in found_records:
            db_id = record["db_id"]
            if db_id is not None:
                paragraph_details = get_paragraph_details_by_db_id(sqlite_conn, db_id)
                if paragraph_details:
                    results.append(paragraph_details)
                else:
                    logger.warning(f"Could not fetch details from SQLite for db_id {db_id}")
            else:
                logger.warning(f"Neo4j returned a Paragraph node without a db_id.")

    except Py2neoDatabaseError as e:
        logger.error(f"Neo4j database error during article number search: {e}", exc_info=True)
    except sqlite3.Error as e:
        logger.error(f"SQLite error during article number search: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during article number search: {e}", exc_info=True)
    finally:
        if sqlite_conn:
            sqlite_conn.close()

    return results


def graph_semantic_search(
    query_text: str,
    k: int,
    *,
    neo4j_uri: str, neo4j_user: str, neo4j_password: str,
    db_path: str
) -> List[Dict]:
    if not query_text or k <= 0:
        return []

    logger.info(f"Performing graph semantic search for query: '{query_text}', k={k}")

    query_terms = preprocess_query(query_text)
    if not query_terms:
        return []

    full_text_query_string = " OR ".join(query_terms)
    logger.debug(f"Processed query terms for graph search: {query_terms}, Full-text query string: '{full_text_query_string}'")

    results = []
    graph_db_conn = None
    sqlite_conn = None

    try:
        graph_db_conn = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

        cypher_query = """
            CALL db.index.fulltext.queryNodes("conceptNamesIndex", $query_string) YIELD node AS c, score AS text_score
            MATCH (p:Paragraph)-[:HAS_CONCEPT]->(c)
            WITH p, sum(text_score) AS match_score
            ORDER BY match_score DESC, p.db_id
            LIMIT $limit
            RETURN p.db_id AS db_id, 
                   p.article_header AS article_header, 
                   p.paragraph_identifier AS paragraph_identifier, 
                   match_score
        """
        params = {"query_string": full_text_query_string, "limit": k}

        logger.debug(f"Executing Cypher for graph search: {cypher_query} with params: {params}")
        neo4j_cursor = graph_db_conn.run(cypher_query, **params)

        paragraph_candidates = [
            {"db_id": record["db_id"], "graph_score": record["match_score"]}
            for record in neo4j_cursor if record["db_id"] is not None
        ]

        if not paragraph_candidates:
            # ...
            return []

        sqlite_conn = create_sqlite_connection(db_path)
        if not sqlite_conn:
            return []

        for candidate in paragraph_candidates:
            full_details = get_paragraph_details_by_db_id(sqlite_conn, candidate["db_id"])
            if full_details:
                full_details["graph_score"] = candidate["graph_score"]
                results.append(full_details)

    except Exception as e:
        logger.error(f"Unexpected error during graph semantic search for query '{query_text}': {e}", exc_info=True)
    finally:
        if sqlite_conn:
            sqlite_conn.close()

    return results