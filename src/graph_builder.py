import json
import sqlite3
import pandas as pd
import networkx as nx
import logging

logger = logging.getLogger(__name__)

# Define relationship types as constants for clarity and consistency
REL_TYPE_HAS_CONCEPT = "HAS_CONCEPT"
REL_TYPE_MENTIONS_ENTITY = "MENTIONS_ENTITY"

# Define node labels as constants
NODE_LABEL_PARAGRAPH = "Paragraph"
NODE_LABEL_CONCEPT = "Concept"
NODE_LABEL_ENTITY = "Entity"
NODE_LABEL_TERM = "Term"  # For SVO components not otherwise typed


def infer_relation_category(predicate: str) -> str:
    """
    Heuristic rules for relation categories based on the predicate string.
    Used for SVO-like triples.
    """
    pred_lower = str(predicate).lower().strip()
    if not pred_lower:
        return "unknown_category"

    modal_keywords = {"poate", "trebuie", "este permis", "nu este permis", "obligat", "interzis", "permis",
                      "obligatoriu"}
    prepositions_as_predicates = {"în", "la", "cu", "pentru", "prin", "de", "pe", "asupra", "sub"}

    if any(modal_keyword in pred_lower for modal_keyword in modal_keywords):
        return "modal"
    if pred_lower in prepositions_as_predicates:
        return "prepositional"

    return "svo"


def _add_or_update_node(G, node_id_str, desired_label, properties=None):
    """
    Helper to add a node or update its label and properties.
    Precedence for labels: Entity > Concept > Term.
    If a node exists, its label is only upgraded or set if it's None or Term.
    """
    if properties is None:
        properties = {}

    # Ensure 'name' property is set based on node_id_str if not provided
    if 'name' not in properties:
        properties['name'] = node_id_str

    if node_id_str not in G:
        G.add_node(node_id_str, label=desired_label, **properties)
    else:
        # Node exists, check/update label and properties
        current_label = G.nodes[node_id_str].get('label')

        # Label update logic based on precedence
        update_label = False
        if desired_label == NODE_LABEL_ENTITY:
            if current_label in [NODE_LABEL_TERM, NODE_LABEL_CONCEPT, None]:
                update_label = True
        elif desired_label == NODE_LABEL_CONCEPT:
            if current_label in [NODE_LABEL_TERM, None]:
                update_label = True
        elif desired_label == NODE_LABEL_TERM:
            if current_label is None:
                update_label = True

        if update_label:
            G.nodes[node_id_str]['label'] = desired_label

        # Update other properties
        for key, value in properties.items():
            G.nodes[node_id_str][key] = value

        # Ensure 'name' property is present if it was somehow missed
        if 'name' not in G.nodes[node_id_str]:
            G.nodes[node_id_str]['name'] = node_id_str


def build_legal_graph(db_path: str) -> nx.DiGraph:
    """
    Builds a NetworkX DiGraph with Paragraph, Concept, Entity, and Term nodes,
    and relationships HAS_CONCEPT, MENTIONS_ENTITY, and SVO-predicate based relations.
    """
    logger.info(f"Building enhanced legal graph from database: {db_path}")
    conn = None
    G = nx.DiGraph()

    try:
        conn = sqlite3.connect(db_path)
        query = """
            SELECT id, article, paragraph, text, concepts, entities, svo_triples 
            FROM articles 
            WHERE id IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        logger.info(f"Retrieved {len(df)} rows from the database for graph building.")

        for _, row in df.iterrows():
            paragraph_db_id = row["id"]
            # Use a prefixed ID for paragraph nodes to ensure uniqueness across node types
            paragraph_node_nx_id = f"Paragraph_{paragraph_db_id}"

            # 1. Create Paragraph Node
            try:
                _add_or_update_node(G, paragraph_node_nx_id, NODE_LABEL_PARAGRAPH, {
                    "db_id": paragraph_db_id,
                    "article_header": row.get("article", ""),
                    "paragraph_identifier": row.get("paragraph", ""),
                    "text_snippet": str(row.get("text", ""))[:200] + "..."  # Store snippet
                })
                # Full text can be large, consider if it's needed directly on the node vs. lookup
            except Exception as e:
                logger.error(f"Error adding Paragraph node for db_id {paragraph_db_id}: {e}", exc_info=True)
                continue

            # 2. Process and Link Concepts
            try:
                concepts_json = row.get("concepts")
                if concepts_json and isinstance(concepts_json, str):
                    concepts_list = json.loads(concepts_json)
                    if isinstance(concepts_list, list):
                        for concept_text in concepts_list:
                            if isinstance(concept_text, str) and concept_text.strip():
                                concept_clean = concept_text.strip()
                                _add_or_update_node(G, concept_clean, NODE_LABEL_CONCEPT)
                                G.add_edge(paragraph_node_nx_id, concept_clean, type=REL_TYPE_HAS_CONCEPT)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Paragraph db_id {paragraph_db_id}: Error decoding concepts JSON: {str(concepts_json)[:100]}. Error: {e}")
            except Exception as e:
                logger.error(f"Paragraph db_id {paragraph_db_id}: Error processing concepts: {e}", exc_info=True)

            # 3. Process and Link Entities
            try:
                entities_json = row.get("entities")
                if entities_json and isinstance(entities_json, str):
                    entities_data_list = json.loads(entities_json)
                    if isinstance(entities_data_list, list):
                        for entity_data in entities_data_list:
                            if isinstance(entity_data, dict):
                                entity_text = entity_data.get("text")
                                entity_ner_type = entity_data.get("type", "UnknownType")
                                if entity_text and isinstance(entity_text, str) and entity_text.strip():
                                    entity_clean = entity_text.strip()
                                    _add_or_update_node(G, entity_clean, NODE_LABEL_ENTITY,
                                                        {"entity_type": entity_ner_type})
                                    G.add_edge(paragraph_node_nx_id, entity_clean, type=REL_TYPE_MENTIONS_ENTITY)
                            else:
                                logger.debug(
                                    f"Paragraph db_id {paragraph_db_id}: Entity item is not a dict: {entity_data}")
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Paragraph db_id {paragraph_db_id}: Error decoding entities JSON: {str(entities_json)[:100]}. Error: {e}")
            except Exception as e:
                logger.error(f"Paragraph db_id {paragraph_db_id}: Error processing entities: {e}", exc_info=True)

            # 4. Process SVO-like Triples
            try:
                svo_triples_json = row.get("svo_triples")
                if svo_triples_json and isinstance(svo_triples_json, str):
                    triples_list = json.loads(svo_triples_json)
                    if isinstance(triples_list, list):
                        for triple in triples_list:
                            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                                continue
                            s, p, o = triple
                            if not all(isinstance(x, str) and x.strip() for x in [s, p, o]):
                                continue

                            subj_clean = s.strip()
                            pred_clean = p.strip()
                            obj_clean = o.strip()

                            _add_or_update_node(G, subj_clean, NODE_LABEL_TERM)
                            _add_or_update_node(G, obj_clean, NODE_LABEL_TERM)

                            inferred_category = infer_relation_category(pred_clean)

                            G.add_edge(
                                subj_clean,
                                obj_clean,
                                predicate=pred_clean,  # For Neo4j rel type
                                category=inferred_category,
                                source_db_id=paragraph_db_id
                            )
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Paragraph db_id {paragraph_db_id}: Error decoding SVO JSON: {str(svo_triples_json)[:100]}. Error: {e}")
            except Exception as e:
                logger.error(f"Paragraph db_id {paragraph_db_id}: Error processing SVO triples: {e}", exc_info=True)

        logger.info(
            f"Enhanced legal graph construction complete. Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    except sqlite3.Error as e:
        logger.error(f"SQLite error while building legal graph: {e}", exc_info=True)
    except pd.errors.DatabaseError as e:
        logger.error(f"Pandas database error while fetching data for graph: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during legal graph construction: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed for graph builder.")

    return G