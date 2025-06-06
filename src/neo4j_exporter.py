from py2neo import Graph, Node, Relationship
from py2neo.errors import DatabaseError as Py2neoDatabaseError
import logging

logger = logging.getLogger(__name__)


def export_graph_to_neo4j(
        nx_graph,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
):
    """
    Exports a NetworkX graph with structured nodes and relationships to Neo4j.
    - Node labels are derived from the 'label' attribute in NetworkX node data.
    - Relationship types are derived from 'predicate' (for SVO-like) or 'type'
      (for structural like HAS_CONCEPT) attributes in NetworkX edge data.
    """
    if not nx_graph:
        logger.warning("NetworkX graph is empty or None. Nothing to export to Neo4j.")
        return

    logger.info(f"Attempting to connect to Neo4j at {uri}")
    graph_db = None  # Initialize to None
    try:
        graph_db = Graph(uri, auth=(user, password))
        graph_db.run("RETURN 1")  # Test connection
        logger.info("Successfully connected to Neo4j.")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j at {uri}: {e}", exc_info=True)
        return

    tx = None  # Initialize transaction
    try:
        logger.info("Clearing previous data from Neo4j...")
        graph_db.run("MATCH (n) DETACH DELETE n")
        logger.info("Previous Neo4j data cleared.")

        tx = graph_db.begin()
        logger.info("Neo4j transaction started.")

        node_map = {}  # Maps NetworkX node ID (string) to py2neo Node object

        # 1. Create Nodes in Neo4j
        logger.info(f"Processing {len(nx_graph.nodes())} nodes for Neo4j import...")
        created_node_count = 0
        for nx_node_id, nx_node_data in nx_graph.nodes(data=True):
            node_label_from_nx = nx_node_data.get("label", "DefaultNode")  # Default if 'label' is missing

            # Prepare properties for the Neo4j node
            # All attributes from nx_node_data, excluding 'label' if desired, though py2neo handles it
            # The 'name' property is crucial for Concept, Entity, Term.
            # 'db_id' is crucial for Paragraph.
            properties = dict(nx_node_data)  # Create a mutable copy
            properties.pop('label', None)  # Remove 'label' from properties if it exists

            # Ensure a primary identifying string representation if not 'name' (e.g. for Paragraphs)
            if 'name' not in properties and node_label_from_nx != "Paragraph":  # Paragraphs use db_id
                properties['id_str'] = str(nx_node_id)  # Store the NetworkX ID string if no name

            try:
                neo_node = Node(node_label_from_nx, **properties)
                tx.create(neo_node)
                node_map[nx_node_id] = neo_node
                created_node_count += 1
            except Exception as e:
                logger.error(f"Failed to create Neo4j node for NX_ID '{nx_node_id}' "
                             f"(Label: {node_label_from_nx}, Data: {properties}): {e}", exc_info=True)
        logger.info(f"Successfully staged {created_node_count} nodes for Neo4j transaction.")

        # 2. Create Relationships in Neo4j
        logger.info(f"Processing {len(nx_graph.edges())} edges for Neo4j import...")
        created_edge_count = 0
        for source_nx_id, target_nx_id, nx_edge_data in nx_graph.edges(data=True):
            if source_nx_id not in node_map or target_nx_id not in node_map:
                logger.warning(f"Skipping edge ({source_nx_id} -> {target_nx_id}) "
                               f"due to missing source or target node in node_map.")
                continue

            source_neo_node = node_map[source_nx_id]
            target_neo_node = node_map[target_nx_id]

            rel_type_base_str = ""
            rel_props = {}

            # Determine relationship type and properties based on edge attributes
            if "predicate" in nx_edge_data:  # SVO-like edge
                rel_type_base_str = str(nx_edge_data["predicate"])
                # Copy all other properties from nx_edge_data
                rel_props = {k: v for k, v in nx_edge_data.items() if k != "predicate"}
            elif "type" in nx_edge_data:  # Structural edge like HAS_CONCEPT, MENTIONS_ENTITY
                rel_type_base_str = str(nx_edge_data["type"])
                rel_props = {k: v for k, v in nx_edge_data.items() if k != "type"}
            else:
                logger.warning(f"Edge ({source_nx_id} -> {target_nx_id}) has no 'predicate' or 'type' "
                               f"attribute. Defaulting to 'RELATED_TO'. Data: {nx_edge_data}")
                rel_type_base_str = "RELATED_TO"
                rel_props = dict(nx_edge_data)  # Copy all data as properties

            # Sanitize relationship type for Neo4j
            rel_type_neo4j = rel_type_base_str.upper().replace(" ", "_").replace("-", "_").replace(".", "_")
            if not rel_type_neo4j:  # Handle cases where predicate might be e.g. punctuation
                rel_type_neo4j = "INTERACTS_WITH"  # A more specific default than RELATED_TO
                logger.debug(
                    f"Sanitized relationship type for '{rel_type_base_str}' was empty, using '{rel_type_neo4j}'.")

            try:
                neo_rel = Relationship(source_neo_node, rel_type_neo4j, target_neo_node, **rel_props)
                tx.create(neo_rel)
                created_edge_count += 1
            except Exception as e:
                logger.error(f"Failed to create Neo4j relationship "
                             f"{source_nx_id}-[{rel_type_neo4j}]->{target_nx_id} "
                             f"(Props: {rel_props}): {e}", exc_info=True)

        logger.info(f"Successfully staged {created_edge_count} relationships for Neo4j transaction.")

        tx.commit()
        logger.info(f"Exported {created_node_count} nodes and {created_edge_count} relationships to Neo4j. "
                    f"Transaction committed.")

    except Py2neoDatabaseError as e:
        logger.error(f"A Neo4j database error occurred during export: {e}", exc_info=True)
        if tx:
            try:
                tx.rollback()
                logger.info("Transaction rolled back due to Neo4j DatabaseError.")
            except Exception as rb_e:  # Use a different variable for the rollback exception
                logger.error(f"Failed to rollback transaction: {rb_e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during Neo4j export: {e}", exc_info=True)
        if tx:
            try:
                tx.rollback()
                logger.info("Transaction rolled back due to unexpected error.")
            except Exception as rb_e:
                logger.error(f"Failed to rollback transaction: {rb_e}", exc_info=True)