from py2neo import Graph, Node, Relationship

def export_graph_to_neo4j(nx_graph, uri="bolt://localhost:7687", user="neo4j", password="password"):
    graph = Graph(uri, auth=(user, password))

    # Clear previous data
    graph.run("MATCH (n) DETACH DELETE n")

    tx = graph.begin()

    node_map = {}
    for node_id, data in nx_graph.nodes(data=True):
        label = data.get("label", "LegalEntity")
        node = Node(label, id=str(node_id), **data)
        tx.create(node)
        node_map[node_id] = node

    for source, target, data in nx_graph.edges(data=True):
        rel_type = data.get("type", "RELATED_TO").upper().replace(" ", "_")
        rel = Relationship(node_map[source], rel_type, node_map[target], **data)
        tx.create(rel)

    tx.commit()
    print(f"✅ Exported {len(nx_graph.nodes)} nodes and {len(nx_graph.edges)} edges to Neo4j.")
