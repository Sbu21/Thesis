import networkx as nx
from typing import List, Tuple, Optional


def get_edges_from_node(
    G: nx.DiGraph,
    node: str,
    relation_type: Optional[str] = None
) -> List[Tuple[str, str, str]]:
    """
    Returns outgoing edges from a node, optionally filtered by relation type.
    """
    if node not in G:
        print(f"Node '{node}' not found in the graph.")
        return []

    edges = []
    for target in G.successors(node):
        edge_data = G.get_edge_data(node, target)
        if edge_data:
            rel_type = edge_data.get("type", "unknown")
            if relation_type is None or rel_type == relation_type:
                edges.append((node, rel_type, target))
    return edges


def get_nodes_connected_to(
    G: nx.DiGraph,
    node: str,
    direction: str = 'out'
) -> List[str]:
    """
    Returns directly connected nodes.
    `direction`: 'out' for successors, 'in' for predecessors
    """
    if node not in G:
        print(f"Node '{node}' not found in the graph.")
        return []

    if direction == 'out':
        return list(G.successors(node))
    elif direction == 'in':
        return list(G.predecessors(node))
    else:
        raise ValueError("Direction must be 'in' or 'out'")


def print_subgraph(
    G: nx.DiGraph,
    start_node: str,
    depth: int = 1
):
    """
    Prints the subgraph rooted at `start_node` up to a given depth.
    """
    if start_node not in G:
        print(f"Node '{start_node}' not found.")
        return

    visited = set()
    queue = [(start_node, 0)]

    while queue:
        current, current_depth = queue.pop(0)
        if current in visited or current_depth > depth:
            continue
        visited.add(current)

        indent = '  ' * current_depth
        print(f"{indent}- {current} (Depth {current_depth})")

        for neighbor in G.successors(current):
            queue.append((neighbor, current_depth + 1))

def find_paths_between_entities(G, source, target, max_depth=3):
    """
    Finds all simple paths (up to a certain depth) between two entities in the knowledge graph.
    Returns paths as lists of (node, relation, node) triplets.
    """
    if source not in G or target not in G:
        return []

    simple_paths = nx.all_simple_edge_paths(G, source=source, target=target, cutoff=max_depth)
    results = []
    for edge_path in simple_paths:
        path_with_relations = []
        for u, v in edge_path:
            relation = G[u][v].get("type", "unknown")
            path_with_relations.append((u, relation, v))
        results.append(path_with_relations)
    return results