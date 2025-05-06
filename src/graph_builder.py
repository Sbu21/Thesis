import json
import sqlite3
import pandas as pd
import networkx as nx

def infer_relation_type(triple):
    subj, pred, obj = triple
    # Heuristic rules for relation types
    modal_keywords = {"poate", "trebuie", "este permis", "nu este permis", "obligat", "interzis"}
    prepositions = {"în", "la", "cu", "pentru", "prin", "de", "pe", "asupra", "sub"}

    if any(pred.startswith(modal) for modal in modal_keywords):
        return "modal"
    elif any(pred.startswith(prep) for prep in prepositions):
        return "prep"
    else:
        return "svo"

def build_legal_graph(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, svo_triples FROM articles", conn)
    conn.close()

    G = nx.DiGraph()

    for _, row in df.iterrows():
        try:
            triples = json.loads(row["svo_triples"])
            for triple in triples:
                if len(triple) != 3:
                    continue
                subj, pred, obj = triple
                relation = infer_relation_type(triple)

                # Add nodes and labeled edge
                G.add_node(subj)
                G.add_node(obj)
                G.add_edge(subj, obj, label=pred, relation=relation)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    return G
