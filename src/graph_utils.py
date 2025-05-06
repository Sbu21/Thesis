import sqlite3
import json
import networkx as nx


def normalize_modal_relation(modal_verb: str, negated: bool = False) -> str:
    modal_verb = modal_verb.lower()
    if modal_verb in ["trebuie", "este obligat", "se impune"]:
        return "obligation" if not negated else "prohibition"
    elif modal_verb in ["poate", "este permis", "are voie"]:
        return "permission" if not negated else "prohibition"
    elif modal_verb in ["nu poate", "nu este permis", "nu are voie"]:
        return "prohibition"
    return "unspecified"


def categorize_preposition(prep: str) -> str:
    prep = prep.lower()
    if prep in ["în", "la", "pe", "din"]:
        return "location"
    elif prep in ["pentru"]:
        return "purpose"
    elif prep in ["cu", "fără"]:
        return "manner"
    elif prep in ["dacă", "în caz de", "în condițiile"]:
        return "condition"
    return "unspecified"


def build_enriched_knowledge_graph(database_path: str) -> nx.DiGraph:
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, svo_triples FROM articles")
    rows = cursor.fetchall()

    G = nx.DiGraph()

    for article_id, triples_json in rows:
        if not triples_json:
            continue

        try:
            triples = json.loads(triples_json)
        except json.JSONDecodeError:
            continue

        for triple in triples:
            subj = triple.get("subject")
            obj = triple.get("object")
            rel_type = triple.get("type")
            label = triple.get("verb") or triple.get("prep") or "unknown"

            if not subj or not obj:
                continue

            # Normalize labels
            if rel_type == "modal":
                is_negated = "nu" in label.lower()
                label = normalize_modal_relation(label, negated=is_negated)
            elif rel_type == "prep":
                label = categorize_preposition(label)

            G.add_edge(subj, obj, label=label, relation_type=rel_type, article_id=article_id)

    conn.close()
    return G
