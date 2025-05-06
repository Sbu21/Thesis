from typing import List, Tuple
import pandas as pd
from src.nlp import get_nlp


# Load Romanian spaCy model
nlp = get_nlp()

def extract_subjects(token):
    """
    Recursively find all subjects connected to a verb (handling conjunctions).
    """
    subjects = []
    for child in token.children:
        if child.dep_ in ("nsubj", "nsubj:pass"):
            subjects.append(child.text)
            # If subject has conjunctions (e.g., biciclete și motociclete)
            for subchild in child.children:
                if subchild.dep_ == "conj":
                    subjects.append(subchild.text)
    return subjects

def extract_objects(token):
    """
    Recursively find all objects connected to a verb (handling conjunctions).
    """
    objects = []
    for child in token.children:
        if child.dep_ in ("obj", "iobj", "obl", "attr"):
            objects.append(child.text)
            for subchild in child.children:
                if subchild.dep_ == "conj":
                    objects.append(subchild.text)
    return objects

def extract_svo_triples(text: str) -> list:
    """
    Extracts (subject, verb, object) triples from the text using dependency parsing.
    Normalizes verbs (lemmatization).
    """
    doc = nlp(text)
    triples = []

    for token in doc:
        # Find a verb
        if token.pos_ == "VERB":
            subjects = extract_subjects(token)
            objects = extract_objects(token)

            for subj in subjects:
                for obj in objects:
                    # Only save meaningful triples
                    if subj.strip() or obj.strip():
                        triples.append((subj.strip(), token.lemma_, obj.strip()))

    return triples


def extract_prepositional_triples(doc) -> List[Tuple[str, str, str]]:
    """
    Extracts (head, preposition, object) triples from prepositional phrases.
    """
    triples = []

    for token in doc:
        if token.dep_ == "prep" and token.head.pos_ in {"VERB", "NOUN", "ADJ"}:
            prep = token.text.lower()

            for pobj in token.children:
                if pobj.dep_ == "pobj" and pobj.pos_ in {"NOUN", "PROPN"}:
                    head = token.head.lemma_.lower()
                    obj = pobj.lemma_.lower()
                    relation = prep.replace(" ", "_")
                    triples.append((head, relation, obj))

    return triples


def extract_modal_constructions(doc) -> List[Tuple[str, str, str]]:
    """
    Extracts modal constructions like:
    - 'este obligatoriu să oprească'
    - 'este interzis să depășească'
    - 'este permis să acorde prioritate'
    Returns (subject, modal_relation, verb) triples.
    """
    triples = []

    for i, token in enumerate(doc):
        if token.lemma_ == "fi" and token.pos_ == "AUX":
            # Look for modal adjective like "obligatoriu", "interzis", "permis"
            for child in token.children:
                if child.dep_ == "acomp" and child.pos_ == "ADJ":
                    modal = child.lemma_.lower()
                    if modal in {"obligatoriu", "interzis", "permis"}:
                        # Find the verb that follows 'să'
                        for sub_token in doc[token.i:]:
                            if sub_token.text.lower() == "să":
                                for v in sub_token.children:
                                    if v.pos_ == "VERB":
                                        relation = f"{modal}_să"
                                        triples.append(("", relation, v.lemma_))
    return triples


def process_dependencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all articles/paragraphs to extract dependency triples.
    """
    df["svo_triples"] = df["text"].apply(lambda x: extract_svo_triples(x))
    return df
