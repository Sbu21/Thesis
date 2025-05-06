import spacy
import pandas as pd

# Load Romanian spaCy model
nlp = spacy.load("ro_core_news_lg")

def get_nlp():
    return nlp

def process_text(text: str) -> dict:
    """
    Process a single text: tokenize, lemmatize, POS-tag, filter named entities.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    pos_tags = [token.pos_ for token in doc]
    named_entities = [ent.text for ent in doc.ents]

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": named_entities
    }

def process_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all articles with NLP and add linguistic annotations.
    """
    processed = df["text"].apply(process_text)
    df["tokens"] = processed.apply(lambda x: x["tokens"])
    df["lemmas"] = processed.apply(lambda x: x["lemmas"])
    df["pos_tags"] = processed.apply(lambda x: x["pos_tags"])
    df["entities"] = processed.apply(lambda x: x["entities"])
    return df

def preprocess_query(query: str):
    """
    Lemmatize and clean a query for concept overlap matching.
    """
    doc = nlp(query)
    return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
