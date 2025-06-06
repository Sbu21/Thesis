import spacy
import pandas as pd
import logging  # Assuming logging is set up in main or you want it here

logger = logging.getLogger(__name__)

# Load Romanian spaCy model
# It's generally better to load the model once if possible,
# for example, by having get_nlp() manage a global instance or by passing it around.
# For now, keeping it as is, but consider optimizing model loading if performance is an issue.
try:
    nlp = spacy.load("ro_core_news_lg")
    logger.info("Romanian spaCy model 'ro_core_news_lg' loaded successfully.")
except OSError:
    logger.error(
        "spaCy model 'ro_core_news_lg' not found. Please download it by running: python -m spacy download ro_core_news_lg")
    # Depending on how critical this is, you might want to exit or raise an error
    nlp = None  # Set to None so get_nlp() can handle it if needed.


def get_nlp():
    """Returns the loaded spaCy NLP object."""
    if nlp is None:
        logger.critical("spaCy NLP model is not loaded. NLP functionality will be impaired.")
    return nlp


def process_text(text: str) -> dict:
    """
    Process a single text: tokenize, lemmatize, POS-tag, filter named entities.
    Named entities are now returned as a list of dictionaries: [{"text": ent.text, "type": ent.label_}].
    """
    if nlp is None:
        logger.error("Cannot process text: spaCy model not loaded.")
        return {
            "tokens": [], "lemmas": [], "pos_tags": [], "entities": []
        }

    if not isinstance(text, str):
        logger.warning(f"process_text received non-string input: {type(text)}. Returning empty results.")
        return {"tokens": [], "lemmas": [], "pos_tags": [], "entities": []}

    try:
        doc = nlp(text)
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]
        named_entities = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]

        return {
            "tokens": tokens,
            "lemmas": lemmas,
            "pos_tags": pos_tags,
            "entities": named_entities  # This now returns List[Dict[str, str]]
        }
    except Exception as e:
        logger.error(f"Error processing text with spaCy: '{str(text)[:100]}...'. Error: {e}", exc_info=True)
        return {  # Return empty structure on error to prevent downstream crashes
            "tokens": [], "lemmas": [], "pos_tags": [], "entities": []
        }


def process_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all articles with NLP and add linguistic annotations.
    The 'entities' column will now contain lists of {"text": ..., "type": ...} dicts.
    """
    if nlp is None:
        logger.error("Cannot process articles: spaCy model not loaded. Returning DataFrame as is.")
        # Ensure columns exist even if processing fails, to maintain schema consistency for db update
        for col in ["tokens", "lemmas", "pos_tags", "entities"]:
            if col not in df.columns:
                df[col] = pd.Series([[] for _ in range(len(df.index))], index=df.index, dtype=object)
        return df

    if "text" not in df.columns:
        logger.error("DataFrame for process_articles does not contain a 'text' column. Returning as is.")
        return df

    logger.info(f"Processing NLP for {len(df)} articles...")

    # Apply process_text and expand the resulting dictionary into new columns
    # Ensure that if 'text' is None or not a string, process_text handles it
    processed_data = df["text"].apply(
        lambda x: process_text(x) if pd.notna(x) and isinstance(x, str) else {"tokens": [], "lemmas": [],
                                                                              "pos_tags": [], "entities": []})

    df["tokens"] = processed_data.apply(lambda x: x.get("tokens", []))
    df["lemmas"] = processed_data.apply(lambda x: x.get("lemmas", []))
    df["pos_tags"] = processed_data.apply(lambda x: x.get("pos_tags", []))
    df["entities"] = processed_data.apply(lambda x: x.get("entities", []))  # This column now gets List[Dict]

    logger.info("NLP processing of articles complete.")
    return df


def preprocess_query(query: str) -> list:
    """
    Lemmatize and clean a query for concept overlap matching.
    """
    if nlp is None:
        logger.error("Cannot preprocess query: spaCy model not loaded.")
        return []
    if not isinstance(query, str):
        logger.warning(f"preprocess_query received non-string input: {type(query)}. Returning empty list.")
        return []

    try:
        doc = nlp(query)
        return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    except Exception as e:
        logger.error(f"Error preprocessing query '{query}': {e}", exc_info=True)
        return []