import pandas as pd
import json
import logging

try:
    from .nlp import get_nlp
except ImportError:
    from nlp import get_nlp

logger = logging.getLogger(__name__)
nlp_instance_concepts = None


def _initialize_nlp_concepts():
    """Initializes NLP instance for concept processing if not already done."""
    global nlp_instance_concepts
    if nlp_instance_concepts is None:
        nlp_instance_concepts = get_nlp()
        if nlp_instance_concepts:
            logger.info("spaCy NLP model initialized for concepts.py")
        else:
            logger.error("Failed to initialize spaCy NLP model in concepts.py. Lemmatization may not work as expected.")


def _lemmatize_phrase(phrase_text: str) -> str:
    """Lemmatizes a phrase, lowercases, and joins tokens, then strips."""
    if not phrase_text:
        return ""
    if not nlp_instance_concepts:
        logger.warning("NLP instance not available for lemmatization, returning only lowercased and stripped text.")
        return phrase_text.lower().strip()

    doc = nlp_instance_concepts(phrase_text.lower())
    lemmatized_tokens = [token.lemma_ for token in doc if
                         not token.is_punct and not token.is_space and token.lemma_.strip()]
    return " ".join(lemmatized_tokens).strip()


def _remove_subphrases(phrases: list) -> list:
    """
    Removes shorter phrases that are substrings of longer phrases in the list.
    Example: If ["bank", "bank account"] exists, "bank" will be removed.
    """
    # if not phrases:
    #     return []
    # to_remove = set()
    # for p1 in phrases:
    #     for p2 in phrases:
    #         if p1 == p2:
    #             continue
    #         if p1 in p2 and len(p1) < len(p2):
    #             to_remove.add(p1)
    # result = [phrase for phrase in phrases if phrase not in to_remove]
    # if len(phrases) > len(result) and logger.isEnabledFor(logging.DEBUG):
    #     phrases_removed_count = len(phrases) - len(result)
    #     example_removed = list(to_remove)[:5] if phrases_removed_count > 5 else list(to_remove)
    #     logger.debug(
    #         f"Sub-phrase removal: original {len(phrases)}, final {len(result)}. Removed {phrases_removed_count} concepts like: {example_removed}")
    # return result
    return phrases


def merge_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges YAKE keywords, N-grams, and (texts of) named entities into a unified 'concepts' list.
    - Keywords and N-grams are lemmatized and lowercased.
    - Entity texts are only lowercased and stripped.
    Removes duplicates, applies length filter, and removes shorter sub-phrases from the final list.
    """
    _initialize_nlp_concepts()  # Ensure NLP model is loaded for lemmatization

    expected_cols = ["keywords", "ngram_phrases", "entities"]

    if df.empty:
        logger.warning("Input DataFrame for merge_concepts is empty. Returning as is.")
        df["concepts"] = pd.Series([[] for _ in range(len(df))], dtype=object)
        return df

    for col in expected_cols:
        if col not in df.columns:
            logger.warning(
                f"Expected column '{col}' not found in DataFrame. It will be initialized as empty for concept merging.")
            df[col] = pd.Series([json.dumps([]) for _ in range(len(df))])

    def combine_and_clean_row_concepts(row):
        all_processed_phrases = set()

        row_id_info = f"Row ID {row.get('id', 'N/A') if isinstance(row, pd.Series) else 'Unknown'}"

        def process_column_phrases(column_name, lemmatize_phrases):
            current_items_json_str = row.get(column_name)
            deserialized_items = []
            try:
                if pd.isna(current_items_json_str):
                    pass
                elif isinstance(current_items_json_str, str):
                    try:
                        deserialized_items = json.loads(current_items_json_str)
                        if not isinstance(deserialized_items, list):
                            logger.warning(
                                f"{row_id_info}, Col '{column_name}': JSON content is not a list ('{str(current_items_json_str)[:50]}...').")
                            deserialized_items = []
                    except json.JSONDecodeError:
                        logger.warning(
                            f"{row_id_info}, Col '{column_name}': Failed to decode JSON string ('{str(current_items_json_str)[:50]}...').")
                elif isinstance(current_items_json_str, list):
                    deserialized_items = current_items_json_str
                else:
                    logger.warning(
                        f"{row_id_info}, Col '{column_name}': Content is not string, list, or NaN but {type(current_items_json_str)}.")
            except Exception as e:
                logger.error(f"{row_id_info}, Col '{column_name}': Error deserializing items: {e}", exc_info=True)
                return

            source_phrases = []
            if column_name == "entities":
                for item_val in deserialized_items:
                    if isinstance(item_val, dict) and "text" in item_val and isinstance(item_val["text"], str):
                        source_phrases.append(item_val["text"])
                    elif isinstance(item_val, str):
                        source_phrases.append(item_val)
            else:
                source_phrases = [str(item_val) for item_val in deserialized_items if
                                  isinstance(item_val, (str, int, float, bool))]

            for phrase_text in source_phrases:
                if not phrase_text.strip():
                    continue

                if lemmatize_phrases:
                    processed_phrase = _lemmatize_phrase(phrase_text)
                else:
                    processed_phrase = phrase_text.lower().strip()

                if len(processed_phrase) >= 3:
                    all_processed_phrases.add(processed_phrase)

        try:
            process_column_phrases("keywords", lemmatize_phrases=True)
            process_column_phrases("ngram_phrases", lemmatize_phrases=True)
            process_column_phrases("entities", lemmatize_phrases=False)
        except Exception as e:
            logger.error(f"{row_id_info}: Unexpected error during column processing loop: {e}", exc_info=True)

        unique_phrases_list = sorted(list(all_processed_phrases))
        final_cleaned_phrases = _remove_subphrases(unique_phrases_list)

        return final_cleaned_phrases

    try:
        logger.info("Applying concept merging with selective lemmatization to DataFrame rows...")
        df["concepts"] = df.apply(combine_and_clean_row_concepts, axis=1)
        logger.info("Successfully merged and processed concepts for the DataFrame.")
    except Exception as e:
        logger.error(f"Fatal error during the .apply() operation for merging concepts: {e}", exc_info=True)
        if "concepts" not in df.columns or not isinstance(df["concepts"], pd.Series):
            df["concepts"] = pd.Series([[] for _ in range(len(df.index))], index=df.index, dtype=object)
    return df