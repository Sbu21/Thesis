import pandas as pd
import json

def merge_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges YAKE keywords, N-grams, and named entities into a unified 'concepts' list.
    Removes duplicates and normalizes phrases.
    """
    def combine(row):
        all_phrases = []

        for col in ["keywords", "ngram_phrases", "entities"]:
            try:
                items = row[col]
                if isinstance(items, str):
                    items = json.loads(items)
                all_phrases.extend(items)
            except:
                pass

        # Normalize: strip, lowercase, remove duplicates
        cleaned = list(set([phrase.strip().lower() for phrase in all_phrases if len(phrase.strip()) >= 3]))
        return cleaned

    df["concepts"] = df.apply(combine, axis=1)
    return df
