import yake
import pandas as pd

def extract_keywords(text: str, max_keywords: int = 5) -> list:
    """
    Extracts keywords using YAKE from the given text.
    """
    kw_extractor = yake.KeywordExtractor(
        lan="ro",
        n=3,            # Allow up to trigrams (3 words)
        dedupLim=0.9,
        top=max_keywords
    )
    keywords = kw_extractor.extract_keywords(text)
    keywords_only = [kw for kw, _ in keywords]
    return clean_keywords(keywords_only)

def clean_keywords(keywords: list) -> list:
    """
    Filters and cleans the extracted keywords: removes very short or meaningless ones.
    """
    stopwords = {"a", "în", "pe", "cu", "de", "şi", "sau", "la", "din", "pentru"}
    cleaned = []
    for kw in keywords:
        kw = kw.strip()
        if len(kw) >= 3 and kw.lower() not in stopwords:
            cleaned.append(kw)
    return cleaned

def process_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes all articles to extract cleaned keywords.
    """
    df["keywords"] = df["text"].apply(lambda x: extract_keywords(x))
    return df
