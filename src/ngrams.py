import pandas as pd
from collections import Counter
from src.nlp import get_nlp

nlp = get_nlp()


def is_valid_token(token):
    """
    Filters tokens to exclude punctuation, stopwords, numbers, and verbs.
    """
    return (
            not token.is_stop and
            not token.is_punct and
            token.pos_ not in {"VERB", "NUM", "PRON", "SCONJ", "DET", "ADP"} and
            token.is_alpha
    )


def extract_ngrams(text, n=2, max_phrases=5):
    """
    Extract top n-grams (up to max_phrases) based on POS-filtered tokens.
    """
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if is_valid_token(token)]

    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngram_phrases = [" ".join(gram) for gram in ngrams]

    counts = Counter(ngram_phrases)
    most_common = [phrase for phrase, _ in counts.most_common(max_phrases)]
    return most_common


def extract_all_ngrams(text):
    """
    Extracts uni-, bi-, and tri-grams and returns a merged list.
    """
    all_ngrams = []
    for n in (1, 2, 3):
        all_ngrams.extend(extract_ngrams(text, n=n, max_phrases=3))
    return list(set(all_ngrams))  # remove duplicates


def process_ngrams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column of n-gram phrases to the DataFrame.
    """
    df["ngram_phrases"] = df["text"].apply(extract_all_ngrams)
    return df
