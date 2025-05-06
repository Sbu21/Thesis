import re
from typing import List, Tuple, Dict

def read_traffic_code(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_into_articles(text: str) -> List[Tuple[str, str]]:
    """
    Splits the text into (article_title, article_content) pairs.
    Detects only actual article headers like:
    - Art. 1
    - Art. 11.1.
    - Art. 45^1
    but skips references like "Art. 5 alin. (1)" in text.
    """
    # Match only if Art. is at the beginning of a line
    pattern = r'(?:^|\n)(Art\.?\s*\d+(?:[\.\^]\d+)?\.?)'

    matches = list(re.finditer(pattern, text))
    articles = []

    for i in range(len(matches)):
        start = matches[i].start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(text)
        header = matches[i].group(1).strip()
        content = text[start + len(header):end].strip()
        articles.append((header, content))

    return articles


def split_article_into_paragraphs(article_text: str) -> List[Tuple[str, str]]:
    """
    Splits article content into paragraphs using patterns like:
    (1), (1.1), (1.1.), (2.3.) etc., but only if they occur at line start.
    """
    # Matches (1), (1.1), (1.1.), etc. only if at beginning of line or after newline
    pattern = r'(?:^|\n)(\(\d+(?:\.\d+)?\.?\))'

    matches = list(re.finditer(pattern, article_text))
    if not matches:
        return [('', article_text.strip())]

    paragraphs = []
    for i in range(len(matches)):
        start = matches[i].start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(article_text)
        paragraph_number = matches[i].group(1)
        paragraph_text = article_text[start + len(paragraph_number):end].strip()
        paragraphs.append((paragraph_number, paragraph_text))

    return paragraphs

def process_traffic_code(file_path: str) -> List[Dict]:
    text = read_traffic_code(file_path)
    articles = split_into_articles(text)

    processed = []
    for article_header, article_text in articles:
        paragraphs = split_article_into_paragraphs(article_text)
        for paragraph_number, paragraph_text in paragraphs:
            processed.append({
                'article': article_header,
                'paragraph': paragraph_number,
                'text': paragraph_text
            })

    return processed
