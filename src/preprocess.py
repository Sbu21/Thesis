import re
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def read_traffic_code(file_path: str) -> str:
    """Reads the content of the traffic code text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Error: The file at {file_path} was not found.")
        return ""


def split_into_articles(text: str) -> List[Tuple[str, str]]:
    """
    Splits the raw text into a list of (article_header, article_content) tuples.
    This function remains largely the same.
    """
    # This regex is designed to match article headers only at the beginning of a line.
    pattern = r'(?:^|\n)(Art\.?\s*\d+(?:[\.\^]\d+)?\.?)'

    # Use re.finditer to get match objects with positions
    matches = list(re.finditer(pattern, text))
    articles = []

    if not matches:
        logger.warning("No article headers (e.g., 'Art. X') found in the text.")
        return []

    for i in range(len(matches)):
        match_start_pos = matches[i].start()
        # The header is the full matched text of the pattern.
        header = text[matches[i].start(1):matches[i].end(1)].strip()

        # Content starts after the current header match
        content_start_pos = matches[i].end()
        # Content ends at the start of the next header, or end of the text
        content_end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[content_start_pos:content_end_pos].strip()

        articles.append((header, content))

    logger.info(f"Segmented text into {len(articles)} articles.")
    return articles


def segment_article_content(article_header: str, article_text: str) -> List[Dict]:
    """
    Segments a single article's content into its constituent parts
    (paragraphs, points, letters) and creates hierarchical identifiers.
    """
    # Regex to find any of our structural markers:
    # 1. Aliniat: (1), (1.1), (1.1.)
    # 2. Punct: 1., 2., 1.1.
    # 3. Litera: a), b), a. (less common, but possible)
    # The pattern uses named groups to identify the type of marker found.
    marker_pattern = re.compile(
        r"""
        ^ # Start of a line
        (?:
            (?P<aliniat>\(\d+(?:\.\d+)?\.?\)) | # Group 'aliniat': e.g., (1), (1.1.)
            (?P<punct>\d+(?:\.\d+)?\.) |      # Group 'punct': e.g., 1., 1.1.
            (?P<litera>[a-z]\))              # Group 'litera': e.g., a), b)
        )
        """,
        re.MULTILINE | re.VERBOSE
    )

    matches = list(marker_pattern.finditer(article_text))
    segments = []

    if not matches:
        # If no markers are found, the entire article text is one segment with no paragraph id.
        if article_text:  # Only add if there is text
            segments.append({
                'article': article_header,
                'paragraph': '',  # No specific identifier
                'text': article_text
            })
        return segments

    # Handle any text that comes *before* the first marker
    first_match_start = matches[0].start()
    if first_match_start > 0:
        intro_text = article_text[:first_match_start].strip()
        if intro_text:
            segments.append({
                'article': article_header,
                'paragraph': '',  # This is introductory text for the article
                'text': intro_text
            })

    # Keep track of the current hierarchical context
    current_aliniat = ""
    current_punct = ""

    for i, match in enumerate(matches):
        marker_type = match.lastgroup  # This will be 'aliniat', 'punct', or 'litera'
        marker_text = match.group(marker_type).strip()

        # Determine the full hierarchical identifier for this segment
        hierarchical_id = ""
        if marker_type == 'aliniat':
            current_aliniat = marker_text
            current_punct = ""  # Reset punct when a new aliniat starts
            hierarchical_id = current_aliniat
        elif marker_type == 'punct':
            current_punct = f"pct. {marker_text.rstrip('.')}"
            if current_aliniat:
                hierarchical_id = f"{current_aliniat} {current_punct}"
            else:
                hierarchical_id = current_punct
        elif marker_type == 'litera':
            litera_id = f"lit. {marker_text.rstrip(')')}"
            if current_punct:
                hierarchical_id = f"{current_aliniat} {current_punct} {litera_id}" if current_aliniat else f"{current_punct} {litera_id}"
            elif current_aliniat:
                hierarchical_id = f"{current_aliniat} {litera_id}"
            else:
                hierarchical_id = litera_id  # Should be rare, but possible

        # Get the text content for this segment
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(article_text)
        text_content = article_text[content_start:content_end].strip()

        # Check if the text content itself is the introductory part of a list
        # e.g., for "Art. 49 (4) Limitele... sunt:", we want to separate this from "a) pe autostrazi..."
        sub_matches = list(marker_pattern.finditer(text_content))

        intro_text_of_segment = ""
        if sub_matches:
            # The text before the first sub-match is the introductory text for this segment
            intro_text_of_segment = text_content[:sub_matches[0].start()].strip()
        else:
            # There are no further sub-markers, so all text belongs to this segment
            intro_text_of_segment = text_content

        if intro_text_of_segment:
            segments.append({
                'article': article_header,
                'paragraph': hierarchical_id,
                'text': intro_text_of_segment
            })

    return segments


def process_traffic_code(file_path: str) -> List[Dict]:
    """
    The main orchestrator function for preprocessing the raw text file.
    Reads the file, splits it into articles, and then segments each article's
    content into granular, hierarchically identified parts.
    """
    logger.info(f"Starting to process traffic code from file: {file_path}")
    raw_text = read_traffic_code(file_path)
    if not raw_text:
        return []

    articles = split_into_articles(raw_text)

    processed_segments = []
    for article_header, article_text in articles:
        # Use the new segmentation function
        article_segments = segment_article_content(article_header, article_text)
        processed_segments.extend(article_segments)

    logger.info(
        f"Completed processing. Found {len(processed_segments)} distinct text segments (paragraphs, points, letters).")
    return processed_segments