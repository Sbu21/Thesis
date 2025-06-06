import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# It looks for "Art." or "articolul", followed by number, then optionally "alin." and "lit."
# It uses named capture groups (?P<group_name>...) to easily extract parts.
REFERENCE_PATTERN = re.compile(
    r"""
    (?:art\.?|articolul)\s+(?P<art>\d+([\.\^]\d+)?)
    (?:
        \s*,?\s+
        (?:alin\.?|alineatul)\s*\((?P<alin>\d+(\.\d+)?\.?)\)
        (?:
            \s*,?\s+
            lit\.?\s*(?P<lit>[a-z])\)
        )?
    )?
    """,
    re.IGNORECASE | re.VERBOSE
)


def _normalize_target(match_dict: Dict[str, Optional[str]]) -> str:
    """Creates a standardized identifier string from a regex match dictionary."""
    art = match_dict.get('art')
    alin = match_dict.get('alin')
    lit = match_dict.get('lit')

    target_parts = []
    if art:
        # Normalize article part, removing special chars for a clean ID
        art_clean = art.replace('^', '_').replace('.', '_')
        target_parts.append(f"Art_{art_clean}")
    if alin:
        # Normalize alin part
        alin_clean = alin.replace('.', '')
        target_parts.append(f"Alin_{alin_clean}")
    if lit:
        target_parts.append(f"Lit_{lit}")

    return "_".join(target_parts)


def extract_references(text: str) -> List[Dict]:
    """
    Extracts legal cross-references from a given text.

    Finds patterns like "Art. X alin. (Y) lit. z)" and returns structured data
    including the text mention, character offsets, and a normalized target identifier.

    Returns:
        A list of dictionaries, where each dict represents a found reference.
    """
    if not isinstance(text, str):
        return []

    found_references = []
    for match in REFERENCE_PATTERN.finditer(text):
        match_dict = match.groupdict()

        # We need a way to find the target article/paragraph in our DB.
        # Our DB stores "Art. X" and "(Y)" in separate columns.
        # The regex gives us the numbers. We need to reconstruct the likely DB identifiers.

        # Reconstruct how the target article header would likely be stored
        target_article_header = ""
        if match_dict.get('art'):
            # This reconstructs the format produced by your preprocess.py, e.g., "Art. 10" or "Art. 45^1"
            # It keeps the original format for lookup.
            target_article_header = f"Art. {match_dict['art']}".replace("Art. Art.",
                                                                        "Art.")  # Handle cases if Art. is already in regex

        # Reconstruct how the target paragraph identifier would be stored
        target_paragraph_identifier = ""
        if match_dict.get('alin'):
            target_paragraph_identifier = f"({match_dict['alin']})"
            # Handle cases like "(1.)" -> "(1)" if your DB format is cleaner
            if target_paragraph_identifier.endswith('.)'):
                target_paragraph_identifier = target_paragraph_identifier[:-2] + ')'

        # The text_mention is the full string that was matched by the regex
        text_mention = match.group(0)

        reference_data = {
            "text_mention": text_mention,
            "start_offset": match.start(),
            "end_offset": match.end(),
            # These can be used to query the database later
            "target_article_header": target_article_header,
            "target_paragraph_identifier": target_paragraph_identifier
        }
        found_references.append(reference_data)

    if found_references:
        logger.debug(f"Found {len(found_references)} cross-references in text snippet: '{text[:100]}...'")

    return found_references