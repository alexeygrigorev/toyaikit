import re


def strip_matching_outer_html_tags(text: str) -> str:
    """
    Strip the matching outer HTML tags from the text.

    Args:
        text (str): The text to strip the matching outer HTML tags from.

    Returns:
        str: The text with the matching outer HTML tags stripped.
    """
    match = re.match(r"^\s*<(\w+)[^>]*>\s*(.*?)\s*</\1>\s*$", text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return text.strip()
