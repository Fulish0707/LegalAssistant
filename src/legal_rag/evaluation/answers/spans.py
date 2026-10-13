"""Cut text into numbered spans that every answer grader selects from."""
import re

SPAN_MAX_CHARS = 600


def spans(text: str, *, include_whole: bool) -> list[str]:
    """Every returned span is an unchanged contiguous slice of the original."""
    spans = [text.strip()] if include_whole and text.strip() else []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        for start in range(0, len(part), SPAN_MAX_CHARS):
            span = part[start:start + SPAN_MAX_CHARS].strip()
            if span:
                spans.append(span)
    return list(dict.fromkeys(spans))
