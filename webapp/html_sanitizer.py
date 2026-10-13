try:
    import bleach  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for local environments
    bleach = None


ALLOWED_TAGS = [
    "a", "blockquote", "br", "code", "em", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "ol", "p", "pre", "strong", "table", "tbody", "td", "th", "thead", "tr", "ul",
]
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title", "target", "rel"],
    "td": ["colspan", "rowspan"],
    "th": ["colspan", "rowspan", "scope"],
}


def sanitize_report_html(value):
    text = value or ""

    if bleach is not None:
        return bleach.clean(
            text,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRIBUTES,
            protocols=["http", "https", "mailto"],
            strip=True,
        )

    # Minimal fallback for environments where bleach is unavailable.
    import html
    import re

    text = re.sub(r"<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\s*[^>]+>", "", text)
    text = html.unescape(text)
    return text.strip()
