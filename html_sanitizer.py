import bleach


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
    return bleach.clean(
        value or "",
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols=["http", "https", "mailto"],
        strip=True,
    )
