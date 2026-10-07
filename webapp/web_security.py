"""Small, fail-closed request controls for the browser service."""

import secrets
import time
from urllib.parse import urlsplit

from flask import abort, jsonify, request, session


def require_text(value, *, limit, allow_empty=False):
    if not isinstance(value, str) or len(value) > limit or (not allow_empty and not value.strip()):
        abort(400, description="invalid_input")
    return value.strip()


def csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def check_browser_request():
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    supplied = request.headers.get("X-CSRF-Token", "")
    expected = session.get("csrf_token", "")
    if not expected or not secrets.compare_digest(supplied.encode(), expected.encode()):
        abort(403, description="csrf_failed")
    if request.headers.get("Sec-Fetch-Site") == "cross-site":
        abort(403, description="cross_site_request")
    origin = request.headers.get("Origin")
    try:
        parsed = urlsplit(origin) if origin else None
    except ValueError:
        abort(403, description="origin_mismatch")
    if parsed and (parsed.scheme != request.scheme or parsed.netloc != request.host):
        abort(403, description="origin_mismatch")


def reserve_burst(conn, identity, *, limit=10, window=60):
    bucket = int(time.time()) // window
    conn.execute("CREATE TABLE IF NOT EXISTS request_burst (identity TEXT, bucket INTEGER, used INTEGER NOT NULL, PRIMARY KEY(identity,bucket))")
    row = conn.execute(
        "INSERT INTO request_burst VALUES (?,?,1) ON CONFLICT(identity,bucket) DO UPDATE SET used=used+1 WHERE used<? RETURNING used",
        (identity, bucket, limit),
    ).fetchone()
    conn.execute("DELETE FROM request_burst WHERE bucket < ?", (bucket - 2,))
    conn.commit()
    return row is not None


def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    if request.is_secure:
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    if request.endpoint != "static":
        response.headers["Cache-Control"] = "no-store"
    return response


def public_error(error):
    status = getattr(error, "code", 500) or 500
    code = {400: "invalid_input", 403: "request_forbidden", 405: "method_not_allowed", 413: "payload_too_large", 429: "rate_limit_exceeded"}.get(status, "request_failed")
    return jsonify({"error": code}), status
