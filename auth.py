"""Google OpenID Connect authentication helpers."""

from functools import wraps
from urllib.parse import urlsplit
import uuid

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.flask_client import OAuth
from flask import current_app, jsonify, redirect, request, session, url_for


oauth = None


def init_oauth(app):
    """Configure the Google OIDC client when all required settings are present."""
    global oauth
    oauth = None
    if not app.config["GOOGLE_OIDC_CONFIGURED"]:
        return

    oauth = OAuth(app)
    oauth.register(
        name="google",
        client_id=app.config["GOOGLE_OIDC_CLIENT_ID"],
        client_secret=app.config["GOOGLE_OIDC_CLIENT_SECRET"],
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def get_current_user_id():
    """Return the authenticated internal account ID, if any."""
    return session.get("authenticated_user_id")


def _safe_next_path(candidate):
    if not isinstance(candidate, str) or not candidate:
        return None
    if "\\" in candidate or any(ord(character) < 32 or ord(character) == 127 for character in candidate):
        return None
    parts = urlsplit(candidate)
    if parts.scheme or parts.netloc or not parts.path.startswith("/") or candidate.startswith("//"):
        return None
    return candidate


def _is_api_request():
    return (
        request.path.startswith(("/knowledge_base/", "/api/", "/statistics/"))
        or request.path in {
            "/process_document_ocr",
            "/transcribe_and_summarize_audio_file",
            "/transcribe_and_summarize_audio_blob",
        }
        or request.accept_mimetypes.best == "application/json"
    )


def login_required(view):
    """Require an account, returning JSON for API callers and redirecting browsers."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if get_current_user_id():
            return view(*args, **kwargs)
        if _is_api_request():
            return jsonify({"error": "authentication_required"}), 401
        return redirect(url_for("auth_login", next=_safe_next_path(request.full_path) or "/"))

    return wrapped


def upsert_google_user(subject, email):
    """Create or update an account keyed by Google subject and return its internal ID."""
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("Google account subject is required")

    from app import get_db

    normalized_subject = subject.strip()
    normalized_email = email.strip() if isinstance(email, str) and email.strip() else None
    connection = get_db()
    row = connection.execute(
        """
        INSERT INTO users (id, google_subject, email)
        VALUES (?, ?, ?)
        ON CONFLICT(google_subject) DO UPDATE SET
            email=excluded.email,
            last_login_at=CURRENT_TIMESTAMP
        RETURNING id
        """,
        (uuid.uuid4().hex, normalized_subject, normalized_email),
    ).fetchone()
    connection.commit()
    return row["id"]


def get_current_user_email():
    user_id = get_current_user_id()
    if not user_id:
        return None
    from app import get_db

    row = get_db().execute("SELECT email FROM users WHERE id=?", (user_id,)).fetchone()
    return row["email"] if row else None


def register_auth_routes(app):
    @app.route("/auth/login")
    def auth_login():
        if not app.config["GOOGLE_OIDC_CONFIGURED"] or oauth is None:
            return jsonify({"error": "google_oidc_not_configured"}), 503
        next_path = _safe_next_path(request.args.get("next"))
        if next_path:
            session["auth_next"] = next_path
        else:
            session.pop("auth_next", None)
        nonce = uuid.uuid4().hex
        session["oidc_nonce"] = nonce
        return oauth.google.authorize_redirect(app.config["GOOGLE_OIDC_REDIRECT_URI"], nonce=nonce)

    @app.route("/auth/callback")
    def auth_callback():
        if not app.config["GOOGLE_OIDC_CONFIGURED"] or oauth is None:
            return jsonify({"error": "google_oidc_not_configured"}), 503
        try:
            token = oauth.google.authorize_access_token()
            userinfo = token.get("userinfo", {})
            subject = userinfo.get("sub")
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError("Google account subject is missing")
            user_id = upsert_google_user(subject, userinfo.get("email"))
        except (OAuthError, ValueError, TypeError, AttributeError):
            current_app.logger.warning("Google OIDC callback failed", exc_info=True)
            return jsonify({"error": "google_oidc_authentication_failed"}), 400

        session.pop("oidc_nonce", None)
        session.pop("guest_draft", None)
        session["authenticated_user_id"] = user_id
        return redirect(_safe_next_path(session.pop("auth_next", None)) or "/")

    @app.route("/auth/logout", methods=["POST"])
    def auth_logout():
        for key in ("authenticated_user_id", "account_email", "guest_draft", "auth_next", "oidc_nonce"):
            session.pop(key, None)
        return redirect("/")
