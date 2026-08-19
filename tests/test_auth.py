import importlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch


class FakeGoogleClient:
    def __init__(self, token):
        self.token = token
        self.redirect_calls = []

    def authorize_redirect(self, redirect_uri, nonce):
        self.redirect_calls.append((redirect_uri, nonce))
        return "redirected"

    def authorize_access_token(self):
        return self.token


class FakeOAuth:
    def __init__(self, token):
        self.google = FakeGoogleClient(token)


class AuthFoundationTests(unittest.TestCase):
    MODULE_NAMES = ("app", "auth", "langchain_config", "llm_model", "llm_prompt")
    ENVIRONMENT_NAMES = (
        "APP_STATE_DB",
        "APP_ENV",
        "FLASK_ENV",
        "SECRET_KEY",
        "GOOGLE_OIDC_CLIENT_ID",
        "GOOGLE_OIDC_CLIENT_SECRET",
        "GOOGLE_OIDC_REDIRECT_URI",
    )

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.db_file = os.path.join(cls.tmpdir.name, "app_state.db")
        cls.original_environment = {
            name: os.environ.get(name)
            for name in cls.ENVIRONMENT_NAMES
        }
        cls.original_modules = {
            name: sys.modules.get(name)
            for name in cls.MODULE_NAMES
        }
        os.environ["APP_STATE_DB"] = cls.db_file
        os.environ["APP_ENV"] = "testing"
        os.environ.pop("FLASK_ENV", None)
        os.environ.pop("SECRET_KEY", None)
        for name in (
            "GOOGLE_OIDC_CLIENT_ID",
            "GOOGLE_OIDC_CLIENT_SECRET",
            "GOOGLE_OIDC_REDIRECT_URI",
        ):
            os.environ.pop(name, None)

        for module_name in cls.MODULE_NAMES:
            sys.modules.pop(module_name, None)

        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        for module_name in cls.MODULE_NAMES:
            sys.modules.pop(module_name, None)
            original_module = cls.original_modules[module_name]
            if original_module is not None:
                sys.modules[module_name] = original_module
        for name, value in cls.original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmpdir.cleanup()

    @classmethod
    @contextmanager
    def imported_app_with_environment(cls, overrides):
        original_environment = {
            name: os.environ.get(name)
            for name in cls.ENVIRONMENT_NAMES
        }
        original_modules = {
            name: sys.modules.get(name)
            for name in cls.MODULE_NAMES
        }
        try:
            for name, value in overrides.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            for module_name in cls.MODULE_NAMES:
                sys.modules.pop(module_name, None)
            yield importlib.import_module("app")
        finally:
            for module_name in cls.MODULE_NAMES:
                sys.modules.pop(module_name, None)
                original_module = original_modules[module_name]
                if original_module is not None:
                    sys.modules[module_name] = original_module
            for name, value in original_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_import_creates_user_and_daily_quota_tables(self):
        with sqlite3.connect(self.app_module.DB_FILE) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }

        self.assertIn("users", tables)
        self.assertIn("daily_ai_quota", tables)

    def test_schema_migration_is_idempotent_and_enforces_foundation_constraints(self):
        with sqlite3.connect(self.app_module.DB_FILE) as conn:
            self.app_module.ensure_database_schema(conn)
            self.app_module.ensure_database_schema(conn)

            conn.execute(
                "INSERT INTO users (id, google_subject, email) VALUES (?, ?, ?)",
                ("schema-user-1", "schema-google-subject-1", "first@example.com"),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO users (id, google_subject) VALUES (?, ?)",
                    ("schema-user-2", "schema-google-subject-1"),
                )

            conn.execute(
                """
                INSERT INTO daily_ai_quota (identity_kind, identity_value, quota_day)
                VALUES (?, ?, ?)
                """,
                ("user", "user-1", "2026-08-20"),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO daily_ai_quota (identity_kind, identity_value, quota_day)
                    VALUES (?, ?, ?)
                    """,
                    ("unknown", "user-1", "2026-08-20"),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO daily_ai_quota
                    (identity_kind, identity_value, quota_day, request_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("user", "user-2", "2026-08-20", -1),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO daily_ai_quota (identity_kind, identity_value, quota_day)
                    VALUES (?, ?, ?)
                    """,
                    ("user", "user-1", "2026-08-20"),
                )

    def test_login_reports_when_google_oidc_is_not_configured(self):
        response = self.client.get("/auth/login")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json, {"error": "google_oidc_not_configured"})

    def test_login_starts_google_authorization_with_relative_next_only(self):
        with self.imported_app_with_environment({
            "GOOGLE_OIDC_CLIENT_ID": "client-id",
            "GOOGLE_OIDC_CLIENT_SECRET": "client-secret",
            "GOOGLE_OIDC_REDIRECT_URI": "http://localhost:5005/auth/callback",
        }) as configured_app:
            fake_oauth = FakeOAuth({"userinfo": {"sub": "unused"}})
            client = configured_app.app.test_client()
            with patch.object(configured_app.auth, "oauth", fake_oauth):
                response = client.get(
                    "/auth/login?next=https://attacker.example/steal"
                )
            with client.session_transaction() as flask_session:
                next_path = flask_session.get("auth_next")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "redirected")
        self.assertEqual(fake_oauth.google.redirect_calls[0][0], "http://localhost:5005/auth/callback")
        self.assertIsNone(next_path)

    def test_callback_creates_internal_user_and_sets_only_account_identity(self):
        with self.imported_app_with_environment({
            "GOOGLE_OIDC_CLIENT_ID": "client-id",
            "GOOGLE_OIDC_CLIENT_SECRET": "client-secret",
            "GOOGLE_OIDC_REDIRECT_URI": "http://localhost:5005/auth/callback",
        }) as configured_app:
            client = configured_app.app.test_client()
            fake_oauth = FakeOAuth({
                "userinfo": {"sub": "google-subject-1", "email": "person@example.com"}
            })
            with client.session_transaction() as flask_session:
                flask_session["guest_draft"] = "discard me"
            with patch.object(configured_app.auth, "oauth", fake_oauth):
                response = client.get("/auth/callback")
            with client.session_transaction() as flask_session:
                user_id = flask_session.get("authenticated_user_id")
                session_values = dict(flask_session)

            with sqlite3.connect(configured_app.DB_FILE) as conn:
                row = conn.execute(
                    "SELECT id, google_subject, email FROM users WHERE google_subject=?",
                    ("google-subject-1",),
                ).fetchone()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
        self.assertIsNotNone(row)
        self.assertEqual(row[0], user_id)
        self.assertEqual(row[1:], ("google-subject-1", "person@example.com"))
        self.assertNotIn("guest_draft", session_values)
        self.assertEqual(
            set(session_values).intersection({"access_token", "id_token", "userinfo", "email", "google_subject"}),
            set(),
        )

    def test_repeat_callback_reuses_google_subject_and_updates_email(self):
        with self.imported_app_with_environment({
            "GOOGLE_OIDC_CLIENT_ID": "client-id",
            "GOOGLE_OIDC_CLIENT_SECRET": "client-secret",
            "GOOGLE_OIDC_REDIRECT_URI": "http://localhost:5005/auth/callback",
        }) as configured_app:
            client = configured_app.app.test_client()
            with patch.object(configured_app.auth, "oauth", FakeOAuth({
                "userinfo": {"sub": "google-subject-1", "email": "first@example.com"}
            })):
                client.get("/auth/callback")
            with sqlite3.connect(configured_app.DB_FILE) as conn:
                first_id = conn.execute(
                    "SELECT id FROM users WHERE google_subject=?", ("google-subject-1",)
                ).fetchone()[0]

            with patch.object(configured_app.auth, "oauth", FakeOAuth({
                "userinfo": {"sub": "google-subject-1", "email": "second@example.com"}
            })):
                response = client.get("/auth/callback")
            with sqlite3.connect(configured_app.DB_FILE) as conn:
                rows = conn.execute(
                    "SELECT id, email, last_login_at FROM users WHERE google_subject=?",
                    ("google-subject-1",),
                ).fetchall()

        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], first_id)
        self.assertEqual(rows[0][1], "second@example.com")
        self.assertTrue(rows[0][2])

    def test_callback_rejects_missing_subject_without_writing_account_identity(self):
        with self.imported_app_with_environment({
            "GOOGLE_OIDC_CLIENT_ID": "client-id",
            "GOOGLE_OIDC_CLIENT_SECRET": "client-secret",
            "GOOGLE_OIDC_REDIRECT_URI": "http://localhost:5005/auth/callback",
        }) as configured_app:
            client = configured_app.app.test_client()
            with sqlite3.connect(configured_app.DB_FILE) as conn:
                user_count_before = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            with patch.object(configured_app.auth, "oauth", FakeOAuth({"userinfo": {"email": "person@example.com"}})):
                response = client.get("/auth/callback")
            with client.session_transaction() as flask_session:
                user_id = flask_session.get("authenticated_user_id")
            with sqlite3.connect(configured_app.DB_FILE) as conn:
                user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

        self.assertEqual(response.status_code, 400)
        self.assertIsNone(user_id)
        self.assertEqual(user_count, user_count_before)

    def test_knowledge_base_status_requires_an_authenticated_account(self):
        response = self.client.get("/knowledge_base/status")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json, {"error": "authentication_required"})

    def test_logout_post_clears_account_and_guest_draft(self):
        with self.client.session_transaction() as flask_session:
            flask_session["authenticated_user_id"] = "user-1"
            flask_session["account_email"] = "person@example.com"
            flask_session["guest_draft"] = "temporary"

        get_response = self.client.get("/auth/logout")
        response = self.client.post("/auth/logout")
        with self.client.session_transaction() as flask_session:
            remaining = dict(flask_session)

        self.assertEqual(get_response.status_code, 405)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
        self.assertNotIn("authenticated_user_id", remaining)
        self.assertNotIn("account_email", remaining)
        self.assertNotIn("guest_draft", remaining)

    def test_unset_environment_requires_secret_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = os.environ.copy()
            environment["APP_STATE_DB"] = os.path.join(temp_dir, "app_state.db")
            for name in ("APP_ENV", "FLASK_ENV", "SECRET_KEY"):
                environment.pop(name, None)

            completed = subprocess.run(
                [sys.executable, "-c", "import app"],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("SECRET_KEY must be configured", completed.stderr)

    def test_unset_environment_uses_secure_session_cookie_with_secret(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = os.environ.copy()
            environment["APP_STATE_DB"] = os.path.join(temp_dir, "app_state.db")
            environment["SECRET_KEY"] = "production-secret"
            for name in ("APP_ENV", "FLASK_ENV"):
                environment.pop(name, None)

            completed = subprocess.run(
                [sys.executable, "-c", "import app; print(app.app.config['SESSION_COOKIE_SECURE'])"],
                cwd=os.path.dirname(os.path.dirname(__file__)),
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("True", completed.stdout)

    def test_explicit_development_and_testing_allow_non_secure_test_sessions(self):
        for environment_name in ("development", "testing"):
            with self.subTest(environment=environment_name), tempfile.TemporaryDirectory() as temp_dir:
                environment = os.environ.copy()
                environment["APP_STATE_DB"] = os.path.join(temp_dir, "app_state.db")
                environment["APP_ENV"] = environment_name
                environment.pop("FLASK_ENV", None)
                environment.pop("SECRET_KEY", None)

                completed = subprocess.run(
                    [sys.executable, "-c", "import app; print(app.app.config['SESSION_COOKIE_SECURE'])"],
                    cwd=os.path.dirname(os.path.dirname(__file__)),
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("False", completed.stdout)


if __name__ == "__main__":
    unittest.main()
