"""Session foundation for the public browser workspaces.

Covers the SECRET_KEY / cookie-security contract that survived the removal of
Google sign-in, the schema behaviour for new versus existing installs, and the
negative regressions: the old authentication routes are gone and the chat page
carries no sign-in strip.
"""

import importlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest


class BrowserSessionTests(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")
    ENVIRONMENT_NAMES = (
        "APP_STATE_DB",
        "AGENT_CHECKPOINT_DB",
        "RAG_PERSIST_DIR",
        "APP_ENV",
        "FLASK_ENV",
        "SECRET_KEY",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "LANGSMITH_TRACING",
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
        os.environ["AGENT_CHECKPOINT_DB"] = os.path.join(cls.tmpdir.name, "checkpoints.db")
        os.environ["RAG_PERSIST_DIR"] = os.path.join(cls.tmpdir.name, "chroma")
        os.environ["APP_ENV"] = "testing"
        os.environ["LLM_PROVIDER"] = "none"
        os.environ["EMBEDDING_PROVIDER"] = "none"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ.pop("FLASK_ENV", None)
        os.environ.pop("SECRET_KEY", None)

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

    def test_new_installs_create_quota_but_no_users_table(self):
        with sqlite3.connect(self.app_module.DB_FILE) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }

        self.assertNotIn("users", tables)
        self.assertIn("daily_ai_quota", tables)
        self.assertIn("chat_history", tables)

    def test_existing_installs_keep_their_users_table_across_schema_calls(self):
        existing_db = os.path.join(self.tmpdir.name, "existing_install.db")
        with sqlite3.connect(existing_db) as conn:
            conn.execute(
                """
                CREATE TABLE users (
                    id TEXT PRIMARY KEY,
                    google_subject TEXT NOT NULL UNIQUE,
                    email TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "INSERT INTO users (id, google_subject, email) VALUES (?, ?, ?)",
                ("legacy-user-1", "legacy-google-subject-1", "legacy@example.com"),
            )
            conn.commit()

            self.app_module.ensure_database_schema(conn)
            self.app_module.ensure_database_schema(conn)

            rows = conn.execute("SELECT id, google_subject, email FROM users").fetchall()
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }

        self.assertEqual(rows, [("legacy-user-1", "legacy-google-subject-1", "legacy@example.com")])
        self.assertIn("daily_ai_quota", tables)

    def test_schema_migration_is_idempotent_and_enforces_quota_constraints(self):
        with sqlite3.connect(self.app_module.DB_FILE) as conn:
            self.app_module.ensure_database_schema(conn)
            self.app_module.ensure_database_schema(conn)

            conn.execute(
                """
                INSERT INTO daily_ai_quota (identity_kind, identity_value, quota_day)
                VALUES (?, ?, ?)
                """,
                ("guest_ip", "hashed-ip-1", "2026-08-20"),
            )
            # Legacy rows written by the old account quota stay valid.
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
                    ("guest_ip", "hashed-ip-2", "2026-08-20", -1),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO daily_ai_quota (identity_kind, identity_value, quota_day)
                    VALUES (?, ?, ?)
                    """,
                    ("guest_ip", "hashed-ip-1", "2026-08-20"),
                )

    def test_google_routes_are_removed(self):
        client = self.app_module.app.test_client()
        for method, path in (("get", "/auth/login"), ("get", "/auth/callback"), ("post", "/auth/logout")):
            with self.subTest(path=path):
                self.assertEqual(getattr(client, method)(path).status_code, 404)

    def test_chat_page_has_no_google_login(self):
        response = self.app_module.app.test_client().get("/chatbot.html")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('id="auth-bar"', response.get_data(as_text=True))
        self.assertNotIn("Sign in with Google", response.get_data(as_text=True))
        self.assertIn('id="app"', response.get_data(as_text=True))

    def test_root_page_renders_for_every_browser(self):
        response = self.app_module.app.test_client().get("/")
        self.assertEqual(response.status_code, 200)

    def test_health_no_longer_reports_oidc_readiness(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("google_oidc_configured", response.json)

    def test_obsolete_account_cookie_keys_are_dropped_on_the_next_request(self):
        client = self.app_module.app.test_client()
        with client.session_transaction() as flask_session:
            flask_session["user_id"] = "existing-browser-workspace"
            flask_session["authenticated_user_id"] = "old-account"
            flask_session["account_email"] = "person@example.com"
            flask_session["auth_next"] = "/chatbot.html"
            flask_session["oidc_nonce"] = "nonce"

        client.get("/health")
        with client.session_transaction() as flask_session:
            remaining = dict(flask_session)

        self.assertEqual(remaining, {"user_id": "existing-browser-workspace"})

    def test_unset_environment_requires_secret_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = os.environ.copy()
            environment["APP_STATE_DB"] = os.path.join(temp_dir, "app_state.db")
            environment["AGENT_CHECKPOINT_DB"] = os.path.join(temp_dir, "checkpoints.db")
            # Empty, not removed: app.py's load_dotenv() fills in variables that are missing.
            for name in ("APP_ENV", "FLASK_ENV", "SECRET_KEY"):
                environment[name] = ""

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
            environment["AGENT_CHECKPOINT_DB"] = os.path.join(temp_dir, "checkpoints.db")
            environment["SECRET_KEY"] = "production-secret"
            for name in ("APP_ENV", "FLASK_ENV"):
                environment[name] = ""

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
                environment["AGENT_CHECKPOINT_DB"] = os.path.join(temp_dir, "checkpoints.db")
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
