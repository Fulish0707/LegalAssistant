import importlib
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager


class AuthFoundationTests(unittest.TestCase):
    MODULE_NAMES = ("app", "langchain_config", "llm_model", "llm_prompt")
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
                ("user-1", "google-subject-1", "first@example.com"),
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO users (id, google_subject) VALUES (?, ?)",
                    ("user-2", "google-subject-1"),
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

    def test_login_returns_safe_placeholder_when_google_oidc_is_configured(self):
        with self.imported_app_with_environment({
            "GOOGLE_OIDC_CLIENT_ID": "client-id",
            "GOOGLE_OIDC_CLIENT_SECRET": "client-secret",
            "GOOGLE_OIDC_REDIRECT_URI": "http://localhost:5005/auth/callback",
        }) as configured_app:
            response = configured_app.app.test_client().get("/auth/login")

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json, {"error": "google_oidc_login_not_implemented"})

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
