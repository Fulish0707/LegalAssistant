import importlib
import os
import sqlite3
import sys
import tempfile
import unittest


class AuthFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.db_file = os.path.join(cls.tmpdir.name, "app_state.db")
        cls.original_environment = {
            name: os.environ.get(name)
            for name in (
                "APP_STATE_DB",
                "GOOGLE_OIDC_CLIENT_ID",
                "GOOGLE_OIDC_CLIENT_SECRET",
                "GOOGLE_OIDC_REDIRECT_URI",
            )
        }
        os.environ["APP_STATE_DB"] = cls.db_file
        for name in (
            "GOOGLE_OIDC_CLIENT_ID",
            "GOOGLE_OIDC_CLIENT_SECRET",
            "GOOGLE_OIDC_REDIRECT_URI",
        ):
            os.environ.pop(name, None)

        for module_name in ["app", "langchain_config", "llm_model", "llm_prompt"]:
            sys.modules.pop(module_name, None)

        cls.app_module = importlib.import_module("app")
        cls.client = cls.app_module.app.test_client()

    @classmethod
    def tearDownClass(cls):
        for name, value in cls.original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls.tmpdir.cleanup()

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

    def test_login_reports_when_google_oidc_is_not_configured(self):
        response = self.client.get("/auth/login")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json, {"error": "google_oidc_not_configured"})


if __name__ == "__main__":
    unittest.main()
