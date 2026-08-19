import os
import subprocess
import sys
import tempfile
import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_importing_app_completes_without_microphone_initialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = os.path.dirname(os.path.dirname(__file__))
            for module_name in ("sounddevice", "soundfile"):
                module_path = os.path.join(temp_dir, f"{module_name}.py")
                with open(module_path, "w", encoding="utf-8") as module_file:
                    module_file.write(
                        "raise RuntimeError('MICROPHONE_IMPORT_SENTINEL')\n"
                    )

            environment = os.environ.copy()
            environment["APP_STATE_DB"] = os.path.join(temp_dir, "app_state.db")
            environment.pop("GOOGLE_API_KEY", None)
            environment.pop("SERPAPI_API_KEY", None)
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None,
                    [temp_dir, project_root, environment.get("PYTHONPATH")],
                )
            )

            try:
                completed = subprocess.run(
                    [sys.executable, "-c", 'import app; print("IMPORT_COMPLETED")'],
                    cwd=project_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                self.fail(f"import app timed out after 8 seconds: {error}")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("IMPORT_COMPLETED", completed.stdout)


if __name__ == "__main__":
    unittest.main()
