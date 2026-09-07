import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "raw"
ROOT = Path(__file__).resolve().parents[3]


def test_ingest_cli_runs(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ingest.py"), "--profile", "default", "--root", str(tmp_path),
         "--raw-dir", str(FIXTURES), "--corpus-version", "cli"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "documents=5" in result.stdout
    assert (tmp_path / "data/processed/cli/manifest.json").exists()
