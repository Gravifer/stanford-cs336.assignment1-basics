import subprocess
import sys
from pathlib import Path


def test_runtime_contracts_run_with_import_hook() -> None:
    repository = Path(__file__).resolve().parents[3]
    runner = repository / "scripts" / "probes" / "run_runtime_type_tests.py"

    completed = subprocess.run(
        [sys.executable, str(runner)],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
