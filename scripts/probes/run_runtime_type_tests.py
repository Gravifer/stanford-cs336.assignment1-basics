"""Run the runtime typing contracts with the import hook installed.

This must execute in a fresh process so ``cs336_basics.nn`` is first imported
while the jaxtyping hook is active.
"""

import os
from pathlib import Path

from jaxtyping import install_import_hook


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    contract_file = repository / "tests" / "student" / "typing_tests" / "test_runtime_checking.py"
    os.environ["CS336_REQUIRE_RUNTIME_IMPORT_HOOK"] = "1"

    with install_import_hook("cs336_basics.nn", "beartype.beartype"):
        import pytest

        return pytest.main([str(contract_file), "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
