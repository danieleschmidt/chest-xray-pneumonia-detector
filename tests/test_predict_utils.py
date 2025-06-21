import sys
import subprocess
import pytest

pytest.importorskip("tensorflow")


def test_cli_help():
    result = subprocess.run([sys.executable, "-m", "src.predict_utils", "--help"], capture_output=True)
    assert result.returncode == 0

