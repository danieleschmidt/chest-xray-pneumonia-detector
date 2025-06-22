import subprocess
import sys
import pytest

pytest.importorskip("tensorflow")

def test_evaluate_cli_help():
    result = subprocess.run([sys.executable, "-m", "src.evaluate", "--help"], capture_output=True)
    assert result.returncode == 0
    assert b"--normalize_cm" in result.stdout
    assert b"--threshold" in result.stdout
    assert b"--metrics_csv" in result.stdout
    assert b"--num_classes" in result.stdout
