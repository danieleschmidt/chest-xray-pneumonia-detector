import subprocess
import sys
import pytest

pytest.importorskip("tensorflow")


def test_inference_cli_help():
    result = subprocess.run([sys.executable, "-m", "src.inference", "--help"], capture_output=True)
    assert result.returncode == 0
    assert b"--num_classes" in result.stdout
