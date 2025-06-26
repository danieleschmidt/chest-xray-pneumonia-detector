import shutil
import subprocess
import pytest


def test_ruff_pass():
    """Ensure ruff passes without emitting any errors."""
    result = subprocess.run(["ruff", "check", "."], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_bandit_pass():
    """Verify bandit reports no medium or high severity issues."""
    if shutil.which("bandit") is None:
        pytest.skip("bandit not installed")
    result = subprocess.run([
        "bandit",
        "-r",
        "src",
        "-ll",
    ], capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Medium: 0" in output
    assert "High: 0" in output
