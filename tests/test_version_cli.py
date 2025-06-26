import subprocess
import sys


def test_version_cli_outputs_version() -> None:
    result = subprocess.run([sys.executable, "-m", "src.version_cli"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip(), "version output should not be empty"
