import subprocess
import sys


def test_module_help_executes():
    result = subprocess.run(
        [sys.executable, "-m", "swarm_forge", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "Swarm Forge v1.0" in result.stdout or "usage:" in result.stdout.lower()