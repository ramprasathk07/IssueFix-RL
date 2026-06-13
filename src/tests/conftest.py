import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"


def pytest_addoption(parser):
    parser.addoption(
        "--ckpt",
        action="store",
        default=None,
        help="Checkpoint dir to verify (default: latest under outputs/)",
    )


def find_latest_checkpoint() -> Path | None:
    checkpoints = sorted(OUTPUTS_DIR.rglob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


@pytest.fixture(scope="session")
def checkpoint_dir(request) -> Path:
    explicit = request.config.getoption("--ckpt")
    ckpt = Path(explicit) if explicit else find_latest_checkpoint()

    if ckpt is None:
        pytest.skip("No checkpoint found under outputs/ — run training first.")
    if not ckpt.is_dir():
        pytest.fail(f"Not a directory: {ckpt}")

    print(f"\nVerifying checkpoint: {ckpt}")
    return ckpt
