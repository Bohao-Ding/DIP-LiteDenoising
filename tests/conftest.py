import os
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def make_movie():
    def _make_movie(frames=13, height=8, width=7):
        data = np.arange(frames * height * width, dtype=np.float32)
        return data.reshape(frames, height, width)

    return _make_movie


def make_ephys_dat(path, frames=16):
    values = np.arange(frames * 192 * 2, dtype=np.int16)
    values.tofile(os.fspath(path))
    return path
