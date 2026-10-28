"""Optional cross-process lock so two jobs never load models into unified memory at once.

Set ``COMPUTE_LOCK`` to a directory path to enable it; ``mkdir`` is atomic, so the
directory doubles as the lock. Without the variable this is a no-op.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def compute_turn(owner: str = "lora-vs-prompt", poll: float = 15.0):
    lock = os.environ.get("COMPUTE_LOCK")
    if not lock:
        yield
        return
    path = Path(lock)
    waited = False
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            if not waited:
                holder = (path / "owner").read_text().strip() if (path / "owner").exists() else "?"
                print(f"waiting for compute lock held by: {holder}", file=sys.stderr)
                waited = True
            time.sleep(poll)
    try:
        owner = os.environ.get("COMPUTE_LOCK_OWNER", owner)
        (path / "owner").write_text(f"{owner} {time.strftime('%H:%M')}\n")
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)
