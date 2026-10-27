"""One heavy run at a time on this machine.

Each process that loads the full dataset holds 7-11 GB. On 25 Oct three ran at once
(10.7 + 8.6 + 6.8 GB on a 16 GB Mac mini) and macOS killed ~20 system daemons for
lack of swap. harness.dataset() takes this lock, so a second harness, explain or
analysis script queues behind the first instead of loading a second copy.

The lock is an flock on data/cache/heavy_run.lock held for the life of the process;
the kernel drops it when the process exits or is killed, so it can never go stale.
IAA_RUN_LOCK_WAIT sets the longest wait in seconds (default 7200);
IAA_RUN_LOCK=off disables the guard.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from typing import IO

from iaa import data

LOCK = data.ROOT / "data" / "cache" / "heavy_run.lock"
_held: IO[str] | None = None


def _holder() -> str:
    try:
        return LOCK.read_text().strip() or "unknown holder"
    except OSError:
        return "unknown holder"


def acquire(wait_s: float | None = None) -> None:
    """Block until this process holds the heavy-run lock (no-op if it already does)."""
    global _held
    if _held is not None or os.environ.get("IAA_RUN_LOCK") == "off":
        return
    limit = float(os.environ.get("IAA_RUN_LOCK_WAIT", "7200")) if wait_s is None else wait_s
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK, "a+")  # noqa: SIM115 - held open for the life of the process
    t0, told = time.time(), False
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not told:
                print(
                    f"runguard: waiting for another heavy run ({_holder()})",
                    file=sys.stderr,
                    flush=True,
                )
                told = True
            if time.time() - t0 > limit:
                fh.close()
                sys.exit(f"runguard: gave up after {limit:.0f}s waiting for {_holder()}")
            time.sleep(5)
    fh.seek(0)
    fh.truncate()
    fh.write(
        f"pid {os.getpid()} since {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(sys.argv)[:200]}\n"
    )
    fh.flush()
    _held = fh


def wait_free() -> None:
    """Wait for any running heavy run to finish, without keeping the lock."""
    global _held
    acquire()
    if _held is not None:
        fcntl.flock(_held, fcntl.LOCK_UN)
        _held.close()
        _held = None
