"""Run the third preparation GPU and coordinate the final audited manifest."""

import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    children = []
    handles = []
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name, command in (
            ("coordinator.log", [sys.executable, "-u", str(ROOT / "finish_preparation.py")]),
            ("group_16.log", [sys.executable, "-u", str(ROOT / "run_gpu_group.py"), "--first-workers", "16"]),
        ):
            log = (ROOT / "logs" / name).open("ab", buffering=0)
            handles.append(log)
            children.append(subprocess.Popen(command, cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True))
        for child in reversed(children):
            code = child.wait()
            if code:
                raise RuntimeError(f"Preparation subprocess exited with code {code}")
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in handles:
            log.close()


if __name__ == "__main__":
    main()
