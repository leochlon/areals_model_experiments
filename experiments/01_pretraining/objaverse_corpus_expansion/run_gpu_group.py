"""Supervise eight preparation workers per allocated GPU."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(args):
    children = []
    handles = []
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for device, first in enumerate(args.first_workers):
            for worker in range(first, first + 8):
                log = (ROOT / "logs" / f"worker_{worker:02d}.log").open("ab", buffering=0)
                handles.append(log)
                command = [sys.executable, "-u", str(ROOT / "gpu_prepare_worker.py"),
                           "--worker", str(worker), "--device", str(device)]
                children.append((worker, subprocess.Popen(command, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)))
        (ROOT / f"group_{args.first_workers[0]:02d}.json").write_text(json.dumps(
            dict(workers=[dict(worker=w, pid=p.pid) for w, p in children]), indent=2) + "\n")
        while children:
            for worker, child in list(children):
                code = child.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"Worker {worker} exited with code {code}")
                    children.remove((worker, child))
            if children:
                time.sleep(2)
    finally:
        for _, child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for _, child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-workers", type=int, nargs="+", required=True)
    main(parser.parse_args())
