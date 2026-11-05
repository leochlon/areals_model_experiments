"""Train fresh geometry weights for fifteen epochs, then score the selected model."""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def status(phase, **extra):
    value = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = ROOT / "status.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(ROOT / "status.json")
    print(json.dumps(value), flush=True)


def run(phase, args, log_name):
    status(phase, command=args)
    with open(ROOT / log_name, "ab", buffering=0) as log:
        child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
        if code:
            raise subprocess.CalledProcessError(code, args)


def main(resume=False):
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        assert not resume, "train_six.py starts from fresh weights; there is no resume"
        args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=6",
                str(ROOT / "train_six.py")]
        run("training_six_gpus", args, "training.log")
        run("evaluating_selected_checkpoint", [sys.executable, str(ROOT / "finish_benchmark.py")], "evaluation.log")
        status("complete", completed_epochs=15, training_geometries=61822, gpus=6,
               comparison=str(ROOT / "comparison.json"))
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args().resume)
