"""One of the 24 field models of the decoder study with the runtime settings of fast_runtime.py.

Tasks 0-17 are the primary models of tasks.json, tasks 18-23 the fresh-decoder models of decoder_controls_tasks.json
(which start from the geometry checkpoints of tasks 0, 2 and 4). Run through bootstrap.py, which supplies the
Point-MAE dependencies.
"""
import argparse
import json

import confirmation_task
import extra_task
import fast_runtime
from setup_confirmation import ROOT,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True,choices=range(24))
    index=parser.parse_args().task
    fast_runtime.install()
    from setup_confirmation import fb
    fb.main=fast_runtime.enforce_selection(fb.main)
    log=ROOT/'logs'/f'task_{index:02d}.log'
    done=confirmation_task.main(index,log) if index<18 else extra_task.main(index-18,log)
    result=json.loads(done.read_text())
    result.update(fixed_development_queries=8192,numerical_query_chunk=2048)
    write(done,result)


if __name__=='__main__':main()
