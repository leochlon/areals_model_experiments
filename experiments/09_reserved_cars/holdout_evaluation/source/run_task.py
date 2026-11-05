import argparse
import json
import subprocess
import sys

from registry import ROOT,AUDIT,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    task=parser.parse_args().task
    model=json.loads((ROOT/'models.json').read_text())[task]
    done=ROOT/'results/tasks'/f'{task:02d}.json'
    assert not done.exists(),f'task {task} was already evaluated'
    if model['family']=='frozen':
        subprocess.run([sys.executable,str(ROOT/'source/frozen_eval.py'),str(task)],check=True)
    else:
        bench=ROOT/'benches'/('completion' if model['family']=='replication' else 'geometry')
        prefix=[]
        if model['family']=='online':
            script=AUDIT/'source/online_backbone.py';prefix=['--backbone',model['arm'],'--audit-seed',str(model['seed'])]
        elif model['family']=='architecture':
            script=AUDIT/'source/transfer_architecture.py';prefix=['--arm',model['arm'],'--seed',str(model['seed'])]
        else:script='/workspace/runs/field_benchmark_4gpu_20261023/field_benchmark.py'
        common=['--bench',str(bench),'--split','test','--final','--method','areal',
                '--name',model['name'],'--model',model['checkpoint'],'--device','cuda']
        log=ROOT/'logs'/f'evaluate_{task:02d}.log';log.parent.mkdir(exist_ok=True)
        with log.open('w') as stream:  # score.py and audit_complete.py count the PASS lines of the check
            for command in [[*prefix,'predict',*common],[*prefix,'check',*common,'--cars','4','--per-car','4']]:
                subprocess.run([sys.executable,str(script),*command],check=True,stdout=stream,stderr=subprocess.STDOUT)
    write(done,dict(passed=True,**model))
    print('task complete',task,model['name'],flush=True)


if __name__=='__main__':main()
