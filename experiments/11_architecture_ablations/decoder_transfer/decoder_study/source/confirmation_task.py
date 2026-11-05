import json
import subprocess
import sys

from setup_confirmation import ROOT,OLD,run_check,write


def main(index,log):
    """Primary model `index` of tasks.json: I/3-head geometry pretraining if the task needs it and its checkpoint does
    not exist yet, then the field fit, the test predictions and the hidden-value check."""
    task=json.loads((ROOT/'tasks.json').read_text())[index]
    done=ROOT/'results/tasks'/f'{index:02d}.json'
    assert not done.exists(),'Refusing duplicate completed task'
    if task['retrain_geometry']:
        folder=ROOT/'results/architecture'/task['arm']/f"seed{task['seed']}"
        if (folder/'test.json').exists():
            result=json.loads((folder/'test.json').read_text())
            assert result['steps']==2048 and result['seed']==task['seed'] and result['arm']==task['arm']
        else:
            subprocess.run([sys.executable,str(ROOT/'source/repair.py'),'--arm',task['arm'],'--seed',str(task['seed'])],check=True)
    sys.path.insert(0,str(OLD/'source'))
    import torch
    import transfer_architecture as transfer
    import field_benchmark as fb
    transfer.ARM,transfer.SEED=task['arm'],task['seed']
    fb.load_model,fb.load_field_encoder,fb.FieldEncoder=transfer.geometry_load,transfer.field_load,transfer.Field
    fb.Completion,fb.Areal,fb.selection_problems=transfer.Completion,transfer.Areal,transfer.selection
    bench=ROOT/'bench';out=bench/'models'/task['name']
    assert not (out/'field_best.pt').exists(),'Refusing accidental training restart'
    fb.main(['train-areal','--bench',str(bench),'--checkpoint',task['pretrained'],'--out',str(out),
        '--epochs','40','--batch-size','8','--lr','1e-3','--new-lr','1e-3','--queries','2048',
        '--normalise','dataset','--mix-unobserved','1','--mix-sensors','0','--seed',str(task['seed']),'--workers','4'])
    path=out/'field_best.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['benchmark']['seed']==task['seed'] and state['benchmark']['train_cars']==380
    common=['--bench',str(bench),'--split','test','--final','--method','areal',
        '--name',task['name'],'--model',str(path),'--device','cuda']
    fb.main(['predict',*common]);run_check(fb,['check',*common,'--cars','4','--per-car','4'],log)
    write(done,dict(passed=True,**task,checkpoint=str(path),true_downstream_seed=state['benchmark']['seed'],
        train_cars=380,selection_epoch=state['epoch']))
    print('CONFIRMATION_TASK_COMPLETE',index,task['name'],flush=True)
    return done
