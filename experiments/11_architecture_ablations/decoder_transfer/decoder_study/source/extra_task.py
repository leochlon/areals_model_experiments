import json
import sys

import torch

from setup_confirmation import ROOT,OLD,run_check,write
sys.path.insert(0,str(OLD/'source'))
import online_backbone as online
import field_benchmark as fb
from transfer_architecture import selection


class MatchedHead(online.Field):
    def __init__(self,base):
        super().__init__(base)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(57300+online.SEED)
            self.head.reset_parameters()
        path=online.OUT/'initialization.json'
        info=json.loads(path.read_text())
        info.update(rca_scalar_head_seed=57300+online.SEED)
        path.write_text(json.dumps(info,indent=2)+'\n')


def main(index,log):
    """Fresh-decoder model `index` of decoder_controls_tasks.json: field fit, test predictions, hidden-value check."""
    task=json.loads((ROOT/'decoder_controls_tasks.json').read_text())[index]
    done=ROOT/'results/extra_tasks'/f"{task['task']:02d}.json";assert not done.exists()
    bench=ROOT/'bench';out=bench/'models'/task['name']
    assert not (out/'field_best.pt').exists()
    online.ARM,online.SEED,online.OUT=task['backbone'],task['seed'],out
    fb.FieldEncoder,fb.load_field_encoder=MatchedHead,online.load
    fb.Completion,fb.Areal,fb.selection_problems=online.Completion,online.Areal,selection
    native_adam=torch.optim.AdamW;torch.optim.AdamW=online.EncoderAdamW
    try:
        fb.main(['train-areal','--bench',str(bench),'--checkpoint',task['checkpoint_source'],'--out',str(out),
            '--epochs','40','--batch-size','8','--lr','1e-3','--new-lr','1e-3','--queries','2048',
            '--normalise','dataset','--mix-unobserved','1','--mix-sensors','0','--seed',str(task['seed']),'--workers','4'])
    finally:torch.optim.AdamW=native_adam
    path=out/'field_best.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert state['benchmark']['seed']==task['seed'] and state['benchmark']['train_cars']==380
    common=['--bench',str(bench),'--split','test','--final','--method','areal',
        '--name',task['name'],'--model',str(path),'--device','cuda']
    fb.main(['predict',*common]);run_check(fb,['check',*common,'--cars','4','--per-car','4'],log)
    gradient=json.loads((out/'encoder_gradient.json').read_text())
    assert gradient['finite'] and gradient['gradient_norm']>0
    write(done,dict(passed=True,**task,checkpoint=str(path),true_downstream_seed=state['benchmark']['seed'],
        encoder_gradient=gradient))
    print('EXTRA_TASK_COMPLETE',task['name'],flush=True)
    return done
