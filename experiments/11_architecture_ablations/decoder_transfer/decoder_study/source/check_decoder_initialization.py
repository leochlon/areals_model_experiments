import sys

import torch

from setup_confirmation import ROOT,OLD,write
sys.path.insert(0,str(OLD/'source'))
import transfer_architecture as transfer
import online_backbone as online
from extra_task import MatchedHead
from areal_tokens.io import load_model


def same(a,b):
    a,b=a.state_dict(),b.state_dict()
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)


def main():
    torch.set_num_threads(2)
    source=ROOT/'results/architecture/reference/seed0/best.pt'
    transfer.ARM,transfer.SEED='reference',0
    torch.manual_seed(0);base,_,_=transfer.geometry_load(source,'cpu');full=transfer.Field(base)
    online.ARM,online.SEED,online.OUT='areals',0,ROOT/'results/decoder_initialization_probe'
    torch.manual_seed(0);base,_,_=load_model(source,'cpu');fresh=MatchedHead(base)
    checked=[]
    for name,value in full.state_dict().items():
        if not name.startswith('base.decoder.'):
            torch.testing.assert_close(value,fresh.state_dict()[name],rtol=0,atol=0)
            checked.append(name)
    assert not same(full.base.decoder,fresh.base.decoder)
    assert same(full.head,fresh.head)
    result=dict(passed=True,shared_non_decoder_tensors=len(checked),
        all_non_decoder_tensors_exactly_identical=True,scalar_head_identical=True,decoders_differ=True)
    write(ROOT/'results/decoder_initialization_check.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
