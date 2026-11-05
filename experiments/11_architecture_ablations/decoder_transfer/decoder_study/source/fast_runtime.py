from dataclasses import replace

import numpy as np
import torch

from setup_confirmation import fb
from areal_tokens.model import SpatialDecoder

native_decoder_forward=SpatialDecoder.forward
native_areal_init=fb.Areal.__init__


def decoder_forward(self,*args,**kwargs):
    original=self.config
    self.config=replace(original,query_chunk=256 if self.training else 2048)
    try:return native_decoder_forward(self,*args,**kwargs)
    finally:self.config=original


def areal_init(self,name,model,experiment,meta,device,batch_size,amp,source=None):
    native_areal_init(self,name,model,experiment,meta,device,batch_size,'none',source)


def selection(config,records,limit):
    problems=[]
    for record in records[:16]:
        original=fb.load_field(config['fields'],record['id'])
        rng=np.random.default_rng(fb.stable_seed(240024,record['id']+':fixed-dev-8192',0))
        indices=rng.choice(len(original.values),8192,replace=True,p=original.weights/original.weights.sum())
        normals=None if original.normals is None else original.normals[indices]
        field=fb.Field(original.points[indices],np.full(8192,original.weights.sum()/8192),
            original.values[indices],normals)
        observed=np.sort(rng.choice(8192,5,replace=False))
        context=fb.Observed(record['id'],dict(name='fixed-development',kind='sensors',count=5),observed,field.values[observed])
        problems.append((fb.surface_of(record,field),field,[context]))
    return problems


def install():
    SpatialDecoder.forward=decoder_forward
    fb.Areal.__init__=areal_init
    fb.selection_problems=selection
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False


def enforce_selection(original):
    def main(argv=None):
        fb.selection_problems=selection
        return original(argv)
    return main
