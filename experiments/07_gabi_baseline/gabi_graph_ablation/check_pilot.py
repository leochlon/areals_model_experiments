import json
import sys

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from common import ROOT, UPSTREAM, VARIANTS, write_json

sys.path.insert(0,str(ROOT/'source'))
import gabi_adapter as ga
sys.path.insert(0,str(UPSTREAM))
from models import GCNGeomAutoencoder
abc_inference = ga.load_module(UPSTREAM/'sampling/abc_inference.py','author_abc').abc_inference

def main():
    for variant in VARIANTS:
        complete=json.loads((ROOT/'pilot'/f'{variant}_seed0'/'complete.json').read_text())
        assert complete['epochs']==2 and complete['updates']==44
    config=json.loads((ROOT/'config.json').read_text())
    model=GCNGeomAutoencoder(**config['data_dims'],**config['hyperparameters'],**config['model_settings']).cuda().eval()
    field=next(iter(sorted((ROOT/'graphs').glob('*.npz'))))
    with np.load(field) as cache:
        points=cache['points'].copy()
        edges={k:cache[k].copy() for k in VARIANTS}
    state=torch.load(ROOT/'pilot/surface_seed0/best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    stats=state['trainset_stats']
    pm,ps=stats['pos']['mean'].double().numpy(),stats['pos']['std'].double().numpy()
    position=(points-pm)/ps
    original=ga.graph(points,None,8,pos=position)
    assert np.array_equal(original.edge_index.numpy(),edges['euclidean'])
    checks={}
    observed=torch.tensor([0,10,100,1000,10000],device='cuda')
    values=torch.tensor([.1,.2,-.3,.4,-.5],device='cuda')
    for variant in VARIANTS:
        data=Data(pos=torch.as_tensor(position,dtype=torch.float32),edge_index=torch.as_tensor(edges[variant]),
                  x=torch.zeros(len(points),1),boundary_encoding=torch.zeros(len(points),0))
        masked=data.clone().cuda()
        masked.x[:]=float('nan')
        masked.x[observed,0]=values
        masked.y=masked.x.clone()
        torch.manual_seed(91)
        exact=abc_inference(model,masked,n_total_samples=100,batch_size=10,n_accepted_samples=10,sigma_tc=.01)
        exact=exact['u_posterior_samples'].reshape(10,len(points)).double().mean(0)
        rng=torch.Generator(device='cuda').manual_seed(91)
        pooled=ga.abc_pool(model,data,[(observed,values)],rng,[rng],100,10,.01,10,'cuda')[0]
        difference=float((exact-pooled.cpu()).abs().max())
        assert difference < 1e-4,(variant,difference)
        independent=ga.abc_pool(model,data,[(observed,values)],torch.Generator(device='cuda').manual_seed(33),
            [torch.Generator(device='cuda').manual_seed(44)],100,10,.01,10,'cuda')[0]
        scrambled=data.clone()
        scrambled.x=torch.randn_like(scrambled.x)*1000
        scrambled.y=torch.randn_like(scrambled.x)*1000
        repeated=ga.abc_pool(model,scrambled,[(observed,values)],torch.Generator(device='cuda').manual_seed(33),
            [torch.Generator(device='cuda').manual_seed(44)],100,10,.01,10,'cuda')[0]
        leak_difference=float((independent-repeated).abs().max())
        assert leak_difference < 1e-4,(variant,leak_difference)
        checks[variant]=dict(author_abc_max_difference=difference,hidden_node_values_max_difference=leak_difference)
    write_json(ROOT/'pilot/complete.json',dict(status='PASS',full_batch_size=100,epochs_per_variant=2,
        real_geometry_abc_parity=checks,torch=torch.__version__,torch_geometric=torch_geometric.__version__,
        numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    print(json.dumps({'event':'pilot checks passed','checks':checks}),flush=True)

if __name__=='__main__':
    main()
