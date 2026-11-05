import json
import sys

import numpy as np
import torch
import yaml

from common import ROOT, ORIGINAL, DATA, UPSTREAM, SEEDS, records, write_json
sys.path.insert(0, str(UPSTREAM))
from datatools import HelmholtzCarDataset

def main():
    graph_manifest=json.loads((ROOT/'graphs/complete.json').read_text())
    assert graph_manifest['status']=='PASS' and graph_manifest['geometries']==272
    graphs={}
    for row in graph_manifest['rows']:
        grid=row['grid_id']
        path=ROOT/'graphs'/(grid+'.npz')
        with np.load(path) as cache:
            graphs[grid]={k:cache[k].copy() for k in ['points','euclidean','surface']}
    cases=0
    for split in ['train','val']:
        dataset=HelmholtzCarDataset(str(ORIGINAL/'bench/gabi'/(split+'.pkl')))
        rows=records(split)
        assert len(dataset)==len(rows)
        for i,row in enumerate(rows):
            path=DATA/'fields'/(row['id']+'.npz')
            data=dataset.get(i)
            with np.load(path) as field:
                assert np.array_equal(data.pos.numpy(),field['points'])
                assert np.array_equal(data.x.numpy().reshape(-1),field['values'])
                assert np.array_equal(data.pos.numpy(),graphs[row['grid_id']]['points'])
                assert np.array_equal(data.edge_index.numpy(),graphs[row['grid_id']]['euclidean'])
            cases+=1
        print(json.dumps({'checked_split':split,'cases':len(rows)}),flush=True)
    # Test geometry is checked without reading test pressures.
    for row in records('test'):
        with np.load(DATA/'fields'/(row['id']+'.npz')) as field:
            assert np.array_equal(field['points'],graphs[row['grid_id']]['points'])
        cases+=1
    original_checkpoint=ORIGINAL/'bench/gabi/runs/GCN_dim_100_resume_20261023T195959Z/trained_models/best.pt'
    state=torch.load(original_checkpoint,map_location='cpu',weights_only=False)
    torch.save(state['trainset_stats'],ROOT/'normalization.pt')
    config=yaml.safe_load((ORIGINAL/'bench/gabi/runs/GCN_dim_100_resume_20261023T195959Z/config.yml').read_text())
    config['io_settings'].update(pretrained_model='',run_dir=str(ROOT/'runs'))
    write_json(ROOT/'config.json',config)
    # Training-set statistics of the single-seed run are reused; none of its trained weights are.
    write_json(ROOT/'protocol.json',dict(
        objective='Euclidean versus mesh-edge shortest-path 8-neighbour graphs in GABI on DoubleDelta.',
        variants=['euclidean','surface'],seeds=list(SEEDS),
        fixed=['32,000 field points, values, weights, normals and node ordering','train/validation/test split',
               'six-layer width-100 latent-100 author GCN, loss and normalization',
               'paired initial weights, per-epoch example order and MMD random draws',
               'batch size 100, 455 epochs, 10,010 updates, AdamW 1e-3, weight decay 1e-4, exponential decay to 1e-5',
               'checkpoint with minimum validation reconstruction plus MMD loss',
               'ABC 50,000 normal latent draws, 100 accepted, noise sigma .01, identical saved observation sets'],
        graph_definition=graph_manifest['definition'],
        original_gabi_commit='c67f98b1d0e1478f4f6691adbbeea56df6350206',
        original_mesh_revision=graph_manifest['source_revision'],
        normalization_source_checkpoint=str(original_checkpoint),
        primary_metric='mean hidden-area-weighted nRMSE at 20 saved sensors; paired surface minus euclidean difference',
        secondary_metrics=['relative integrated pressure-force error','all other saved observation settings'],
        cases=cases,geometries=len(graphs)))
    print(json.dumps({'status':'prepared','cases':cases,'geometries':len(graphs)}),flush=True)

if __name__=='__main__':
    main()
