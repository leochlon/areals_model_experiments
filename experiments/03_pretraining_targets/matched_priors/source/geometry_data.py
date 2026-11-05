"""Object-indexed readers for the consolidated, split-preserving corpus."""
from collections import OrderedDict
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import train_mesh_geometry as patch
import train_query_geometry as query


class Packs:
    def __init__(self,index,arm):
        self.path=Path(index).resolve()
        doc=json.loads(self.path.read_text())
        self.rows=[r for r in doc['objects'] if r[arm]['status']=='ok']
        self.arm=arm
        self.handles=OrderedDict()

    def group(self,i):
        row=self.rows[int(i)];path=row['pack']
        if path not in self.handles:
            self.handles[path]=h5py.File(path,'r')
            if len(self.handles)>8:self.handles.popitem(last=False)[1].close()
        self.handles.move_to_end(path)
        return self.handles[path][row[self.arm]['group']]

    def __getstate__(self):
        state=self.__dict__.copy();state['handles']=OrderedDict();return state


def object_order(rows,split,seed,length):
    eligible=np.array([i for i,r in enumerate(rows) if r['split']==split],dtype=np.int64)
    if not len(eligible):raise ValueError(f'No eligible {split} objects')
    parts=[]
    for epoch in range((length+len(eligible)-1)//len(eligible)):
        rng=np.random.default_rng([seed,epoch])
        parts.append(rng.permutation(eligible))
    return np.concatenate(parts)[:length]


class PatchObjects(torch.utils.data.Dataset):
    def __init__(self,index,split,seed,length,patches_per_object=8,queries=64):
        self.store=Packs(index,'patch');self.seed=seed;self.queries=queries
        self.patches_per_object=patches_per_object
        self.order=object_order(self.store.rows,split,seed,length)
        self.offset=0

    def __len__(self):return len(self.order)-self.offset

    def __getitem__(self,index):
        index=int(index)+self.offset;rng=np.random.default_rng([self.seed,index])
        group=self.store.group(self.order[index])
        mass=group['area_mass'][:]
        ids=rng.choice(len(mass),self.patches_per_object,replace=True,p=mass/mass.sum())
        arrays={k:np.stack([group[k][int(i)] for i in ids]) for k in ['inputs','coeff','radius','offset']}
        uv=[]
        for i in ids:
            start,end=group['triangle_offsets'][int(i):int(i)+2]
            tri=group['triangles'][int(start):int(end)].astype(np.float64)
            uv.append(patch.triangle_samples(tri,self.queries,rng,projected=True)[0][:,:2])
        tau=rng.uniform(.1,1,(len(ids),self.queries))*arrays['offset'][:,None]
        tau[:,::3]=0.;tau[:,1::3]*=-1
        return dict(**arrays,uv=np.stack(uv),tau=tau,object_index=np.int64(self.order[index]))


def collate_patches(rows):
    return {k:torch.as_tensor(np.concatenate([r[k] for r in rows],axis=0))
            for k in ['inputs','coeff','radius','uv','tau']}


class QueryCorpus:
    def __init__(self,index):
        self.store=Packs(index,'query')
        self.scales=[.02,.05,.15]
        self.shards=[dict(split=r['split'],id=r['id']) for r in self.store.rows]
        self.split={s:[i for i,r in enumerate(self.store.rows) if r['split']==s] for s in ['train','val','test']}
        self.cache=OrderedDict()

    def load(self,i):
        if i not in self.cache:
            self.cache[i]={k:v[:] for k,v in self.store.group(i).items()}
            if len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(i);return self.cache[i]


class QueryObjects(query.Examples):
    def __init__(self,index,split,seed,length,evaluation=False):
        corpus=QueryCorpus(index)
        cfg=dict(input_points=4096,queries=1024,tokens=256,mask_ratio=0.,mask_patches=32,
                 whole_fraction=1.,rotate=not evaluation,views=1)
        super().__init__(corpus,split,cfg,seed,length,evaluation)
        self.order=object_order(corpus.store.rows,split,seed,length)
        self.original_ids=self.ids
        self.offset=0

    def __len__(self):return self.length-self.offset

    def __getitem__(self,index):
        index=int(index)+self.offset
        # The supplied example constructor handles rotation and labels; pin its
        # one-object choice to the shared epoch-permutation order.
        self.ids=[int(self.order[index])]
        try:return super().__getitem__(index)
        finally:self.ids=self.original_ids
