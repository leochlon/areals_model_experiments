"""Forward and backward pass of every arm on two training objects; the arms start from the same common tensors."""
import json
import time

import torch
from architecture import ROOT, ARMS, build, write
from areal_tokens.data import GeometryDataset, collate_clouds, read_manifest, to_device
from areal_tokens.losses import reconstruction_loss


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    records, _ = read_manifest(ROOT / 'architecture_manifest.json')
    records = [r for r in records if r['split'] == 'train'][:2]
    rows, first = [], None
    for arm in ARMS:
        model, config, shared = build(arm, 0)
        common = {name: model.state_dict()[name].clone() for name in shared}
        if first is None:
            first = common
        assert common.keys() == first.keys() and all(torch.equal(common[k], first[k]) for k in common), arm
        model.cuda().train()
        dataset = GeometryDataset(records, config.model, config.data, 0, training=True)
        batch = to_device(collate_clouds([dataset[i] for i in range(2)]), torch.device('cuda'))
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(batch)
            loss = reconstruction_loss(prediction, batch, config.train)['loss']
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        grad = float(model.encoder.embed[0].weight.grad.norm())
        assert grad > 0
        if arm == 'unanchored_latents':
            assert float(model.encoder.query_latents.grad.norm()) > 0
            assert not model.encoder(batch).positions.any()
        rows.append(dict(arm=arm, loss=float(loss.detach()), input_gradient_norm=grad,
                         common_initial_tensors=len(common), seconds=time.perf_counter() - started))
        del model
        torch.cuda.empty_cache()
    write(ROOT / 'results/arm_check.json', rows)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
