import json
import sys
from pathlib import Path

import numpy as np
import torch

ORIGINAL = Path('/workspace/home/areal_aerojepa_20260928')
ROOT = Path('/workspace/runs/areals_architecture_controls_20261024/frozen_replication')
sys.path.insert(0, str(ORIGINAL))
import run_field as author

native_score = author.score_pressure
counter = 0


class SmokeComplete(Exception):
    pass


@torch.inference_mode()
def score(model, loader, dataset, device, world_size):
    global counter
    assert world_size == 1
    reported = native_score(model, loader, dataset, device, world_size)
    arrays, independent = {}, []
    for batch in loader:
        batch = author.move_batch_to_device(batch, device)
        for index in range(batch['context_pos'].shape[0]):
            sample = author._slice_batch_sample(batch, index)
            with author.get_autocast_context(device, 'bf16'):
                prediction = model(context_pos=sample['context_pos'], context_feat=sample['context_feat'],
                                   gen_params=sample['gen_params'], query_pos=sample['query_pos'],
                                   query_sdf=sample['query_sdf'])
            p = (prediction.float() * dataset.std + dataset.mean).cpu().numpy().astype(np.float64).reshape(-1)
            y = (sample['query_target'].float() * dataset.std + dataset.mean).cpu().numpy().astype(np.float64).reshape(-1)
            independent.append(float(np.sqrt(np.sum((p-y)**2) / max(np.sum(y**2), 1e-24))))
            arrays[str(sample['case_id']) + '_prediction'] = p
            arrays[str(sample['case_id']) + '_truth'] = y
    difference = abs(float(np.mean(independent)) - reported['rl2'])
    assert difference < 2e-5, ('independent fixed-query rL2 mismatch', difference)
    counter += 1
    name = sys.argv[sys.argv.index('--run-name') + 1]
    out = ROOT / 'runs' / name / 'raw_validation'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f'call{counter:03d}.npz', **arrays)
    (out / f'call{counter:03d}.json').write_text(json.dumps(dict(independently_recomputed_rl2=float(np.mean(independent)),
          author_rl2=reported['rl2'], absolute_difference=difference, cases=len(independent)), indent=2)+'\n')
    return reported


if __name__ == '__main__':
    smoke = '--smoke' in sys.argv
    if smoke:
        sys.argv.remove('--smoke')
        assert 'smoke' in sys.argv[sys.argv.index('--run-name') + 1]
        native_epoch = author._run_epoch
        calls = [0]
        def epoch(*args, **kwargs):
            if calls[0] >= 2:
                raise SmokeComplete()
            calls[0] += 1
            kwargs['max_batches'] = 1
            return native_epoch(*args, **kwargs)
        author._run_epoch = epoch
    if '--normals' in sys.argv:
        sys.argv.remove('--normals')
        assert sys.argv[sys.argv.index('--arm') + 1] == 'native'
        seed = int(sys.argv[sys.argv.index('--seed') + 1])
        NativeDataset = author.DrivAerMLDataset
        class NormalsDataset(NativeDataset):
            def __getitem__(self, index):
                item = super().__getitem__(index)
                features = item['context_feat']
                # Native AeroJEPA reads normals from its trailing optional channels.
                item['context_feat'] = torch.cat([features, features[..., :3]], -1)
                return item
        author.DrivAerMLDataset = NormalsDataset
        native_load = author.OmegaConf.load
        def configuration(path):
            cfg = native_load(path)
            if Path(path) == author.EXAMPLE / 'conf/model/aerojepa.yaml':
                cfg.trunk.context_encoder.point_input_dim = 6
                cfg.trunk.context_encoder.use_solid_normals = True
            return cfg
        author.OmegaConf.load = configuration
        native_instantiate = author.hydra.utils.instantiate
        def instantiate(cfg, *args, **kwargs):
            plain = author.OmegaConf.create(author.OmegaConf.to_container(cfg, resolve=True))
            plain.trunk.context_encoder.point_input_dim = 3
            plain.trunk.context_encoder.use_solid_normals = False
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                reference = native_instantiate(plain, *args, **kwargs)
                post_reference_rng = torch.get_rng_state().clone()
            model = native_instantiate(cfg, *args, **kwargs)
            for name in ['target_encoder', 'predictor', 'decoder']:
                getattr(model, name).load_state_dict(getattr(reference, name).state_dict(), strict=True)
            torch.set_rng_state(post_reference_rng)
            return model
        author.hydra.utils.instantiate = instantiate
    ROOT.mkdir(exist_ok=True)
    for name in ['data', 'data_ready.json']:
        try:
            (ROOT / name).symlink_to(ORIGINAL / name)
        except FileExistsError:
            assert (ROOT / name).resolve() == (ORIGINAL / name).resolve()
    author.ROOT = ROOT
    author.score_pressure = score
    try:
        author.main()
    except SmokeComplete:
        assert smoke
        out = ROOT / 'runs' / sys.argv[sys.argv.index('--run-name') + 1]
        (out / 'software_smoke.json').write_text(json.dumps(dict(passed=True, epoch_function_calls=calls[0],
                  maximum_batches_per_call=1), indent=2)+'\n')  # one training and one validation batch
        print('smoke test passed', flush=True)
