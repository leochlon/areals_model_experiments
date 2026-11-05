import json
from pathlib import Path

import numpy as np
import torch

import architecture as study
import transfer_architecture as transfer
from areal_tokens.config import ExperimentConfig
from areal_tokens.data import read_manifest

ARMS = ['reference', 'ordinary_attention', 'unanchored_latents', 'oriented_normal_target',
        'unoriented_vector_target', 'anchored_no_rotary', 'unanchored_no_rotary']


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    records, _ = read_manifest(study.ROOT / 'architecture_manifest.json')
    records = [r for r in records if r['split'] == 'test']
    assert len(records) == 128
    assert np.mean((np.asarray([1., 2.]) / 2) ** 2) == .625
    output = study.ROOT / 'results/geometry_audit'
    output.mkdir(exist_ok=True)
    results, table = [], []
    for arm in ARMS:
        for seed in [0, 1, 2]:
            folder = study.ROOT / 'results/architecture' / arm / f'seed{seed}'
            original = json.loads((folder / 'test.json').read_text())
            assert original['steps'] == 2048 and len(original['rows']) == 128
            state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            config = ExperimentConfig.from_dict(state['config'])
            transfer.ARM = arm
            model = transfer.base_model(config, seed)
            model.load_state_dict(state['model'], strict=True)
            model.cuda().eval()
            raw, independent = {}, {}
            def capture(module, inputs, prediction):
                batch = inputs[0]
                identity = batch['id'][0]
                norm_distance = prediction['normalized_udf'][0].float().cpu().numpy()
                normals = prediction['normals'][0].float().cpu().numpy()
                target = batch['udf'][0].float().cpu().numpy()
                scale = float(batch['scale'][0, 0])
                true_normals = batch['query_normals'][0].float().cpu().numpy()
                valid = batch['normal_valid'][0].cpu().numpy()
                mse = float(np.mean((norm_distance.astype(np.float64) - target.astype(np.float64) / scale) ** 2))
                cosine = np.einsum('ij,ij->i', normals.astype(np.float64), true_normals.astype(np.float64))
                angle = float(np.degrees(np.arccos(np.clip(np.abs(cosine[valid]), 0, 1))).mean()) if valid.any() else None
                independent[identity] = dict(normalized_mse=mse, normal_degrees=angle)
                for name, value in dict(normalized_prediction=norm_distance, normals=normals,
                     target=target, scale=np.asarray(scale), target_normals=true_normals, valid=valid).items():
                    raw[identity + '_' + name] = value
            handle = model.register_forward_hook(capture)
            report = study.evaluate(model, records, config, torch.device('cuda'))
            handle.remove()
            worst_mse, worst_angle = 0., 0.
            for row in report['rows']:
                audit = independent[row['id']]
                worst_mse = max(worst_mse, abs(audit['normalized_mse'] - row['normalized_mse']))
                if row['normal_degrees'] is not None:
                    worst_angle = max(worst_angle, abs(audit['normal_degrees'] - row['normal_degrees']))
            assert worst_mse < 1e-7 and worst_angle < .02
            original_delta = abs(original['normalized_mse'] - report['normalized_mse'])
            assert original_delta < 1e-5, ('saved-checkpoint repeat disagrees', arm, seed, original_delta)
            np.savez_compressed(output / f'{arm}_seed{seed}.npz', **raw)
            result = dict(arm=arm, seed=seed, raw_shapes=len(independent), independent_mse_maximum_difference=worst_mse,
                          independent_angle_maximum_difference_degrees=worst_angle, repeated_mean_mse_difference=original_delta)
            results.append(result)
            table.append(dict(arm=arm, seed=seed, **{k:v for k,v in report.items() if k!='rows'}))
            study.write(output / 'verification.json', dict(passed=True, models=len(results), results=results))
            study.write(output / 'table.json', table)
            print('GEOMETRY_AUDIT', json.dumps(result), flush=True)
            del model, state, raw
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
