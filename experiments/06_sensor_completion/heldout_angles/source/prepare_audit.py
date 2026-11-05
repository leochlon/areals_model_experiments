import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
ORIGINAL = Path('/workspace/runs/areals_multicondition_wings_20261023')
sys.path.insert(0, str(ORIGINAL))
import field_benchmark as fb
from areal_tokens.data import read_manifest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    source = Path('/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/manifest.json')
    raw = json.loads(source.read_text())
    records, _ = read_manifest(source)
    selected = []
    for split, count in [('train', 8192), ('val', 128), ('test', 128)]:
        pool = sorted((r for r in records if r['split'] == split),
                      key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        chosen = pool[:count]
        assert len(chosen) == count
        chosen = [{**r, 'original_path': r['path'],
                   'path': str(ROOT / 'restored_geometry' / (r['id'] + '.npz'))} for r in chosen]
        selected.extend(chosen)
    assert len({r['family'] for r in selected}) == len(selected)
    write(ROOT / 'architecture_manifest.json', {**raw, 'records': selected})
    production = Path('/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/config.json')
    (ROOT / 'production_config.json').write_bytes(production.read_bytes())
    # Keep the original family splits; training and development use odd angles, the test uses even angles.
    original = json.loads((ORIGINAL / 'manifest.json').read_text())
    reference = json.loads((ORIGINAL / 'conditions_reference_only.json').read_text())
    keep = []
    for r in original['records']:
        angle = int(reference['cases'][r['id']]['aoa'])
        if (r['split'] in ['train', 'val'] and angle % 2 == 1) or (r['split'] == 'test' and angle % 2 == 0):
            keep.append(r)
    train = [r for r in keep if r['split'] == 'train']
    assert len(train) == 1200
    assert len([r for r in keep if r['split'] == 'val']) == 80
    assert len([r for r in keep if r['split'] == 'test']) == 64
    angles = [reference['cases'][r['id']]['aoa'] for r in train]
    import numpy as np
    reference['mean'], reference['std'] = float(np.mean(angles)), float(np.std(angles))
    root = ROOT / 'unseen_conditions'
    write(root / 'manifest.json', {**original, 'records': keep})
    write(root / 'conditions_reference_only.json', reference)
    for name in ['fields', 'geometry']:
        if not (root / name).exists():
            (root / name).symlink_to(ORIGINAL / name, target_is_directory=True)
    bench = root / 'bench'
    cfg = json.loads((ORIGINAL / 'bench/config.json').read_text())
    cfg.update(manifest=str(root / 'manifest.json'), fields=str(root / 'fields'))
    write(bench / 'config.json', cfg)
    for split in ['val', 'test']:
        meta, arrays = fb.read_observations(ORIGINAL / 'bench', split)
        ids = {r['id'] for r in keep if r['split'] == split}
        meta['entries'] = [e for e in meta['entries'] if e['car'] in ids]
        meta['cars'] = {k: v for k, v in meta['cars'].items() if k in ids}
        arrays = {e['key']: arrays[e['key']] for e in meta['entries']}
        write(bench / 'observations' / (split + '.json'), meta)
        np.savez_compressed(bench / 'observations' / (split + '.npz'), **arrays)
    car_manifest = Path('/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z/manifest.json')
    cars = json.loads(car_manifest.read_text())['records']
    official = json.loads((ROOT / 'official-split-manifest.json').read_text())
    official_test = set(official['full_test'])
    overlap = {split: sorted(official_test & {r['id'] for r in cars if r['split'] == split}) for split in ['train', 'val']}
    raw_root = Path('/workspace/home/drivaerml_transfer_pilot_20260925T121525Z')
    inventory = dict(architecture_counts={split: sum(r['split'] == split for r in selected) for split in ['train', 'val', 'test']},
         restored_geometry_files_ready=sum(Path(r['path']).is_file() for r in selected),
         official_test_count=len(official_test),
         official_test_overlap_with_historical=overlap,
         prepared_official_test_ids=sorted(i for i in official_test if (raw_root / 'data' / (i + '.npz')).exists()),
         withheld_angle_counts={split: sum(r['split'] == split for r in keep) for split in ['train', 'val', 'test']},
         withheld_training_angles=sorted(set(angles)), condition_mean=reference['mean'], condition_std=reference['std'])
    write(ROOT / 'results/intake_inventory.json', inventory)
    print(json.dumps(inventory, indent=2), flush=True)


if __name__ == '__main__':
    main()
