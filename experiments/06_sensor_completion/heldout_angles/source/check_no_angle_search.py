import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
sys.path.insert(0, '/workspace/runs/areals_multicondition_wings_20261023')
import study


def main():
    torch.set_num_threads(2)
    study.ROOT = ROOT / 'unseen_conditions'
    text_read, array_read = Path.read_text, np.load
    counts = dict(encoder_calls=0, condition_input_calls=0, forbidden_file_reads=0)
    def forbidden_name(value):
        return isinstance(value, (str, Path)) and any(s in str(value) for s in ['conditions_reference_only', 'candidate_spec', '/candidates/', '/angle_search/'])
    def read_text(path, *args, **kwargs):
        if forbidden_name(path):
            counts['forbidden_file_reads'] += 1
            raise AssertionError('prediction accessed a condition/search artifact')
        return text_read(path, *args, **kwargs)
    def load(path, *args, **kwargs):
        if forbidden_name(path):
            counts['forbidden_file_reads'] += 1
            raise AssertionError('prediction accessed cached condition candidates')
        return array_read(path, *args, **kwargs)
    def condition_input(*args, **kwargs):
        counts['condition_input_calls'] += 1
        raise AssertionError('angle-blind prediction invoked the condition-input builder')
    with patch.object(Path, 'read_text', read_text), patch.object(np, 'load', load), patch.object(study, 'condition_inputs', condition_input):
        study.configure('measurements')
        assert study.REF is None
        fb = study.fb
        config = fb.bench_config(study.ROOT / 'bench')
        record = fb.records_of(config, 'val')[0]
        field = fb.load_field(config['fields'], record['id'])
        meta, arrays = fb.read_observations(study.ROOT / 'bench', 'val')
        entry = next(e for e in meta['entries'] if e['car'] == record['id'] and e['setting'] == 'sensors-20')
        index = arrays[entry['key']]
        observed = fb.Observed(entry['key'], dict(name='sensors-20', kind='sensors', count=20), index, field.values[index])
        checkpoint = study.ROOT / 'bench/models/measurements/field_best.pt'
        method = study.Areal.load('measurements', checkpoint, torch.device('cpu'), 1, 'none')
        native_encode = method.model.encode
        def encode(*args, **kwargs):
            counts['encoder_calls'] += 1
            return native_encode(*args, **kwargs)
        method.model.encode = encode
        prediction = method.predict_car(fb.surface_of(record, field), [observed])[0]
        assert counts == dict(encoder_calls=1, condition_input_calls=0, forbidden_file_reads=0)
        assert prediction.shape == field.values.shape and np.isfinite(prediction).all()
    result = dict(passed=True, **counts, observed_sensors=len(index), predicted_surface_points=len(prediction),
                  checkpoint=str(checkpoint))
    (ROOT / 'results/no_angle_search.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
