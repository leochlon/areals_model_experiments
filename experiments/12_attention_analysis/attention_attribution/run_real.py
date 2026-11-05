import argparse
from pathlib import Path
import native_hooks as hooks
from export_attention import export

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', required=True)
ap.add_argument('--fields', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--ckpt', action='append', required=True)
args = ap.parse_args()
assert not list(Path(args.out).glob('attention_decoder/*/*.npz')), 'Use a fresh output directory.'
hooks.configure(args.manifest, args.fields, args.out)
export(args.out, [tuple(c.split('=', 1)) for c in args.ckpt], 'none', 2048, {'finetuned', 'scratch'}, 0,
       {k: getattr(hooks, k) for k in ('load_model', 'iterate_cases', 'encode', 'decoder_attention',
                                      'encoder_attention', 'predict')})
hooks.AUDIT['checks']['all_144_cases_exported'] = len(hooks.AUDIT['cases']) == 144
assert hooks.AUDIT['checks']['all_144_cases_exported']
hooks.save_audit()
