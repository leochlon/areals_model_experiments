#!/usr/bin/env python3
"""Export attention profiles and per-point predictions for `theory_tests.py attention` and `attribution`.

Section 2 predicts that pretrained attention is short-range and that fine-tuning adds long-range attention, and asks
whether pretraining's gains sit in curved, attached regions or where the field is set globally (vortex footprints,
wakes). This script sends the same inputs (whole surface, one sensor setting) through every checkpoint, records the
distance profile of decoder cross-attention (query to token) and, if available, encoder self-attention (token to
token), and stores per-point predictions of the fine-tuned and from-scratch models.

  python3 export_attention.py --dummy --out demo_attn
  python3 theory_tests.py attention --dir demo_attn/attention_decoder --scratch scratch
  python3 theory_tests.py attention --dir demo_attn/attention_encoder --scratch scratch
  python3 theory_tests.py attribution --dir demo_attn/attribution --local-region 0 --global-region 1

Real use, after filling in the hooks:

  python3 export_attention.py --out ATTN --setting none --queries 2048 \
      --ckpt pretrained=/path/geometry.ckpt --ckpt finetuned=/path/field.ckpt --ckpt scratch=/path/scratch.ckpt

Checkpoints:
  pretrained  the geometry checkpoint loaded into the field architecture with the new field columns at zero, which is
              exactly the field model at the start of fine-tuning
  finetuned   that model after fine-tuning (AREALS)
  scratch     the same architecture trained from random initialisation on the same data and updates

Output: DIR/attention_decoder/<ckpt>/<case>.npz, DIR/attention_encoder/<ckpt>/<case>.npz (when the encoder hook is
implemented) and DIR/attribution/<case>.npz. Attention files hold distance profiles (add --raw to also keep the
weights); attribution files hold X, N, A, f, the scored mask, pred_<ckpt> for the predicting checkpoints, R and group.
"""
import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theory_tests import _ellipsoid, attention_summaries, rms_radius, shape_operator_norm  # noqa: E402

# ======================================================================================================================
# HOOKS: replace these with calls into your codebase


def load_model(path, device='cuda'):
    from native_hooks import load_model as hook
    return hook(path, device)


def iterate_cases(setting):
    from native_hooks import iterate_cases as hook
    return hook(setting)


def encode(model, X, N, A, observed, values):
    from native_hooks import encode as hook
    return hook(model, X, N, A, observed, values)


def decoder_attention(model, tokens, Q):
    from native_hooks import decoder_attention as hook
    return hook(model, tokens, Q)


def encoder_attention(model, tokens):
    from native_hooks import encoder_attention as hook
    return hook(model, tokens)


def predict(model, tokens, Q):
    from native_hooks import predict as hook
    return hook(model, tokens, Q)


def softmax_attention(q, k, log_w, scale=None):
    """Recompute attention weights from queries q [..., M, d], keys k [..., T, d] (after rotary embedding) and the
    log-area bias log_w [T]. Use this when your attention calls a fused kernel that does not return weights:
    register forward hooks on each attention module that store its q and k, run one forward pass, and apply this to
    every stored pair. Works with numpy arrays or torch tensors."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    logits = (q @ k.swapaxes(-1, -2) if hasattr(q, 'swapaxes') else q @ k.transpose(-1, -2)) * scale + log_w
    logits = logits - logits.max(-1, keepdims=True) if isinstance(logits, np.ndarray) else \
        logits - logits.amax(-1, keepdim=True)
    e = np.exp(logits) if isinstance(logits, np.ndarray) else logits.exp()
    return e / e.sum(-1, keepdims=True) if isinstance(e, np.ndarray) else e / e.sum(-1, keepdim=True)


def to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


# ======================================================================================================================
# Pipeline

def stable_rng(key, seed):
    return np.random.default_rng(int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:12], 16))


def _save_profile(path, Q, anchors, tw, att, R, group, raw, query_weights=None):
    s = attention_summaries(Q, anchors, tw, att, R)
    extra = dict(Q=Q.astype(np.float32), anchors=np.asarray(anchors, np.float32), tw=np.asarray(tw, np.float32),
                 A=np.asarray(att, np.float16)) if raw else {}
    qw = np.ones(len(Q)) if query_weights is None else np.asarray(query_weights, float)
    np.savez_compressed(path, R=R, group=group, qw=qw,
                        query_sha256=hashlib.sha256(np.asarray(Q, '<f8').tobytes()).hexdigest(),
                        anchor_sha256=hashlib.sha256(np.asarray(anchors, '<f8').tobytes()).hexdigest(),
                        **s, **extra)


def export(out, ckpts, setting, n_queries, predict_names, seed, hooks, raw=False):
    models = {name: hooks['load_model'](path) for name, path in ckpts}
    for name in models:
        os.makedirs(os.path.join(out, 'attention_decoder', name), exist_ok=True)
    os.makedirs(os.path.join(out, 'attribution'), exist_ok=True)
    n_cases, has_encoder = 0, False
    for cid, X, N, A, f, group, observed in hooks['iterate_cases'](setting):
        X, N, A, f = (np.asarray(v, float) for v in (X, N, A, f))
        observed = np.asarray(observed, np.int64)
        R = rms_radius(X, A)
        rng = stable_rng(cid, seed)
        qi = np.sort(rng.choice(len(X), size=min(n_queries, len(X)), replace=False, p=A / A.sum()))
        Q = X[qi]                          # area-weighted queries, so equal query weights integrate over the surface
        preds = {}
        for name, model in models.items():
            tokens, anchors, tw = hooks['encode'](model, X, N, A, observed, f[observed])
            anchors, tw = to_numpy(anchors).astype(float), to_numpy(tw).astype(float)
            att = to_numpy(hooks['decoder_attention'](model, tokens, Q))
            _save_profile(os.path.join(out, 'attention_decoder', name, f'{cid}.npz'), Q, anchors, tw, att, R,
                          str(group), raw)
            enc = hooks['encoder_attention'](model, tokens)
            if enc is not None:
                has_encoder = True
                os.makedirs(os.path.join(out, 'attention_encoder', name), exist_ok=True)
                _save_profile(os.path.join(out, 'attention_encoder', name, f'{cid}.npz'), anchors, anchors, tw,
                              to_numpy(enc), R, str(group), raw, query_weights=tw)
            if name in predict_names:
                preds[f'pred_{name}'] = to_numpy(hooks['predict'](model, tokens, X)).astype(float)
        scored = np.ones(len(X), bool)
        scored[observed] = False
        np.savez_compressed(os.path.join(out, 'attribution', f'{cid}.npz'), X=X, N=N, A=A, f=f, scored=scored, R=R,
                 group=str(group), **preds)
        n_cases += 1
        print(f"exported case {n_cases}: {cid} ({group})", flush=True)
    print(f'exported {n_cases} cases for {", ".join(models)}'
          f'{"" if has_encoder else " (decoder only: encoder_attention returned None)"}')


# ======================================================================================================================
# Dummy stand-in: ellipsoids with a suction peak over the flat face. The "pretrained" model attends locally,
# fine-tuning mixes in long-range attention, the scratch model attends diffusely; the fine-tuned model halves its
# error on the most curved third of each surface.

def dummy_hooks(seed=0, n_shapes=12, conditions=2, npts=1500, n_tokens=64):
    rng = np.random.default_rng(seed)

    class Model:
        def __init__(self, kind):
            self.kind = kind

    def load(path):
        return Model(path)

    def cases(setting):
        for i in range(n_shapes):
            s = 0.8 + 0.4 * i / max(1, n_shapes - 1)
            X, N, A = _ellipsoid(npts, (1.0 * s, 0.8 * s, 0.25 * s))
            for c in range(conditions):
                f = -(1.5 + 0.5 * c) * np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2) / (0.3 * s * s)) * (X[:, 2] > 0) \
                    + 0.3 * X[:, 0] / s
                obs = np.sort(rng.choice(npts, 20, replace=False)) if setting != 'none' else np.zeros(0, np.int64)
                yield f'shape{i:02d}_c{c}', X, N, A, f, f'shape{i:02d}', obs

    def enc(model, X, N, A, observed, values):
        R = rms_radius(X, A)
        ti = np.sort(stable_rng(f'{len(X)}:{X[0, 0]:.6f}', seed).choice(len(X), n_tokens, replace=False,
                                                                         p=A / A.sum()))
        anchors = X[ti]
        owner = np.argmin(np.linalg.norm(X[:, None] - anchors[None], axis=2), axis=1)
        tw = np.bincount(owner, weights=A, minlength=n_tokens) + 1e-12
        kappa = shape_operator_norm(X, N, 0.25 * R) * R
        ok = np.isfinite(kappa)
        curved = ok & (kappa >= np.quantile(kappa[ok], 2 / 3))
        return {'R': R, 'anchors': anchors, 'tw': tw, 'curved': curved}, anchors, tw

    def profile(kind, Q, anchors, tw, R):
        d = np.linalg.norm(Q[:, None] - anchors[None], axis=2) / R
        width = {'pretrained': 0.1, 'finetuned': 0.1, 'scratch': 0.4}[kind]
        A = tw * np.exp(-d ** 2 / (2 * width ** 2))
        A /= A.sum(1, keepdims=True)
        if kind == 'finetuned':
            A = 0.8 * A + 0.2 * tw / tw.sum()
        return np.stack([np.stack([A] * 2)] * 2)          # [layers, heads, queries, tokens]

    def dec_att(model, tok, Q):
        return profile(model.kind, Q, tok['anchors'], tok['tw'], tok['R'])

    def enc_att(model, tok):
        return profile(model.kind, tok['anchors'], tok['anchors'], tok['tw'], tok['R'])

    def pred(model, tok, Q):
        raise RuntimeError('dummy predictions come from predict_for_case')

    return {'load_model': load, 'iterate_cases': cases, 'encode': enc, 'decoder_attention': dec_att,
            'encoder_attention': enc_att, 'predict': pred}


def _dummy_export(out, seed):
    """The dummy predictions need the true field, so they are written here rather than through predict()."""
    hooks = dummy_hooks(seed)
    hooks['predict'] = lambda model, tok, Q: np.zeros(len(Q))
    export(out, [('pretrained', 'pretrained'), ('finetuned', 'finetuned'), ('scratch', 'scratch')], 'none', 512,
           (), seed, hooks)
    rng = np.random.default_rng(seed + 1)
    for cid, X, N, A, f, group, observed in hooks['iterate_cases']('none'):
        tok, _, _ = hooks['encode'](None, X, N, A, observed, f[observed])
        w = A / A.sum()
        spread = float(np.sqrt(w @ (f - w @ f) ** 2))
        eb = rng.normal(0, 0.2 * spread, len(X))
        ea = eb * np.where(tok['curved'], 0.5, 1.0)
        path = os.path.join(out, 'attribution', f'{cid}.npz')
        z = dict(np.load(path))
        z.update(pred_finetuned=f + ea, pred_scratch=f + eb, region=np.where(tok['curved'], 0, 1))
        np.savez(path, **z)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', action='append', default=[], help='name=path; use pretrained, finetuned and scratch')
    ap.add_argument('--setting', default='none', help="observation setting passed to iterate_cases ('none' = no "
                                                      'sensors, the same input for every checkpoint)')
    ap.add_argument('--queries', type=int, default=2048, help='area-weighted surface queries per case')
    ap.add_argument('--predict', nargs='*', default=['finetuned', 'scratch'],
                    help='checkpoints whose predictions go into the attribution files')
    ap.add_argument('--raw', action='store_true', help='also store the attention weights (float16)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dummy', action='store_true', help='synthetic ellipsoids and an analytic stand-in model')
    args = ap.parse_args()
    if args.dummy:
        _dummy_export(args.out, args.seed)
        return
    if not args.ckpt:
        ap.error('give --ckpt name=path for pretrained, finetuned and scratch')
    hooks = {'load_model': load_model, 'iterate_cases': iterate_cases, 'encode': encode,
             'decoder_attention': decoder_attention, 'encoder_attention': encoder_attention, 'predict': predict}
    export(args.out, [tuple(c.split('=', 1)) for c in args.ckpt], args.setting, args.queries, set(args.predict),
           args.seed, hooks, args.raw)


if __name__ == '__main__':
    main()
