import argparse
from types import MethodType

import architecture as study
from areal_tokens.model import Rotary3D

native_build = study.build


def identity(self, value, positions):
    return value


def strip_rotary(model):
    for module in model.modules():
        if isinstance(module, Rotary3D):
            module.forward = MethodType(identity, module)
    return model


def build(arm, seed):
    original = {'anchored_no_rotary': 'reference', 'unanchored_no_rotary': 'unanchored_latents'}[arm]
    model, config, shared = native_build(original, seed)
    return strip_rotary(model), config, shared


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=['anchored_no_rotary', 'unanchored_no_rotary'])
    p.add_argument('--seed', required=True, type=int, choices=[0, 1, 2])
    args = p.parse_args()
    study.build = build
    study.main(args)
