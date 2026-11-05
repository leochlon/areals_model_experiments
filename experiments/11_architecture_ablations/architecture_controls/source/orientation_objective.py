import argparse
from dataclasses import replace

import architecture as study

native_build = study.build


def build(arm, seed):
    assert arm == 'unoriented_vector_target'
    model, config, shared = native_build('oriented_normal_target', seed)
    # This pair changes only signed versus absolute-cosine supervision.
    # The model, vector head, initial tensors, geometry inputs and augmentation stay identical.
    config = replace(config, train=replace(config.train, normal_loss='unoriented'))
    return model, config, shared


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    args = p.parse_args()
    args.arm = 'unoriented_vector_target'
    study.build = build
    study.main(args)
