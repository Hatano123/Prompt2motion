#!/usr/bin/env python3
"""Create deterministic insertion initial states without launching MuJoCo."""

import argparse
import json
import os

import numpy as np


def generate_states(seed, count):
    rng = np.random.RandomState(seed)
    states = []
    for _ in range(count):
        peg_xyz = rng.uniform([0.1, 0.4, 0.05], [0.2, 0.6, 0.05])
        socket_xyz = rng.uniform([-0.2, 0.4, 0.05], [-0.1, 0.6, 0.05])
        states.append(np.concatenate([
            peg_xyz, [1.0, 0.0, 0.0, 0.0],
            socket_xyz, [1.0, 0.0, 0.0, 0.0],
        ]).tolist())
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--count', type=int, default=50)
    args = parser.parse_args()

    payload = {
        'task_name': 'sim_insertion_scripted',
        'eval_seed': args.seed,
        'initial_states': generate_states(args.seed, args.count),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = f'{args.output}.tmp'
    with open(temporary, 'w') as target:
        json.dump(payload, target, indent=2, sort_keys=True)
        target.write('\n')
    os.replace(temporary, args.output)


if __name__ == '__main__':
    main()
