#!/usr/bin/env python3
"""Summarize held-out Clean-50 test results and apply the preregistered gate."""

import argparse
import csv
import json
import math
import os


def wilson(successes, count, z=1.959963984540054):
    rate = successes / count
    scale = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / scale
    return center - radius, center + radius


def exact_mcnemar(first, second):
    """Return discordant counts and an exact two-sided binomial p-value."""
    first_only = sum(a and not b for a, b in zip(first, second))
    second_only = sum(b and not a for a, b in zip(first, second))
    discordant = first_only + second_only
    if not discordant:
        return first_only, second_only, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(first_only, second_only) + 1))
    p_value = min(1.0, 2 * tail / (2 ** discordant))
    return first_only, second_only, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/act/clean50')
    parser.add_argument('--output', default='results/act/clean50/heldout_test_summary.csv')
    parser.add_argument('--gate-output', default='results/act/clean50/clean_gate.json')
    parser.add_argument('--paired-output', default='results/act/clean50/paired_comparisons.json')
    parser.add_argument('--gate', type=float, default=0.4)
    args = parser.parse_args()
    rows = []
    outcomes = {}
    for model in ('single_top', 'single_side', 'multi'):
        for seed in (0, 1, 2):
            run_dir = os.path.join(args.root, model, f'train_seed{seed}')
            path = os.path.join(run_dir, 'eval_test', 'rollouts_selected_policy.jsonl')
            summary_path = os.path.join(run_dir, 'validation_summary.json')
            if not os.path.exists(path) or not os.path.exists(summary_path):
                continue
            rollouts = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
            outcomes[(model, seed)] = [bool(row['success']) for row in rollouts]
            summary = json.load(open(summary_path, encoding='utf-8'))
            successes = sum(bool(row['success']) for row in rollouts)
            rate = successes / len(rollouts)
            low, high = wilson(successes, len(rollouts))
            rows.append({'model': model, 'training_seed': seed,
                         'selected_epoch': summary['selected_epoch'], 'successes': successes,
                         'rollouts': len(rollouts), 'success_rate': rate,
                         'ci95_low': low, 'ci95_high': high,
                         'average_return': sum(row['episode_return'] for row in rollouts) / len(rollouts)})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fields = ('model', 'training_seed', 'selected_epoch', 'successes', 'rollouts',
              'success_rate', 'ci95_low', 'ci95_high', 'average_return')
    with open(args.output, 'w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    gate = {}
    for model in ('single_top', 'single_side', 'multi'):
        rates = [row['success_rate'] for row in rows if row['model'] == model]
        passing = sum(rate >= args.gate for rate in rates)
        gate[model] = {'completed_seeds': len(rates), 'passing_seeds': passing,
                       'threshold': args.gate, 'pass': len(rates) == 3 and passing >= 2}
    with open(args.gate_output, 'w', encoding='utf-8') as target:
        json.dump({'rule': 'at least 2 of 3 seeds meet threshold', 'models': gate}, target, indent=2)
        target.write('\n')
    comparisons = []
    for seed in (0, 1, 2):
        for baseline in ('single_top', 'single_side'):
            multi = outcomes.get(('multi', seed)); single = outcomes.get((baseline, seed))
            if multi is None or single is None:
                continue
            if len(multi) != len(single):
                raise ValueError('Paired test requires equal rollout counts')
            multi_only, single_only, p_value = exact_mcnemar(multi, single)
            comparisons.append({'training_seed': seed, 'comparison': f'multi_vs_{baseline}',
                                'multi_only_success': multi_only,
                                'single_only_success': single_only,
                                'exact_mcnemar_p_value': p_value})
    with open(args.paired_output, 'w', encoding='utf-8') as target:
        json.dump(comparisons, target, indent=2); target.write('\n')
    print(f'wrote {len(rows)} held-out rows to {args.output}')
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
