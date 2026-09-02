#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='act50_baseline')
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    checkpoint_dir = args.workspace / 'checkpoints' / 'act' / args.experiment
    result_dir = args.workspace / 'results' / 'act' / args.experiment
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = checkpoint_dir / 'metrics.jsonl'
    loss_rows = []
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            item = json.loads(line)
            loss_rows.append({
                'epoch': item['epoch'] + 1,
                'train_loss': item['train']['loss'],
                'validation_loss': item['validation']['loss'],
                'best_epoch': item['best_epoch'] + 1,
                'min_validation_loss': item['min_val_loss'],
            })
    with (result_dir / 'loss_history.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=[
            'epoch', 'train_loss', 'validation_loss',
            'best_epoch', 'min_validation_loss'])
        writer.writeheader()
        writer.writerows(loss_rows)

    evaluation_rows = []
    for summary_path in sorted((result_dir / 'eval').glob('epoch_*/summary.csv')):
        with summary_path.open(newline='') as source:
            evaluation_rows.extend(csv.DictReader(source))
    with (result_dir / 'evaluation_summary.csv').open('w', newline='') as output:
        fieldnames = ['epoch', 'success_rate', 'average_return', 'rollouts', 'checkpoint', 'job_id']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_rows)

    failure_rows = []
    for rollout_path in sorted((result_dir / 'eval').glob('epoch_*/rollouts_*.jsonl')):
        epoch = int(rollout_path.parent.name.removeprefix('epoch_'))
        for line in rollout_path.read_text().splitlines():
            item = json.loads(line)
            if not item['success']:
                item['epoch'] = epoch
                item['video'] = str(rollout_path.parent / item['video'])
                failure_rows.append(item)
    with (result_dir / 'failure_cases.csv').open('w', newline='') as output:
        fieldnames = ['epoch', 'rollout', 'episode_return', 'highest_reward', 'success', 'video']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_rows)

    print(f'loss rows: {len(loss_rows)}')
    print(f'evaluation rows: {len(evaluation_rows)}')
    print(f'failure cases: {len(failure_rows)}')
    print(f'output: {result_dir}')


if __name__ == '__main__':
    main()
