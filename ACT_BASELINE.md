# ACT 50-demo Baseline

## 現在の到達点（2026-09-01）

- Dataset archive: `sim_transfer_cube_scripted`, 50 demonstrations, 50 members確認済み
- Model/Input/Output: ACT / top RGB + 14-dim robot state / left-right arm action
- GPU smoke: MI210 1枚で学習可能
- 既存 smoke (`95525`): 4 demos, 2 epochs, checkpoint・evaluation・動画保存に成功
- 既存 50-demo run (`95652`): 100 epochs完走
  - best validation loss: `0.294329` at epoch 90（0始まり）
  - 10 rollouts: Success Rate `0.0`, Average Return `0.4`
  - 推論動画10本: `results/act/qsub_95652/eval_video*.mp4`
- 新規実装: optimizer/epoch/RNG/loss履歴を含む resume checkpoint、定期policy保存、epoch別evaluation、failure-case一覧
- 未実行: 新しい resume smoke と 2000-epoch分割学習（PBS server接続復旧後に投入）

## Step 1: Resume Smoke Test

2 epochsを別々のPBS jobに分け、2番目が `training_state_last.ckpt` から再開します。最後に1 rollout評価します。

```bash
cd /home/0/y263357/Prompt2Motion
./scripts/submit_act_smoke.sh
```

確認項目:

- 1番目の `state_summary.txt`: `next_epoch=1`
- 2番目のtrain log: `Resumed: ... (next epoch: 1)`
- 2番目の `state_summary.txt`: `next_epoch=2`
- `checkpoints/act/act50_resume_smoke/training_state_last.ckpt` が存在
- `results/act/act50_resume_smoke/eval/epoch_2/` に評価結果と動画が存在

## Step 2/3: 本学習と定期評価

デフォルトは元ACT設定の2000 epochsを500 epochs x 4 jobsに分割し、各500 epochs後に50 rolloutsを評価します。依存関係は `train -> eval -> train(resume)` です。

```bash
cd /home/0/y263357/Prompt2Motion
./scripts/submit_act_baseline.sh
```

主な上書き可能値:

```bash
TOTAL_EPOCHS=2000 EPOCHS_PER_JOB=500 EVAL_ROLLOUTS=50 \
  ./scripts/submit_act_baseline.sh
```

学習checkpointは `checkpoints/act/act50_baseline/`、ジョブログと評価動画は `results/act/act50_baseline/` に保存されます。学習中は100 epochsごと、および各job末尾でcheckpointを保存します。

全evaluation終了後、表を集約します。

```bash
python3 scripts/summarize_act_baseline.py --experiment act50_baseline
```

生成物:

- `loss_history.csv`: epochごとのTraining/Validation Loss
- `evaluation_summary.csv`: 学習量ごとのSuccess Rate/Average Return
- `failure_cases.csv`: 失敗rolloutと対応する推論動画
