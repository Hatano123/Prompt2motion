# Prompt2Motion / ACT ベースライン引継ぎ資料

## 1. このワークスペースの目的

ACT（Action Chunking with Transformers）のシミュレーション課題 `sim_transfer_cube_scripted` を対象に、50デモから方策を学習し、定期的にロールアウト評価するための実験ワークスペースです。

主な設計方針は次のとおりです。

- Gitには実験制御スクリプト、環境定義、ACTへの改修パッチ、資料だけを保存する。
- データセット、チェックポイント、ログ、動画、外部リポジトリ、実行バイナリはGitに含めない。
- ホーム領域を圧迫しないよう、各PBS jobではリポジトリ、データ、Python環境をローカル `/tmp` に展開する。
- 学習状態を一定間隔で保存し、複数のPBS jobに分割してresumeできるようにする。
- 学習jobと評価jobをPBSの依存関係で直列化する。

## 2. 取得元とバージョン

### ACTベースライン

上流リポジトリは以下です。

- Repository: <https://github.com/tonyzhaozh/act.git>
- Branch: `research`
- Base commit: `742c753c0d4a5d87076c8f69e5628c79a8cc5488`

このプロジェクトでは上流コードをそのまま使わず、`patches/act-local.patch` を適用します。パッチには以下が含まれます。

- `ACT_DATA_DIR` と `ACT_NUM_EPISODES` によるデータセット指定
- optimizer、epoch、履歴、乱数状態を含むresume checkpoint
- 分割学習用の `--resume` と `--max_epochs_this_run`
- 定期checkpoint保存用の `--save_every`
- 任意checkpoint・任意ロールアウト数の評価
- 評価動画、結果、rollout単位JSONLの出力先分離

再取得とパッチ適用:

```bash
cd /path/to/Prompt2Motion
git clone https://github.com/tonyzhaozh/act.git repos/act
git -C repos/act checkout 742c753c0d4a5d87076c8f69e5628c79a8cc5488
git -C repos/act apply ../../patches/act-local.patch
git -C repos/act status --short
```

最後のコマンドで `.gitignore`、`constants.py`、`detr/main.py`、`imitate_episodes.py` の4ファイルだけが変更済みとして表示されれば想定どおりです。

### データセット

#### 来歴について確認できた事実

「Kaggleから取得したZIPを展開した」という記憶がありますが、Kaggleのdataset URL、slug、downloadコマンド、元ZIP名を確認できる記録は、このワークスペース内には残っていません。そのため、Kaggle由来であることは未確認情報として扱います。

現在置かれている `act_sim_transfer_cube_scripted_50episodes.zip` については、次のローカル記録が揃っています。

- PBS job `95652` の標準出力に `train_episodes=50` と生成用workdirが記録されている。
- `record_training_episodes.log` にepisode 0～49のMuJoCo scripted-policy rolloutと保存処理が記録されている。
- `archive_training_dataset.log` に同じworkdir内で50 memberをZIP化した記録がある。
- ZIP内部のHDF5更新時刻はjob内の生成時間帯と連続している。
- ACT公式READMEも、`record_sim_episodes.py --num_episodes 50` による生成を標準手順としている。

したがって、**現在のZIPは上記ACTリポジトリの `record_sim_episodes.py` と scripted policyを使い、MuJoCo上でローカル生成して圧縮したもの**と判断するのが、残存証拠と最も整合します。「ZIPから展開した」という点は正しく、各学習jobがこのローカルZIPを `/tmp` へ展開して使用します。

なお、ACT公式READMEにはscripted/human simulation demoの配布先としてGoogle Driveも案内されています。

- 公式案内: <https://github.com/tonyzhaozh/act#updates>
- 公式デモフォルダ: <https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link>

Kaggle由来の別データを過去に利用した可能性までは否定できません。URLやKaggle dataset slugが判明した場合は、本節に取得コマンド、取得日、元ファイル名、SHA-256を追記してください。出所が確定していないKaggleデータと現在の検証済みZIPを同一物として扱わないでください。

#### 現在の検証済みZIP

- Task: `sim_transfer_cube_scripted`
- Episodes: 50 (`episode_0.hdf5` ～ `episode_49.hdf5`)
- Cameras used by training: `top`
- Robot state/action dimension: 14
- Episode length: 400
- 生成job: PBS job `95652`
- 生成時のACT commit: `742c753c0d4a5d87076c8f69e5628c79a8cc5488`
- 展開時サイズ: 約18GB
- ZIPサイズ: 約507MiB
- ZIP SHA-256: `5029bf2e2696b6c53501a5eeef826813308c7de63cc8d98de1e050d56232c618`
- 配置先: `archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip`

既存アーカイブを引き継ぐ場合は、Gitとは別の共有ストレージ等で受け取り、次のように検証します。

```bash
mkdir -p archives/datasets
sha256sum archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip
unzip -t archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip
unzip -Z1 archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip \
  | grep -E '/episode_[0-9]+\.hdf5$' \
  | wc -l
```

最後の件数は `50` である必要があります。

#### データの用意方法

再現性の優先順位は次のとおりです。

1. 同じ実験を再現する場合は、上記SHA-256の検証済みZIPを共有ストレージから受け取る。
2. ZIPがない場合は、ACTの標準生成コードで50デモを再生成する。
3. ACT公式配布データを使う場合は、Google Driveから取得し、別の実験名で結果を分離する。
4. Kaggle版を使う場合は、dataset slugとファイルchecksumを先に記録し、現在のZIPとは別名で保存する。

アーカイブがない場合は、ACTを取得・パッチ適用した後に次のjobで再生成できます。このjobはデータ生成確認に加えて、最小の学習と評価も行います。

```bash
cd /path/to/Prompt2Motion
qsub -v TRAIN_EPISODES=50,TRAIN_EPOCHS=1,EVAL_ROLLOUTS=1,TRAIN_BATCH_SIZE=8,REQUIRED_TMP_GIB=50,ARCHIVE_TRAINING_DATASET=1 \
  scripts/run_act_smoke.pbs
```

成功すると `archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip` が生成されます。元データ生成コードは乱数seedを固定していないため、再生成したデータは同じ50デモ構成でも上記SHA-256とは一致しない可能性があります。

外部配布データを使う場合も、最終的にbaselineが要求する配置と構造は同じです。

```text
archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip
└── sim_transfer_cube_scripted/
    ├── episode_0.hdf5
    ├── ...
    └── episode_49.hdf5
```

外部ZIPのディレクトリ名やファイル名が異なる場合は、そのまま置き換えず、この構造へ正規化した新しいZIPを作り、出所とSHA-256を記録してください。

## 3. 実行環境

確認済み環境は次のとおりです。

- PBS queue: `Eduq`
- GPU: AMD Instinct MI210 × 1
- CPU: 4 cores/job
- Memory: 学習64GB、評価32GB
- ROCm対応PyTorch: `torch==2.8.0`、`torchvision==0.23.0`、ROCm 6.4 wheel
- Python: 3.10
- MuJoCo: 2.3.7
- dm_control: 1.0.14
- micromamba: 2.9.0で動作確認

必要コマンド:

```bash
qsub
qstat
rocm-smi
unzip
git
```

`tools/micromamba` はGit対象外です。Linux x86-64版micromambaを取得し、実行権限付きで次の場所へ配置してください。

```text
Prompt2Motion/tools/micromamba
```

各jobは `environment-act-rocm.yml` からconda環境を作り、続いて `requirements-act.txt` とROCm版PyTorchをインストールします。そのため、計算nodeからconda-forgeとPyTorch package indexへアクセスできる必要があります。

## 4. ディレクトリ設計

```text
Prompt2Motion/
├── ACT_BASELINE.md              現在の結果と短い実行メモ
├── HANDOVER.md                  本資料
├── environment-act-rocm.yml     micromamba環境の基礎定義
├── requirements-act.txt         Python依存関係
├── patches/
│   └── act-local.patch          上流ACTへ適用する必須差分
├── scripts/                     PBS投入・実行・集計スクリプト
├── repos/act/                   ACT checkout（Git対象外）
├── tools/micromamba             実行バイナリ（Git対象外）
├── archives/datasets/           データセットZIP（Git対象外）
├── checkpoints/act/             resume可能な学習状態（Git対象外）
├── results/act/                 ログ、評価値、動画（Git対象外）
└── logs/                        その他ログ（Git対象外）
```

永続化する大容量データはホーム側の `archives/`、`checkpoints/`、`results/` に置きます。一方、job実行中だけ必要な展開済みデータ、ACTコード、conda環境、cacheは `${TMP_WORK_BASE:-/tmp/${USER}_work}` 以下へ置き、正常終了・異常終了のどちらでも原則削除します。デバッグ時だけ `KEEP_TMP=1` を指定してください。

## 5. 新しい環境での初期セットアップ

```bash
git clone https://github.com/Hatano123/Prompt2motion.git
cd Prompt2motion

git clone https://github.com/tonyzhaozh/act.git repos/act
git -C repos/act checkout 742c753c0d4a5d87076c8f69e5628c79a8cc5488
git -C repos/act apply ../../patches/act-local.patch

mkdir -p tools archives/datasets checkpoints/act results/act
# tools/micromamba とデータセットZIPをここで配置する
chmod +x tools/micromamba scripts/*.sh scripts/*.pbs
```

配置確認:

```bash
test -x tools/micromamba
test -d repos/act/.git
test -s archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip
git -C repos/act rev-parse HEAD
sha256sum archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip
```

## 6. 推奨実行順序

### 6.1 resume smoke test

最初に2 epochを2つの学習jobへ分割し、resume後に1 rollout評価します。

```bash
cd /path/to/Prompt2Motion
./scripts/submit_act_smoke.sh
```

標準出力に3つのjob IDが表示されます。

```text
smoke_train_1=<job id>
smoke_resume_2=<job id>
smoke_eval=<job id>
```

完了確認:

```bash
qstat
cat checkpoints/act/act50_resume_smoke/metrics.jsonl
find results/act/act50_resume_smoke -maxdepth 4 -type f | sort
```

確認ポイント:

- 1つ目の `state_summary.txt` が `next_epoch=1`
- 2つ目のtrain logに `Resumed:` があり、`next_epoch=2`
- `checkpoints/act/act50_resume_smoke/training_state_last.ckpt` が存在
- `results/act/act50_resume_smoke/eval/epoch_2/` に評価結果と動画が存在

同名実験の古いcheckpointがあると自動resumeされます。完全な再実行では、既存成果物を退避してから別名を指定してください。

```bash
EXPERIMENT_NAME=act50_resume_smoke_v2 ./scripts/submit_act_smoke.sh
```

### 6.2 本学習と定期評価

デフォルトでは2000 epochを500 epochずつ4学習jobに分割し、各区間の後に50 rollout評価します。

```bash
./scripts/submit_act_baseline.sh
```

依存関係は次の順で自動設定されます。

```text
train 0-499 -> eval 500 -> train 500-999 -> eval 1000
 -> train 1000-1499 -> eval 1500 -> train 1500-1999 -> eval 2000
```

設定例:

```bash
EXPERIMENT_NAME=act50_baseline_v2 \
TOTAL_EPOCHS=2000 \
EPOCHS_PER_JOB=500 \
EVAL_ROLLOUTS=50 \
./scripts/submit_act_baseline.sh
```

`TOTAL_EPOCHS` は `EPOCHS_PER_JOB` で割り切れる必要があります。学習job内の主な既定値は `BATCH_SIZE=8`、`SAVE_EVERY=100`、`REQUIRED_TMP_GIB=30` です。

### 6.3 結果集計

全評価job完了後:

```bash
python3 scripts/summarize_act_baseline.py \
  --experiment act50_baseline
```

主な生成物:

- `loss_history.csv`: epochごとのtrain/validation loss
- `evaluation_summary.csv`: 評価epochごとのsuccess rateとaverage return
- `failure_cases.csv`: 失敗rolloutと動画の対応

## 7. checkpointとresumeの仕様

`checkpoints/act/<experiment>/training_state_last.ckpt` には次を保存します。

- 次に実行するepoch
- policy state
- optimizer state
- train/validation履歴
- best epochとminimum validation loss
- Python、NumPy、PyTorch、GPUの乱数状態
- 実験config

`run_act_baseline.pbs` はこのファイルが存在すると自動的に `--resume` を付けます。保存は一時ファイルへ書いた後に `os.replace` するため、書込み途中のcheckpointを正式ファイルとして残しにくい設計です。起動時には残存する `*.tmp` を削除します。

重要: `EXPERIMENT_NAME` が同じ実験は同じcheckpoint directoryを共有します。条件を変更する実験では必ず別名を使ってください。

## 8. ログと成果物の見方

学習job:

```text
checkpoints/act/<experiment>/
  training_state_last.ckpt
  policy_best.ckpt
  policy_last.ckpt
  policy_epoch_<zero-based epoch>_seed_0.ckpt
  dataset_stats.pkl
  metrics.jsonl
  train_val_*.png

results/act/<experiment>/jobs/<job id>/
  train.log
  gpu_usage.log
  state_summary.txt
  metrics_snapshot.jsonl
```

評価job:

```text
results/act/<experiment>/eval/epoch_<completed epochs>/
  eval.log
  summary.csv
  result_policy_*.txt
  rollouts_policy_*.jsonl
  video*.mp4
```

ファイル名の `policy_epoch_499` は「500 epoch完了時点」のモデルです。コード内部のepochが0始まりであるため、評価ディレクトリの `epoch_500` と1ずれます。

## 9. 既存の確認済み結果

旧パイプラインのPBS job `95652` では、50デモ・100 epoch・batch size 8・10 rolloutで完走しています。

- Best validation loss: `0.294329` at epoch 90（0始まり）
- Success rate: `0.0`
- Average return: `0.4`
- 使用GPU: AMD Instinct MI210

詳細は `ACT_BASELINE.md` を参照してください。既存の `results/` と `checkpoints/` はローカルには残っていますがGit対象外です。

## 10. よくある問題

### `src refspec main does not match any`

ローカルにコミットがない状態でpushした場合に発生します。現在のPrompt2Motionには初回コミット済みです。

### GitHubへpushできない

HTTPS認証またはSSH鍵が必要です。現在のremoteは次です。

```text
https://github.com/Hatano123/Prompt2motion.git
```

### `repos/act` がない

第2節のclone、checkout、patch適用を実行してください。`repos/` は意図的にGit管理対象外です。

### dataset archiveがない、または50件でない

既存ZIPを別ストレージから配置するか、第2節の生成jobを実行してください。`stage_act50_dataset` は `episode_0.hdf5` ～ `episode_49.hdf5` と総数50を検証してから学習します。

### `/tmp requires at least ... GiB free`

展開済みデータだけで約18GB必要です。空きのあるローカルscratchを `TMP_WORK_BASE` で指定するか、不要な一時データを管理者方針に従って整理してください。

```bash
TMP_WORK_BASE=/path/to/local/scratch ./scripts/submit_act_baseline.sh
```

ただし、submit scriptから渡されない環境変数はPBS側に自動継承されない構成もあります。その場合はPBS scriptの `qsub -v` 指定へ追加してください。

### GPUが見えない

PBSが設定する `CUDA_VISIBLE_DEVICES` をスクリプト内でROCm/PyTorch向けに正規化しています。`torch.cuda.is_available()` がfalse、または可視GPU数が1でない場合は安全のため停止します。まず `rocm-smi`、PBS資源割当、job log冒頭を確認してください。

### package installで失敗する

各jobが環境を新規作成するため、ネットワーク障害やpackage更新の影響を受けます。成功jobの `environment_freeze.txt` が残っている場合はバージョン比較に使ってください。長期的にはconda lockfileまたは事前構築済み環境への移行を推奨します。

## 11. 引継ぎ時に別経路で渡すもの

Git cloneだけでは実験を開始できません。次を別途用意してください。

1. `tools/micromamba`（または同等のmicromamba実行環境）
2. `archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip`（再生成する場合は不要）
3. 継続学習する場合は `checkpoints/act/<experiment>/` 一式
4. 過去結果を比較する場合は `results/act/<experiment>/` 一式

特にresumeにはpolicyだけでなく `training_state_last.ckpt` と `dataset_stats.pkl` が必要です。大容量成果物はGitへ追加せず、アクセス制御された共有ストレージで受け渡してください。
