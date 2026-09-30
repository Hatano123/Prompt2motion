# Phase B–E 実験パイプライン

更新日: 2026-09-27

このパイプラインは既存成果物を削除・上書きしない。完了markerがあるrunはskipし、phase gate、固定manifest、
checkpoint hash、空き容量reserveを満たさないジョブは投入しない。全体仕様は
`configs/experiment_pipeline.json`に固定する。

## 実装範囲

- Phase B: 5/10/25/40 demos × top/side/multi × seed 0/1/2の固定10,000 optimizer-step学習、
  deterministic validation選択、held-out test、詳細時系列ログ。
  40-demoの過去指標はClean-50に保持されているがcheckpoint本体が削除済みのため、既存結果を変更せず
  `phase_b/40demos`へ再構築する。
  5/10-demoでも学習メタデータのみ残りcheckpointが無いrunは、既存directoryを変更せず
  `phase_b_rebuild`へ再構築する。
- Phase C: 10/40 demosのclean、25%/50%遮蔽、camera missing、3 cm + 5° camera pose shift。
  同じ100初期状態を使い、各使用cameraを個別に外乱させる。
- Phase D: modality dropout baselineとViewDrop-ACT。後者はview別backbone、availability embedding、
  feature scale正規化、信頼度gate、missing/local occlusion混合学習を用いる。
- Phase E: vendor adapter境界、ACT互換収録、観測鮮度・camera同期・関節limit・step limit・deadmanを
  fail-closedで検査する。納入構成と安全確認が未完了のため、現状はmockとoffline readiness auditまで。

## 実施順序

まずdry-runで候補を確認する（JSONを標準出力するだけで、ジョブは投入しない）。

```bash
python3 scripts/plan_experiment_pipeline.py --stage all --max-jobs 10
```

現在実行可能な前段ジョブを投入する場合のみ`--submit`を付ける。容量reserveを越えると自動停止する。

```bash
python3 scripts/plan_experiment_pipeline.py --stage select --max-jobs 1 --submit
python3 scripts/plan_experiment_pipeline.py --stage test --max-jobs 1 --submit
```

各段階の完了後、immutable gateを作る。監査が失敗した場合は出力ファイルを作らない。

```bash
python3 scripts/audit_phase_gate.py --phase b0 \
  --output results/act/phase_b/preflight/phase_b0_complete.json
python3 scripts/audit_phase_gate.py --phase b \
  --output results/act/phase_b/preflight/phase_b_complete.json
python3 scripts/audit_phase_gate.py --phase c \
  --output results/act/phase_c/phase_c_complete.json
python3 scripts/audit_phase_gate.py --phase d \
  --output results/act/phase_d/phase_d_complete.json
```

Phase B seed 1/2はB0 gate、Phase CはPhase B gate、Phase DはPhase C gateがなければplanner上でblockedになる。

## 集計

各`train_seed*`のcondition directoryを指定すると、成功率、cleanからのpaired差、paired bootstrap 95% CI、
exact McNemar p値をCSV/JSON/Markdownへ出力する。出力先は新規directoryを指定する。

```bash
python3 scripts/summarize_phase_b.py \
  --output results/act/phase_b/analysis/summary_YYYYMMDD

python3 scripts/summarize_robustness.py \
  results/act/phase_c/10demos/multi/train_seed0 \
  --output results/act/phase_c/analysis/10demos_multi_seed0
```

## Phase E readiness

`configs/hr4c_hardware.template.json`を別名へコピーして納入仕様を記入する。templateのままでは必ず失敗し、
hardwareへ接続しない。

```bash
python3 scripts/audit_hr4c_readiness.py configs/hr4c_hardware.template.json
python3 scripts/run_hr4c_mock.py
```

実機adapterを追加しても、readiness audit、メーカー手順によるE-stop確認、protective stop、双腕同期停止、
workspace/挿入力limitの全項目がtrueになるまではhardware実行を許可しない。

## 容量

plannerは常に2 GiBをreserveする。見積りはnaive/dropout学習1 runあたり約1.34 GB、gated学習は
保守的に2.7 GB、100 rollout時系列評価は実測約43 MBに余裕を加え50 MBである。実行中、queue待ち、
依存hold中のPBSジョブも将来使用量として予約する。既存成果物を保持したままPhase B–Dを全完了する
概算追加容量は約65 GBであり、現quotaの空きだけでは足りない。外部保存先またはquota増量後も、
設定ファイルの見積りと実測値を更新してから投入する。
## checkpointの選択復帰

Google Driveから取得した分割バックアップは、既存成果物を上書きせず、Phase B seed 0の
1 runずつ復帰できる。最初にdry-runする。

```bash
scripts/restore_phase_b_checkpoints.sh \
  --archive-dir /path/to/archive_parts_complete \
  --demo-count 5 \
  --view-config single_top \
  --output-namespace phase_b_rebuild \
  --dry-run
```

4個のcheckpoint候補を復帰する場合は`--dry-run`を外す。demo数は`5`または`10`、viewは
`single_top`、`single_side`、`multi`を指定する。スクリプトは`SHA256SUMS.txt`、分割番号の連続性、
各checkpointのサイズとSHA-256、復帰先の2 GiB reserveを検査し、既存ファイルは置換しない。
既存runに未完了の評価物がある場合は`--output-namespace phase_b_rebuild`を指定し、学習メタデータと
checkpointだけを独立namespaceへ復帰する。
