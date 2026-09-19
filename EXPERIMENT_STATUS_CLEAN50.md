# ACT視点比較 Clean-50 現状記録

## 1. 記録情報

| 項目 | 内容 |
|---|---|
| 記録日 | 2026-09-19（Asia/Tokyo） |
| 対応仕様書 | [EXPERIMENT_SPEC_CLEAN50.md](EXPERIMENT_SPEC_CLEAN50.md) |
| 対象プロジェクト | Prompt2Motion / ACT視点比較 |
| main repository基点 | `9fc3ce7fd6d805249b76b0b371ebeab5c8a93d74` |
| ACT upstream基点 | `742c753c0d4a5d87076c8f69e5628c79a8cc5488` |
| 記録時点の状態 | Clean-50実装完了、ジョブ未投入 |
| Git作業ブランチ | `experiment/clean50-protocol` |

現在の実装はGit作業ブランチのHEADと `patches/act-local.patch` を正本とする。
ACT upstream基点へ同パッチを適用することで、実験用ACTコードを再構築できる。

## 2. 仕様書との対応状況

| 仕様書 | 要求 | 2026-09-19時点 | 状態 |
|---|---|---|---|
| §1 | clean条件で3視点構成を比較 | pilotのみ完了。本実験は未実施 | 未完了 |
| §2 | multi-view主仮説、seed別評価 | 仮説と実験単位を仕様化 | 準備完了 |
| §3 | 成功50デモ、40/10 split、3 seed | 生成・学習コードを実装。50デモarchiveは未生成 | 未実施 |
| §4 | validation/test分離 | seed 1100/50件とseed 1200/100件を生成する実装済み | 準備完了 |
| §5 | CI、対応比較、40%ゲート | Wilson CI、exact McNemar、ゲート集計を実装 | 準備完了 |
| §6 | hash、split、metrics、動画 | 保存処理を実装。実成果物は未生成 | 未実施 |
| §7 | PBS一括実行 | 依存関係付き投入スクリプトを実装 | 投入待ち |
| §8 | 頑健性評価はclean合格後 | 新規頑健性ジョブは未投入 | 方針どおり |

## 3. 直近pilotの実績

実験名は `paired_pilot`。成功デモ10件、batch size 8、各モデル3 training seed、
共通の50初期状態で評価した。9学習runと9頑健性評価runはすべて完了している。
最後の完了は2026-09-18 20:29:38 JST、multi / training seed 2の頑健性評価である。

選択checkpointのclean結果は次のとおり。

| モデル | seed 0 | seed 1 | seed 2 | seed平均成功率 | seed平均return |
|---|---:|---:|---:|---:|---:|
| single_top | 6%（epoch 2000） | 4%（2000） | 4%（2000） | 4.67% | 115.79 |
| single_side | 2%（2000） | 10%（2000） | 2%（1500） | 4.67% | 102.61 |
| multi | 4%（1500） | 8%（1000） | 18%（1500） | 10.00% | 122.14 |

頑健性評価では、全モデル・全seed・全実施条件の成功率が0%だった。multiの
平均return（3 seed平均）は、missing top 7.05、missing side 0.71、
occlusion top 17.88、occlusion side 19.71だった。

この結果は最終的な視点比較結果としては使用しない。10デモpilotによる実行系確認と、
本実験の設計を改善する根拠として保存する。
Git掲載用の小容量集計は
[`reports/paired_pilot_2026-09-18`](reports/paired_pilot_2026-09-18/README.md) に保存した。

## 4. 妥当性監査の記録

確認できた妥当な点:

- 全モデル・seedで同じtop/side同時収録データを使用している。
- train episode IDとvalidation episode IDは全runで一致している。
- 評価用50初期状態は全runで共通かつ重複がない。
- カメラ構成以外の主要学習条件は一致している。
- dataset metadataでは10件すべて成功デモである。

pilotを最終結論に使えない理由:

- checkpoint選択と性能報告に同じ50初期状態を使用しており、選択バイアスがある。
- clean成功率が低く、頑健性評価に床効果が生じている。
- training seedだけを変えており、デモ集合の不確実性は評価していない。
- 従来実装は正規化統計にvalidation episodeも含めていた。
- ランダム遮蔽が対象物・手先・背景のどこを隠したか区別していない。
- rollout動画を保存しておらず、pilotの入力外乱と失敗段階を目視監査できない。

## 5. Clean-50向け実装済み項目

- `scripts/run_prepare_clean50_dataset.pbs`: 成功50デモと2種類の評価manifestを生成。
- `scripts/run_clean50_train.pbs`: 40/10 splitで学習し、validationだけでcheckpoint選択。
- `scripts/run_clean50_test.pbs`: 選択済みcheckpointを未見test 100状態で一度だけ評価。
- `scripts/submit_clean50.sh`: 3モデル×3 seedの学習・testをPBS依存関係付きで投入。
- `scripts/summarize_clean50.py`: Wilson CI、seed別結果、exact McNemar、40%ゲートを出力。
- `configs/clean50.json`: 実験条件を機械可読形式で固定。
- `repos/act/utils.py`: 正規化統計をtrain episodeだけから計算するよう変更。
- `repos/act/imitate_episodes.py`: 先頭N動画保存とrollout別遮蔽座標記録を実装。
- `tests/test_clean50_design.py`: validation/test混入防止などの設計テストを追加。

検証は8テスト成功、1テストskip。skipは現在のホストPythonに`h5py`がないためで、
データ生成PBSジョブではACT依存環境を作成してから同テストを実行する。
シェル構文、Python構文、Git diff whitespace checkは通過済み。

## 6. 現在存在する入力と未生成物

既存pilot入力:

| ファイル | SHA-256 |
|---|---|
| `act_sim_insertion_scripted_top_side_seed0_10episodes.zip` | `959d5d39edf22206938a8b9b9b5617fd75328547490163653c4573f23c711562` |
| `sim_insertion_eval_seed1000_50.json` | `4fb8dbc60ef7f155ab7c55c383f0d676a57522b16b8b6e21d39c8bc57f54fb86` |

Clean-50については、次の成果物はまだ存在しない。

- 50デモdataset archive
- validation seed 1100の50状態manifest
- test seed 1200の100状態manifest
- 9学習runのcheckpointとvalidation結果
- 9 held-out test結果と動画
- `heldout_test_summary.csv`、`paired_comparisons.json`、`clean_gate.json`

`results/act/clean50` 以下の実験成果物数は記録時点で0である。

## 7. 次回再開時の手順

1. `experiment/clean50-protocol` のHEADと差分を確認する。
2. PBSの利用可能容量を確認する。非圧縮50デモは概算約36.9 GBとなる。
3. `bash scripts/submit_clean50.sh`でデータ生成、9学習、9 testを直列投入する。
4. 生成ジョブのテスト結果とdataset metadata（50成功、camera順）を確認する。
5. 全test完了後、`python3 scripts/summarize_clean50.py`を実行する。
6. 仕様書§5の40%ゲートに従い、頑健性評価へ進むか判断する。

容量確認（2026-09-19）ではhome空き約5 GB、ユーザー用共有work領域は未割当だった。
このため、圧縮dataset archiveはhomeに保持し、各runのtest成功後に評価済みselected
checkpointを削除する。再評価には同じ入力hashと設定からの再学習が必要となる。

## 8. 変更管理

この文書は事実のスナップショットであり、実験条件の正本は
[EXPERIMENT_SPEC_CLEAN50.md](EXPERIMENT_SPEC_CLEAN50.md) とする。条件変更時は先に仕様書を
更新し、この文書には変更日、理由、投入済みjobへの影響を追記する。実験結果の追記では、
job ID、完了日時、入力hash、選択epoch、集計ファイルへの相対パスを記録する。
