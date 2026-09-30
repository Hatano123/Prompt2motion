# 実験進捗・暫定結果（2026-09-30）

この文書は実験中の暫定スナップショットである。Clean-50の確定結果は
[`EXPERIMENT_RESULTS_CLEAN50_PHASE_A.md`](EXPERIMENT_RESULTS_CLEAN50_PHASE_A.md)を正とする。

## 現在地

| Phase | 状態 | 備考 |
|---|---|---|
| A: Clean-50視点比較・診断 | 完了 | 3 seed × 100 held-out state の確定解析済み |
| B: デモ数・seed感度 | 実行中 | seed 0 の5/25デモ評価済み。10/40デモはPBSに投入済み |
| C: camera外乱paired評価 | 実験基盤完成 | Bの確定checkpointを入力として待機 |
| D: ViewDrop-ACT | 実験基盤完成 | Cのbaseline後に比較する |
| E: HR4C実機移行 | mock/安全監査完成 | 実機仕様・SDK・安全資料が未設定のため接続はロック |

## Phase Aの確定結果

Clean-50（40学習デモ、3 training seed）ではheld-out成功率のseed平均は
single-top 42.3%、single-side 50.0%、multi 40.0%だった。naive multi-viewの平均改善は
確認されず、最終挿入段階での失敗が主な差だった。詳細なpaired解析と制約はPhase A報告書にある。

## Phase B seed 0 の暫定結果

全行とも独立held-out 100 state、deterministic checkpoint選択後の評価である。
まだ各条件1 seedだけなので、モデル間の優劣を結論づけない。

| 学習デモ数 | single-top | single-side | multi |
|---:|---:|---:|---:|
| 5 | 5/100 (5%) | 0/100 (0%) | 3/100 (3%) |
| 25 | 28/100 (28%) | 12/100 (12%) | 28/100 (28%) |

平均returnは、5デモで top 64.66 / side 34.64 / multi 68.04、25デモで
top 309.99 / side 266.06 / multi 295.88 だった。

10デモはtopのtestが実行中、side testとmulti選択・testがキュー／依存待ちである。
40デモはsingle-topとsingle-sideの学習がキュー済みで、multiは容量reservationを満たし次第投入する。
Phase Bのseed 1/2、ならびにPhase C/DはB0のimmutable gate後に開始する。

## 容量と再現性

現在のホーム空きは約4.8 GiBである。未選択checkpointは、選択完了・ハッシュ検証・Drive readback検証を
満たしたものだけを外部退避する運用にしている。root直下の完了PBSログは結果本体ではないため、
`scripts/clean_obsolete_pbs_logs.sh --apply` で実行中／依存中ジョブを自動除外して削除できる。
`results/`以下のcheckpoint、rollout、manifest、選択結果はこのスクリプトの対象外である。

実行計画、依存関係、容量planner、gate条件は[`EXPERIMENT_PIPELINE.md`](EXPERIMENT_PIPELINE.md)に記載した。
