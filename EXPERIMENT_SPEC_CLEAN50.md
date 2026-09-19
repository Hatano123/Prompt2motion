# ACT視点比較 Clean-50 実験仕様書

> 現在の進捗、直近の結果、監査結果、未実施項目は
> [EXPERIMENT_STATUS_CLEAN50.md](EXPERIMENT_STATUS_CLEAN50.md) を参照する。
> 状況記録の基準日は2026-09-19（Asia/Tokyo）。

## 1. 目的

同一の成功デモ、分割、評価初期状態を用い、`single_top`、`single_side`、
`multi`（top+side）のclean条件における性能差を検証する。本実験は頑健性評価へ
進む前の性能ゲートであり、test結果をcheckpoint選択に使用しない。

## 2. 仮説と実験単位

- 主仮説: multi-viewのタスク成功率は最良のsingle-viewより高い。
- 副指標: 平均return、報酬段階別到達率、学習seed間変動。
- 学習の実験単位はtraining seed、rollout比較の単位は共通初期状態とする。
- デモ集合は本実験では1集合に固定するため、異なるデモ集合への一般化は主張しない。

## 3. 固定条件

| 項目 | 仕様 |
|---|---|
| task | `sim_insertion_scripted` |
| デモ | 成功50 episode、生成seed 0、top/side同時収録 |
| split | episode 0–39=train、40–49=validation |
| モデル | single_top / single_side / multi |
| training seed | 0 / 1 / 2 |
| epoch | 2,000（500刻みで候補保存） |
| batch size | 8 |
| optimizer設定 | lr `1e-5`、chunk 100、KL 10 |
| 正規化統計 | train 40 episodeのみ |

## 4. 評価分離

1. Validation rollout: seed 1100の固定50状態。4 checkpointから成功率、
   平均return、早いepochの順で1つを選ぶ。
2. Test rollout: seed 1200の固定100状態。選択済みcheckpointを一度だけ評価する。
3. test結果を使ったcheckpoint・ハイパーパラメータの再選択は禁止する。

manifestは事前生成し、全モデル・seedで共有する。各test評価では先頭5 rolloutの
動画を保存し、カメラ入力、成功判定、代表的失敗を目視確認する。

## 5. 主要解析とゲート

- seed別成功率とWilson 95%信頼区間を報告する。
- 共通初期状態を用いた対応比較（McNemar検定または対応bootstrap）を行う。
- 3 seedを独立な300 rolloutとして単純に扱わず、seed別結果を必ず併記する。
- 3 seed中2 seed以上でclean成功率40%以上なら頑健性評価へ進む。
- 20–40%ならデータ量・学習設定を再検討し、20%未満ならデータ、成功判定、
  タスク難度、方策設定を診断する。この閾値は統計的有意性ではなく運用ゲートである。

## 6. 成果物と再現性

- dataset archive、validation/test manifestのSHA-256
- split manifest、train-only正規化統計、学習metrics
- validation全候補、選択epoch、test rollout JSONL、先頭5動画
- モデル、camera順序、training seed、ジョブログ

結果ルートは `results/act/clean50/<model>/train_seed<seed>` とする。

## 7. 実行

```bash
bash scripts/submit_clean50.sh
```

データが未作成なら生成ジョブを先行投入し、9学習ジョブと各testジョブを依存関係付きで
投入する。ホーム容量制約下ではrunを直列実行し、test成功後に評価済みselected
checkpointを削除する。学習履歴、選択epoch、test結果、動画、hashは保持する。
完了後の集計は次で行う。

```bash
python3 scripts/summarize_clean50.py
```

## 8. スコープ外

遮蔽、camera missing、位置ずれ、camera dropout学習はcleanゲート通過後の別実験とする。
本実験だけから実機性能や異なるデモ集合への一般化は主張しない。
