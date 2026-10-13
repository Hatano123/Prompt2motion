# HR4Cとシミュレーション実験の比較・整理候補（2026-10-13）

完了済みPhase B探索test 23 runと、HR4C側の保存済み小成果物 4 runを集計した。シミュレーションは各run 100 rollout。HR4Cは保存済みstatus/configから取得した。新たな学習・実機試験は行っていない。

## シミュレーションの結果

| デモ数 | 視点 | 完了seed数 | 平均成功率 | seed間標準偏差 |
|---:|---|---:|---:|---:|
| 5 | multi | 1 | 3.0% | — |
| 5 | single_side | 1 | 0.0% | — |
| 5 | single_top | 1 | 5.0% | — |
| 10 | multi | 2 | 7.5% | 2.1ポイント |
| 10 | single_side | 2 | 4.5% | 3.5ポイント |
| 10 | single_top | 2 | 4.5% | 6.4ポイント |
| 25 | multi | 2 | 21.5% | 9.2ポイント |
| 25 | single_side | 3 | 24.3% | 13.7ポイント |
| 25 | single_top | 3 | 28.3% | 0.6ポイント |
| 40 | multi | 2 | 41.5% | 7.8ポイント |
| 40 | single_side | 2 | 54.5% | 26.2ポイント |
| 40 | single_top | 2 | 31.5% | 6.4ポイント |

固定subset・共通探索test seed 1200の結果であり、独立subsetや未閲覧の確認用testではない。完了seed数が異なる群の単純な順位付けは避ける。個別結果と元ファイルの対応は simulation_runs.csv に保存した。

## HR4C側の結果

| 実験 | 初期化 | optimizer step | validation MAE (rad) | checkpoint |
|---|---|---:|---:|---|
| hr4c_controlled_gpu_pretrained_low_20261012_2201 | pretrained | 1000 | 0.035070 | True |
| hr4c_controlled_gpu_random_low_20261012_2231 | random | 1000 | 0.044185 | True |
| hr4c_controlled_gpu_smoke_20261012_2101 | pretrained | 100 | 0.078860 | True |
| hr4c_gpu_cost_probe_20261012_2021 | not_recorded | 20 | — | False |

1000-step比較では事前学習初期化のMAEは0.035070 rad、ランダム初期化は0.044185 radで、約20.6%低い。これはvalidation上の比較であり、実機成功率や未閲覧test性能の証拠ではない。保存済みstatusではrobot_commandsは0。コストprobe・100-step smokeと1000-step比較は目的が異なる。

シミュレーションの成功率とHR4Cの関節誤差は単位も評価方法も異なるため、直接優劣を比較できない。HR4Cの実機成功率はこの成果物には記録されていない。

## 保持・退避・削除候補

| 対象 | 判断 | 理由・削除条件 |
|---|---|---|
| hr4c/、configs/hr4c_mock.json、HR4C_RUNBOOK.md | 保持 | HR4C接続境界・mock・設定 |
| repos/act のpolicy.py・detrと環境定義 | 保持 | HR4C学習がACTのpolicy/backbone実装を利用。シミュレーション専用と一括判定しない |
| hr4c_bridge のデータ・コードsnapshot・選択重み・manifest | 保持／永続退避 | HR4Cの再現性に必要。/tmpの救出コピーはバックアップではない |
| Phase Bの選択重み・manifest・結果CSV・receipt | 永続退避後に検討 | 比較根拠を残す。Windows側コピーの現時点での再検証が必要 |
| Phase Bの非選択checkpoint | 条件付き削除候補 | 既存controllerはWindows側で全件ハッシュ照合・受領後にのみ削除する |
| 完了実験の動画・詳細timeseries・古いログ | 退避後の整理候補 | 研究記録と障害調査に使うため、未検証のまま一括削除しない |
| シミュレーション実験スクリプト・計画・テスト | 現時点では保持 | runnerの固定計画が未完了。HR4C専用化は別の明示した変更として依存を除去してから行う |

本作業ではファイル削除を実行していない。削除範囲の回答待ち。既存のWindows受領記録だけでは現在のWindowsコピーの健全性を再確認できないため、削除承認と再検証がそろうまで保持する。
