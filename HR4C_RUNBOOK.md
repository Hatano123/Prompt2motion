# HR4C最小実験ランブック

## 現在できること

実機SDKに依存しない境界として、`Observation -> policy target -> SafetyFilter -> adapter -> HDF5` を実装した。次でモックの20 stepを実行できる。

```bash
python3 scripts/run_hr4c_mock.py
python3 -m unittest tests.test_hr4c_runtime
```

`h5py` がある学習環境では、生成HDF5はACTと同じ `action`, `observations/qpos`, `observations/images/{top,side}` を持ち、追加で各timestampを保存する。`h5py` がない接続確認環境では診断用NPZへフォールバックする（NPZはACT学習へ直接投入しない）。

## 実機接続前ゲート（未完了なら動かさない）

- [ ] メーカー安全マニュアルと現場リスクアセスメントを承認済み
- [ ] 納入機の左右腕・グリッパ・カメラ構成を実見して記録済み
- [ ] joint名、順序、単位、hard/soft limitをSDK値と照合済み
- [ ] 非常停止、保護停止、watchdog、通信断時停止を単独試験済み
- [ ] top/sideのtimestamp基準と最大skewを実測済み
- [ ] 無負荷・低速・単関節でcommand_positions相当APIを確認済み
- [ ] 無人workspaceと非常停止担当者を確保済み

`configs/hr4c_mock.json` の関節範囲は試験用の仮値であり、実機へ転用禁止。実機アダプタはメーカーAPI入手後に `read()`, `command_positions()`, `stop()` の3境界を実装し、設定にはSDK由来の値と出典を記録する。

## 本番への順序

1. SDK/URDF/topic一覧を入手し、実機configとadapterを実装する。
2. commandを無効にして1000 frame収録し、欠損率、camera skew、state ageを集計する。
3. 固定軌道を最低速度で1回、次に10回実行し、安全停止ログを確認する。
4. 成功教示10 episodeでpilotを行い、top / side / multiを同一splitで学習する。
5. 50成功episode、3 seedへ拡張し、各モデルを共通配置でclean 30試行する。
6. clean合格後のみ、片視点遮蔽とcamera missingを各30試行する。

本番の各試行では、episode ID、モデル、seed、初期配置、条件、成功段階、介入、停止理由をmanifestへ必ず残す。
