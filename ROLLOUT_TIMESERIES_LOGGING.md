# ACT rollout時系列ログ仕様

Phase B以降の`sim_insertion_scripted`評価では、既存のresult TXT、rollout JSONL、動画に加えて、`--save_timeseries`指定時に詳細ログを保存する。

## 出力構成

```text
eval_output_dir/
  rollouts_<checkpoint>.jsonl       # 既存形式（変更なし）
  result_<checkpoint>.txt           # 既存形式（変更なし）
  video*.mp4                        # 既存設定に従う
  timeseries/
    index.jsonl
    rollout_000_summary.json
    rollout_000_timeseries.npz
    ...
```

NPZには画像を含めず、数値配列をfloat32中心で圧縮保存する。

## Timestepの整列

時刻`t`の`observed_qpos`をpolicyへ入力し、得られた`policy_raw_action`をdataset statisticsで逆正規化した14次元絶対joint commandが`action`および`target_qpos`である。これを`env.step`へ渡した後のreward、`post_step_qpos`、object pose、contactを同じ行`t`へ保存する。

### 主な配列

| 配列 | shape | 取得元 |
|---|---:|---|
| `timestep` | `(T,)` | evaluation loop index |
| `reward` | `(T,)` | `env.step`後の`TimeStep.reward` |
| `policy_raw_action` | `(T,14)` | ACT出力、逆正規化前 |
| `action`, `target_qpos` | `(T,14)` | `env.step`へ渡す絶対joint command |
| `observed_qpos` | `(T,14)` | action適用前のobservation |
| `post_step_qpos` | `(T,14)` | action適用後のobservation |
| `observed_gripper_state` | `(T,2)` | `observed_qpos[:, [6,13]]` |
| `post_step_gripper_state` | `(T,2)` | `post_step_qpos[:, [6,13]]` |
| `peg_position` | `(T,3)` | action適用後の`env_state[0:3]` |
| `peg_quaternion_wxyz` | `(T,4)` | action適用後の`env_state[3:7]` |
| `socket_position` | `(T,3)` | action適用後の`env_state[7:10]` |
| `socket_quaternion_wxyz` | `(T,4)` | action適用後の`env_state[10:14]` |

`env_state`は`InsertionTask.get_env_state()`が返す`physics.data.qpos[16:]`で、XML上の`red_peg_joint`、`blue_socket_joint`はいずれもfree jointである。

### Contact

各stepでMuJoCoの`physics.data.contact`を読み、以下を保存する。

- `contact_count`
- `contact_geom1_id`, `contact_geom2_id`
- `contact_geom1_name`, `contact_geom2_name`
- `contact_peg_socket`
- `contact_peg_pin`
- `contact_right_gripper_peg`
- `contact_left_gripper_socket`
- `contact_peg_table`, `contact_socket_table`

raw contact配列はepisode内の最大contact数へpaddingし、IDは`-1`、nameは空文字をpadding値とする。意味フラグはgeom順序に依存しない。

## Episode summary

JSONにはmodel/view、training/evaluation seed、rollout/state ID、manifest pathとSHA-256、initial state、checkpoint pathとSHA-256、condition、episode length、return、highest reward、successを保存する。

reward系列から`reward >= n`へ初めて到達したstepとして`first_reward1_timestep`から`first_reward4_timestep`を計算する。未到達は`null`。`reward3_remaining_steps`と`reward3_to_reward4_steps`も保存する。

## 実行

通常評価へ次を追加する。

```bash
--save_timeseries --model_name single_side
```

既存Clean-50 test PBSは`SAVE_TIMESERIES=1`を既定とし、不要な場合だけ`SAVE_TIMESERIES=0`で無効化できる。

検証:

```bash
python scripts/validate_rollout_timeseries.py \
  results/.../timeseries --manifest archives/environments/<manifest>.json
```

logging ON/OFFの実checkpoint比較は`run_rollout_logging_smoke.pbs`が同一checkpoint・manifestを別processで各1--3 rollout実行し、initial state、observed qpos、action、reward、successをexact比較する。

## 容量

400 step、14次元action/qpos、object pose、4 contact pairを含む圧縮fixtureは1 episode 139,284 bytesだった。100 rolloutでは約13.9 MB（13.3 MiB）。実容量はcontact数とtrajectoryに依存するため、最初のPhase B smoke testでvalidatorの実測値を確認する。

## 現時点の検証範囲

依存なしunit testでは、round-trip、shape、dtype、milestone、NaN/Inf、manifest state、contact判定、collectorの非破壊性を確認済み。既存Clean-50 checkpointは容量対策で削除済みのため、実checkpointによるON/OFF smoke testはまだ実行していない。Phase B seed 0の最初のcheckpointが得られた時点で、full test前にsmoke PBSを通す。
