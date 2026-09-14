# Single-view / Multi-view 設定差分

比較対象はMulti-view job `99396` とSingle-view baselineである。

| 項目 | Single-view | Multi-view |
|---|---|---|
| `camera_names` | `top` | `top,side` |
| Vision backbone数 | 1 | 2 |

上記以外の指定実験条件は同一: `sim_insertion_scripted`、10 demonstrations、
8/2 split、training seed 0、split seed 1、100 epochs、batch size 1、
learning rate `1e-5`、ACT hyperparameters、Clean評価20 rollouts、評価seed 1000。

注意: Multi-view job `99396` の生成データはjob終了時に削除されている。このため
Single-viewは同じ生成分布とsplit規則を使うが、episodeの実体は同一ではない。
