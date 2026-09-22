# Clean-50実験結果・Phase A診断報告

作成日: 2026-09-22  
対象: ACT `sim_insertion_scripted` 視点比較  
実験仕様: [EXPERIMENT_SPEC_CLEAN50.md](EXPERIMENT_SPEC_CLEAN50.md)  
研究計画: [EXPERIMENT_PLAN_HR4C.md](EXPERIMENT_PLAN_HR4C.md)

## 1. 要約

Clean-50では、同じ成功デモ、episode split、validation初期状態、held-out test初期状態を使い、
`single_top`、`single_side`、`multi`（top+side）を3 training seedで比較した。データ生成、9学習、
9 held-out testの全19 PBSジョブはExit status 0で完了した。

held-out test成功率のseed平均は、single-top 42.3%、single-side 50.0%、multi 40.0%だった。
今回の40学習デモ条件では、naive multi-viewによる平均性能改善は確認されず、single-sideが最高だった。
ただし、モデル固有の成功状態が存在し、multiが全状態で一様に劣るわけではなかった。

Phase Aでは、新規学習、rollout、checkpoint再選択を行わず、既存の3 seed × 100 held-out stateを
pairedに解析した。主な差は把持以前ではなく、peg/socket接触後に最終挿入へ移行できるかに現れた。
初期XY位置だけではside優位状態とmulti優位状態を明確に分離できなかった。初期姿勢は全stateで
identityに固定されていたため、姿勢差と成否の関係は今回のデータからは評価できなかった。

## 2. 実験条件と妥当性

| 項目 | 条件 |
|---|---|
| task | `sim_insertion_scripted` |
| dataset | 成功50 episode、生成seed 0 |
| camera | top / sideを同一episode・同一時刻で収録 |
| train split | episode 0–39 |
| dataset validation | episode 40–49 |
| model | single-top / single-side / multi |
| training seed | 0 / 1 / 2 |
| training | ACT、2,000 epoch、batch 8、lr `1e-5`、chunk 100、KL 10 |
| checkpoint selection | seed 1100の独立validation 50状態 |
| held-out test | seed 1200の独立test 100状態 |

### 2.1 入力の同一性

- 全9学習runが同じdataset archiveを使用した。
  - SHA-256: `e1513528847f2f565b1398369183bb27bde254490a46efd4107d51949d915893`
- 全runでtrain IDは0–39、dataset validation IDは40–49だった。
- 全9 testが同じtest manifestを使用した。
  - SHA-256: `d4462f7927d7beda491204f355ae30638e03408c6966c29ec353ac3f9a5dcd25`
- 全900 rolloutについて、rollout ID、初期状態、順序がtest manifestと完全一致した。
- 全testは`clean`条件で、camera corruptionはなかった。
- 全学習runの記録されたGit commitとsource hashは一致した。

### 2.2 デモとcamera

datasetの各HDF5には、同一の行動・関節状態に対応するtop画像とside画像が同時保存されている。
モデル間で異なるのは使用する画像streamであり、デモ軌道自体は同一である。

MuJoCo cameraはscene XML内で固定されている。

- top: `pos="0 0.6 0.8"`、tableを注視
- side: `pos="0 0 0.6"`、tableを注視

ただし、生成時scene XML hashとcamera extrinsicsそのものはdataset metadataへ埋め込まれていない。
次回のデータ生成では、このprovenanceも保存する必要がある。

## 3. Clean-50 held-out test結果

各seedは、学習・checkpoint選択に使っていない同一の100状態で評価した。

| モデル | seed 0 | seed 1 | seed 2 | 平均成功率 ± seed SD | 平均return |
|---|---:|---:|---:|---:|---:|
| single-top | 35% | 46% | 46% | 42.3 ± 6.4% | 364.2 |
| single-side | 41% | 65% | 44% | **50.0 ± 13.1%** | **380.5** |
| multi | 37% | 30% | 53% | 40.0 ± 11.8% | 357.6 |

### 3.1 解釈

- single-sideはsingle-topより平均7.7 percentage points高かった。
- single-sideはmultiより平均10.0 points高かった。
- multiはseed 2では53%で最良だったが、seed 1では30%に低下した。
- single-sideも41–65%とseed変動が大きい。
- 3 seedだけなので、一般的なside優位を断定する結果ではない。

現時点の適切な主張は次である。

> 40学習デモ条件ではmulti-viewの平均改善は確認されず、single-sideが最高だった。
> 一方、モデル固有の成功状態と大きなseed変動があり、視点補完性と融合安定性の診断が必要である。

## 4. Paired成功パターン

sideとmultiの成功・失敗を、同一training seed・同一held-out stateで比較した。

| パターン | seed 0 | seed 1 | seed 2 | 合計 |
|---|---:|---:|---:|---:|
| side成功・multi失敗 | 23 | 43 | 17 | **83** |
| multi成功・side失敗 | 19 | 8 | 26 | **53** |
| 両方成功 | 18 | 22 | 27 | **67** |
| 両方失敗 | 40 | 27 | 30 | **97** |

multiにしか成功できなかったpaired stateが53件あるため、multiはsideの単純な劣化版ではない。
ただしside成功・multi失敗が30件多く、平均ではmultiの補完効果より取りこぼしが大きかった。
この差は特にseed 1で大きく、seed 2では逆にmulti優位だった。

成果物:

- `results/act/clean50/paired_outcomes.csv`
- `results/act/clean50/paired_pattern_summary.csv`
- `results/act/clean50/paired_outcomes_with_geometry.csv`

## 5. 初期XY geometry診断

test manifestは14要素の
`[peg_xyz, peg_quaternion_wxyz, socket_xyz, socket_quaternion_wxyz]`として生成・消費されている。
manifestと全rollout JSONLの初期状態を完全一致で照合した上で、次を計算した。

- `relative_x = peg_x - socket_x`
- `relative_y = peg_y - socket_y`
- `relative_distance_xy`

### 5.1 Discordant群の比較

| 群 | 特徴 | 平均 | 中央値 | SD |
|---|---|---:|---:|---:|
| side成功・multi失敗 | relative_x | 0.2969 m | 0.2991 | 0.0425 |
|  | relative_y | 0.0033 m | 0.0111 | 0.0899 |
|  | distance_xy | 0.3103 m | 0.3074 | 0.0408 |
| multi成功・side失敗 | relative_x | 0.3050 m | 0.2998 | 0.0430 |
|  | relative_y | 0.0133 m | 0.0111 | 0.0847 |
|  | distance_xy | 0.3167 m | 0.3134 | 0.0422 |

multi優位群の平均は、side優位群よりrelative Xで約8.1 mm、relative Yで約9.9 mm、
XY距離で約6.5 mm大きかった。一方、群内SDは約41–90 mmであり、散布図でも両群は強く重なった。
したがって、今回の初期XY位置だけではside/multiの成否差を明確に説明できない。

図:

- `results/act/clean50/analysis/relative_xy_discordant.png`
- `results/act/clean50/analysis/relative_distance_xy_discordant.png`
- `results/act/clean50/analysis/workspace_positions_discordant.png`

## 6. 最高到達段階診断

環境の`InsertionTask.get_reward`に従い、highest rewardを次の段階として扱った。

| Reward | 意味 |
|---:|---|
| 0 | 両グリッパが各物体へ接触する前 |
| 1 | 両グリッパが各物体に接触 |
| 2 | 両物体を把持し、tableから離した |
| 3 | pegとsocketが接触し、tableには接触していない |
| 4 | pegがpinへ接触し、挿入成功 |

### 6.1 全300 paired stateの分布

| Reward | Side | Multi |
|---:|---:|---:|
| 0 | 3 | 5 |
| 1 | 0 | 5 |
| 2 | 11 | 11 |
| 3 | 136 | 159 |
| 4 | 150 | 120 |

差の中心はreward 0–2ではなく、reward 3から4への最終挿入移行にある。multiはsideより
reward 3が23件多く、挿入成功が30件少なかった。

### 6.2 Discordant成功群

`side_success_multi_fail` 83件におけるmultiの停止段階:

| Multi停止段階 | 件数 | 割合 |
|---:|---:|---:|
| 0 | 1 | 1.2% |
| 1 | 3 | 3.6% |
| 2 | 3 | 3.6% |
| 3 | **76** | **91.6%** |

`multi_success_side_fail` 53件におけるsideの停止段階:

| Side停止段階 | 件数 | 割合 |
|---:|---:|---:|
| 0 | 1 | 1.9% |
| 1 | 0 | 0.0% |
| 2 | 5 | 9.4% |
| 3 | **47** | **88.7%** |

両方向とも、失敗側の約9割はpeg/socket接触までは到達していた。視点差は物体への接近や把持より、
接触後の微細位置合わせ、保持安定性、押し込みに関係している可能性が高い。

### 6.3 Stage gap

`stage_gap = highest_reward_side - highest_reward_multi`とした。

- 全300件: 平均`+0.153`、中央値`0`
- sideが先: 94件
- 同段階: 150件
- multiが先: 56件

両方失敗の97件では、sideが先11件、同段階83件、multiが先3件だった。both-failの大部分は
同じ段階で停止しており、わずかにsideが先へ進む傾向があった。

### 6.4 Returnとの関係

| Highest reward | Side平均return | Multi平均return |
|---:|---:|---:|
| 0 | 0.00 | 0.00 |
| 1 | — | 35.80 |
| 2 | 178.91 | 140.09 |
| 3 | 350.84 | 357.23 |
| 4 | 429.75 | 406.28 |

reward 3では平均returnが近い。成功episodeではsideのreturnが高く、より早い挿入または成功状態を
長く維持した可能性がある。ただし、これは同じhighest reward内の記述的集計であり、因果的・pairedな
効率差としては解釈しない。

成果物:

- `results/act/clean50/analysis/paired_stage_summary.md`
- `results/act/clean50/analysis/paired_stage_statistics.csv`
- `results/act/clean50/analysis/paired_stage_contingency.csv`
- `results/act/clean50/analysis/paired_stage_gap.csv`
- `results/act/clean50/analysis/side_multi_stage_contingency.png`

## 7. 初期姿勢診断

quaternion順序は、MuJoCoおよび既存の`pyquaternion.Quaternion(...).elements`利用箇所から
scalar-first WXYZと確認した。相対回転は`inverse(socket) * peg`で計算し、yawは`[-pi, pi]`へwrap、
3D相対回転角は`[0, pi]`とした。

しかしtest manifestの全100状態で、pegとsocketのquaternionはともに`[1, 0, 0, 0]`だった。
したがって全300 paired rowで以下が一定だった。

- peg yaw = 0
- socket yaw = 0
- relative yaw = 0
- absolute relative yaw = 0
- relative quaternion = `[1, 0, 0, 0]`
- relative rotation angle = 0

姿勢quantileを作ることはできないため、値を捏造せず`constant_zero_orientation`という単一binとした。
この結果は「姿勢が成否へ影響しない」ことを示すものではなく、**今回の評価集合では姿勢効果を
識別できない**ことを示す。

### 7.1 接触から挿入への条件付き成功率

`P(reward=4 | highest_reward>=3)`を算出した。

| Seed | Side | Multi |
|---:|---:|---:|
| 0 | 41/96 = **42.7%** | 37/92 = **40.2%** |
| 1 | 65/97 = **67.0%** | 30/95 = **31.6%** |
| 2 | 44/93 = **47.3%** | 53/92 = **57.6%** |
| 全体 | 150/286 = **52.4%** | 120/279 = **43.0%** |

全体ではsideが9.4 points高いが、差は主にseed 1による。seed 2ではmultiが10.3 points高く、
方向はseed間で一貫していない。姿勢差が全て0なので、「姿勢差が大きいほどmultiが悪化する」という
仮説は今回のデータから検証できない。

成果物:

- `results/act/clean50/analysis/paired_orientation_summary.md`
- `results/act/clean50/analysis/paired_orientation_features.csv`
- `results/act/clean50/analysis/orientation_success_statistics.csv`
- `results/act/clean50/analysis/contact_to_insertion_statistics.csv`
- `results/act/clean50/analysis/orientation_vs_success.png`

## 8. 総合考察

### 8.1 Naive multi-viewは単純に認識へ失敗しているわけではない

multiの失敗の多くはreward 3まで到達している。したがって、top+side入力によって物体を見失い、
接近や把持そのものが崩壊したという説明だけでは不十分である。問題は接触後の微小な制御、両腕の
協調、または視覚特徴融合の不安定性にある可能性が高い。

### 8.2 Side優位はseedに依存する

seed 1ではsideの接触→挿入率が67.0%、multiが31.6%で大差がある。一方seed 2ではmultiが
57.6%、sideが47.3%で逆転する。multi-viewの平均低下を一般的効果として扱わず、学習seedに
対する融合の不安定性として調べる必要がある。

### 8.3 初期状態だけでは差を十分説明できない

初期XY位置分布は強く重なり、初期姿勢は全状態で同一だった。現時点では、初期状態の難しさより、
接触時点で生じた動的状態の差が重要である可能性が高い。

### 8.4 現在の解析限界

- 3 seedのみで、異なるデモ集合の不確実性は評価していない。
- highest rewardは最大到達段階であり、到達後の落下・再接触・振動を表さない。
- per-timestep reward、contact、object pose、actionはheld-out JSONLに保存されていない。
- 動画は各testの先頭5 rolloutのみで、全discordant stateを目視できない。
- 初期姿勢が固定されており、姿勢頑健性を評価できない。
- test結果は診断にのみ使用し、checkpointやハイパーパラメータの再選択には使わない。

## 9. 次に行う診断

既存動画では、次のpaired例を優先して比較する。

- side成功 / multi reward 3:
  - seed 0、state 3
  - seed 1、state 0
  - seed 1、state 3
- multi成功 / side reward 3:
  - seed 2、state 2
  - seed 2、state 3

動画では次を確認する。

1. peg/socket接触時の軸ずれと傾き。
2. 接触後の押し込み方向と速度。
3. 左右グリッパの保持位置・保持ずれ。
4. 接触後の振動、overshoot、反発。
5. 接触時刻とepisode残り時間。
6. multiで左右腕指令が不安定になっていないか。

詳細な失敗段階を全100 stateで分析するには、今後の評価でper-timestep object pose、contact pair、reward、
action、画像timestampを保存する必要がある。初期姿勢効果を検証する場合は、yawまたは3D回転を
事前定義して変化させた新しい評価manifestを、checkpoint選択と分離して設計する。

## 10. 結論

Clean-50ではsingle-sideが平均成功率50.0%で最良、single-topが42.3%、multiが40.0%だった。
paired診断から、主な差は接近・把持ではなく、peg/socket接触後の最終挿入にあることが分かった。
初期XY位置だけではdiscordant outcomeを説明できず、初期姿勢は固定されていたため評価不能だった。

したがって次の中心課題は、接触時点の動的状態と制御挙動を調べ、multi-view融合が最終位置合わせを
不安定化する条件を特定することである。同時に、5/10/25/40デモ条件でこの傾向がデータ量によって
変化するかを検証する。ただし追加実験は、現在のheld-out testを用いた診断結果でcheckpointを
再選択せず、事前に固定した仕様に基づいて実施する。
## 11. Dynamic diagnosis (reward 3 → 4)

既存動画が保存されているdiscordant 5 pairについて、reward 3到達後に相当する終盤挙動をpairedで比較した。詳細は`results/act/clean50/analysis/paired_dynamic_summary.md`を参照。

- 5件中4件では、失敗側のpegがsocket端への接触後に上向きへ偏向し、挿入軸を維持できなかった。
- 残る1件（seed 2/state 3のside）では、接触後にpegを落下させており、grasp instabilityが明瞭だった。
- 動画由来の接触近接proxyでは、失敗側も終了59--82 frame前には接触域へ到達していた。厳密なreward 3時刻は未保存だが、少なくとも対象5件ではlate contactより接触後の角度・保持不安定性が強い手掛かりだった。
- seed 1のmulti失敗2件は同じ角度ずれ傾向、seed 2のside失敗2件は角度ずれとpeg落下に分かれた。
- timestep reward/action/qpos/object pose/contactは保存されていないため、厳密なreward遷移時刻、action振動、接触後returnは判定不能とした。

注意: 保存動画は各seedのstate 0--4のみで、該当discordant pairは5件しかない。この内訳を全136 discordant pairの発生頻度として一般化してはならない。
