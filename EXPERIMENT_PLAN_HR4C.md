# 少数データ双腕挿入におけるACT視点構成・頑健性 実験計画書

初版: 2026-09-16

改訂日: 2026-09-22

対象: MuJoCo `sim_insertion_scripted`、HR4C実機候補

関連仕様: [EXPERIMENT_SPEC_CLEAN50.md](EXPERIMENT_SPEC_CLEAN50.md)

Clean-50成果物: `results/act/clean50`

Clean-50結果・Phase A診断: [EXPERIMENT_RESULTS_CLEAN50_PHASE_A.md](EXPERIMENT_RESULTS_CLEAN50_PHASE_A.md)

## 1. 研究の中心

本研究は、少数実演による双腕精密操作において、カメラ視点の追加が模倣学習方策の
性能と頑健性をいつ改善し、いつ悪化させるかを調べる。

> 少数データ条件では、複数視点は常に有効とは限らない。視点間の補完性、学習データ量、
> 遮蔽・欠損、および融合方法によって、性能とseed安定性が変化する。

シミュレーションでは現在の双腕peg/socket insertionを体系的に評価する。実機では、
構成が許せば左腕でsocketまたはfixtureを安定化し、右腕でpegを把持・位置合わせ・挿入する
`bimanual stabilized insertion`を採用する。実機で再学習する場合は厳密なsim-to-real transfer
ではなく、シミュレーションで得た知見のreal-world validationとして位置づける。

## 2. 研究質問

### RQ1: Data efficiency × View configuration

学習デモ数が5、10、25、40と変化するとき、`top`、`side`、`multi`（top+side）の
成功率、学習効率、seed間変動はどう変化するか。

### RQ2: View complementarity and robustness

どの初期状態・幾何条件で各視点が有効か。また、遮蔽、camera missing、camera pose shiftに
対して、multi-viewはsingle-viewよりcleanからの性能低下を抑えられるか。

### RQ3: Robust multi-view learning

視点欠損を考慮した学習・融合方法により、naive multi-viewのclean性能、seed不安定性、
視点欠損耐性を改善できるか。

実機検証は独立した広いRQではなく、上記の主要傾向が双腕実機のstabilized insertionでも
再現するかを確認する外的妥当性検証とする。

## 3. 既存Clean-50実験

### 3.1 設計

| 項目 | 条件 |
|---|---|
| task | `sim_insertion_scripted` |
| データ | 成功50 episode、生成seed 0、top/side同時収録 |
| split | episode 0–39=train、40–49=dataset validation |
| view | single-top / single-side / multi |
| training seed | 0 / 1 / 2 |
| 学習 | ACT、2,000 epoch、batch 8、lr `1e-5`、chunk 100、KL 10 |
| checkpoint選択 | 独立validation 50状態、seed 1100 |
| held-out test | 独立test 100状態、seed 1200 |

全モデルは同一のデモarchive、episode split、validation/test manifestを使用した。test 9 runの
初期状態列がmanifestと完全一致し、全rolloutがclean・camera corruptionなしであることを監査済み
である。カメラはMuJoCo XML内の固定top/side cameraである。

### 3.2 結果

| モデル | seed 0 | seed 1 | seed 2 | 平均成功率 ± seed SD | 平均return |
|---|---:|---:|---:|---:|---:|
| single-top | 35% | 46% | 46% | 42.3 ± 6.4% | 364.2 |
| single-side | 41% | 65% | 44% | **50.0 ± 13.1%** | **380.5** |
| multi | 37% | 30% | 53% | 40.0 ± 11.8% | 357.6 |

Clean-50ではsingle-sideが最高で、naive multi-viewの改善は確認できなかった。ただしmultiには
固有の成功状態があり、単純な包含関係ではない。

| paired pattern | 300状態中の件数 |
|---|---:|
| side成功・multi失敗 | 83 |
| multi成功・side失敗 | 53 |
| sideだけ成功 | 51 |
| multiだけ成功 | 22 |
| 全モデル成功 | 37 |
| 全モデル失敗 | 70 |

状態別成果物は `results/act/clean50/paired_outcomes.csv`、
`paired_pattern_summary.csv`、`scripts/analyze_clean50_paired.py` に固定する。

現時点の結論は「multi-viewは弱い」ではなく、次のように限定する。

> 40学習デモ条件ではmulti-viewの平均改善は確認されず、single-sideが最高だった。
> 一方でモデル固有の成功状態が存在し、視点補完性と融合失敗の診断が必要である。

## 4. Phase A: Paired失敗診断

新規の大規模学習より先に、既存300 paired rolloutを分析する。

- pegとsocketの初期相対距離、x/y方向の相対変位
- peg/socketのworkspace内位置
- top/side画像平面上の投影距離（取得可能な場合）
- 各モデルのepisode return、highest reward、成功パターン
- 失敗段階（接近、把持維持、位置合わせ、接触、挿入）
- side優位・multi優位状態の代表動画

初期状態特徴付きpaired CSV、成功パターン別位置分布図、モデル別成功領域を出力する。この解析から
RQ2の事前仮説とRobustness条件を決める。test結果を使って既存checkpointを再選択しない。

## 5. Phase B: データ量実験（RQ1）

| 因子 | 条件 |
|---|---|
| view | top / side / multi |
| 学習デモ数 | 5 / 10 / 25 / 40 |
| training seed | 0 / 1 / 2 |
| clean test | Clean-50と同じ100状態 |
| checkpoint選択 | Clean-50と同じvalidation 50状態 |
| 主指標 | test成功率 |
| 副指標 | return、seed SD、学習曲線、成功パターン |

デモ集合はnestedに固定する。

- 5 demos: episode 0–4
- 10 demos: episode 0–9
- 25 demos: episode 0–24
- 40 demos: episode 0–39（完了済み）
- dataset validation: episode 40–49（全条件共通）

新規runは27学習＋27 testである。まずseed 0の9条件を実行し、成果物と学習挙動を監査した後に
seed 1・2の18条件を実行する。

### 学習budgetの統制

同じepoch数だけでは、デモ数によって総optimizer step数が変わり、データ量と最適化量が交絡する。
主解析では総optimizer step数を40-demo条件に合わせ、小規模集合を反復samplingする。同一epoch条件は
必要に応じて補助解析とする。全条件でoptimizer step数、各episodeの露出回数、parameter数、
camera順、推論時間、checkpoint選択規則を固定・記録する。

## 6. Phase C: 頑健性評価（RQ2）

実験数を抑えるため、10-demoと40-demoのみを対象にする。Phase Bで選択済みのcheckpointを
再学習せず評価する。

| 条件 | 内容 |
|---|---|
| clean | 基準条件 |
| occlusion 25% / 50% | topまたはsideの一定面積を遮蔽 |
| camera missing | 一方のviewを欠損 |
| camera pose shift | 位置・姿勢を事前定義量だけ変更 |

multiではtopとsideを個別に外乱させ、single-viewでは使用viewを外乱させる。ランダム矩形だけでなく、
peg、socket、左腕、右腕のどれが遮蔽されたかを記録する。主要指標は
`success_corrupted - success_clean` とし、同一初期状態・同一外乱seedを共有する。

## 7. Phase D: 改善法（RQ3）

Baselineはsingle-top、single-side、naive multi、training-time camera/modality dropoutとする。
単純なsensor/modality dropout自体は既存研究があるため、新規手法とは主張しない。

- Akinola et al., *Learning Precise 3D Manipulation from Multiple Uncalibrated Cameras*, 2020

  <https://arxiv.org/abs/2002.09107>
- Cheng et al., *Robust Bimanual Vision-Language-Action Models via Embarrassingly Simple Modality Masking*, 2026

  <https://arxiv.org/abs/2608.22419>

仮称ViewDrop-ACTは単純dropoutで終わらせず、paired診断が支持する最小構成を採用する。

- view availability token
- view別encoderとgated fusion
- view信頼度の推定と重み付け
- feature scaleの正規化
- camera missingと局所遮蔽を混合したtraining corruption

主張は「dropoutの発明」ではなく、少数データ双腕挿入におけるnaive multi-viewの失敗を診断し、
信頼度を考慮した融合で改善することに置く。本評価は10-demoと40-demoに限定する。

## 8. Phase E: HR4C実機検証

### 課題選択

納入機が双腕構成で、左右腕の同期API、安全機能、可動域が確認できる場合、実機課題は
`bimanual stabilized insertion`とする。

- 左腕: socketまたはfixtureを把持・安定化
- 右腕: pegを把持し、接近、6DoF位置合わせ、挿入
- top/side camera: 同一実演時に同期収録

双腕を利用できない場合は、socketを治具で固定した単腕insertionへ縮退する。旧計画の
pick-and-placeは接続確認用候補に留め、研究本実験は可能な限りinsertionへ合わせる。

本評価は20～30成功デモ程度でsingle-side、naive multi、改善版multiを比較する。cleanに加え、
左腕による自然遮蔽、右腕による自然遮蔽、片側camera missingを評価する。

### 実機構成確定前の必須確認

1. 単腕／双腕、グリッパ、頭部cameraの納入構成。
2. ROS version、Python SDK、topic/service/action、URDF。
3. 関節順序、単位、可動範囲、速度・トルク上限、command timeout。
4. 左右腕の指令同期とstate timestamp。
5. 非常停止、保護停止、衝突検知、手動教示の正式手順。
6. camera型番、fps、同期方法、intrinsics/extrinsics。
7. camera/base/fixture poseを固定・記録する方法。
8. peg/socketの公差、挿入力、最大許容力、成功判定。

公開仕様と納入構成の確認が終わるまで、双腕利用を確定事項として扱わない。

## 9. 統計解析

- seed別成功率とseed平均±SDを必ず併記する。
- 300 rolloutを独立な300学習反復として扱わない。
- 同一seed・初期状態の成功差はMcNemar検定またはpaired bootstrapで評価する。
- 成功率差はp値だけでなくpercentage-point差と95%信頼区間を示す。
- 複数の外乱・pairを検定する場合は多重比較を補正する。
- checkpoint選択後のheld-out testを再選択に使用しない。

## 10. 再現性・妥当性要件

各runについてdataset・manifest・sourceのSHA-256、Git commitとdirty diff、episode ID、camera名・順序・
intrinsics/extrinsics・scene XML hash、optimizer step数、sampling回数、training seed、parameter数、
推論時間、validation全候補、rollout別初期状態・成功・return・失敗段階・外乱情報を保存する。

Clean-50では生成時scene XML hashとcamera extrinsicsをarchive metadataへ埋め込んでいない。
次回データ生成からこのprovenanceを追加する。

Phase B以降のtest rolloutでは`--save_timeseries`を有効にし、既存JSONL・動画とは別に、各stepの
reward、policy action、commanded qpos、pre/post qpos、gripper state、peg/socket pose、raw contact pairを
圧縮NPZへ保存する。形式とvalidatorは[ROLLOUT_TIMESERIES_LOGGING.md](ROLLOUT_TIMESERIES_LOGGING.md)に従う。
各本評価の前に同一checkpoint・manifestの1--3 rolloutでlogging ON/OFFのaction・reward・success一致を
確認する。

## 11. 実機安全ゲート

1. 停止状態でcamera、state、action timestampを監査する。
2. 無負荷・低速で各腕の単関節指令とwatchdogを検証する。
3. workspace、関節速度、加速度、挿入力を制限して固定軌道を再生する。
4. 左右腕の同期停止と片腕異常時の両腕停止を確認する。
5. 人がworkspace内にいない状態で1 episodeだけ方策を実行する。
6. 失敗時停止を確認してから反復評価へ進む。

メーカーの安全マニュアルとリスクアセスメントを最優先とし、「協働ロボット」という名称だけを
根拠に人との接触を許容しない。

## 12. 実施順序と停止条件

| Phase | 内容 | 完了条件 |
|---|---|---|
| A | Clean-50 paired診断 | 幾何特徴、失敗段階、代表動画を整理 |
| B0 | 5/10/25-demo、seed 0 | 9条件の設定・成果物・学習budgetを監査 |
| B1 | seed 1・2追加 | 全データ量×viewの3 seed完了 |
| C | 10/40-demo robustness | clean低下量とpaired結果を集計 |
| D0 | modality dropout baseline | naive multiとの差を確認 |
| D1 | 改善版multi | cleanとrobustnessの改善を確認 |
| E0 | 実機仕様・安全確認 | 双腕／単腕課題を確定 |
| E1 | 10-demo接続pilot | 収録から低速実行までend-to-end確認 |
| E2 | 実機本評価 | side / multi / 改善版multiを比較 |

各段階の前に入力hash、評価状態、camera設定、学習budget、動画を監査する。seed 0で床効果、
学習崩壊、入力不一致が見つかった場合はseed追加を止める。

## 13. 主張できる範囲

- Phase B完了: データ量と視点構成の関係をシミュレーション内で主張できる。
- Phase C完了: 視点補完性と外乱別の性能低下を主張できる。
- Phase D完了: baselineとの差が複数seedで再現した場合のみ改善法の有効性を主張する。
- 実機pilotのみ: end-to-end実装可能性までとし、性能優位は主張しない。
- 実機本評価完了: 限定された装置・課題条件でシミュレーション知見の再現性を主張できる。
- 実機で再学習した場合はsim-to-real transferとは呼ばない。

複数視点が勝たない場合も、追加視点が有効になるデータ量・幾何条件・外乱と、失敗する融合条件を
明らかにできれば研究成果となる。

## 14. 直近の作業

1. Phase B seed 0の最初のcheckpointでlogging ON/OFF smoke testを通す。
2. 5/10/25-demo用に固定optimizer step数とnested episode集合を仕様化する。
3. seed 0の9条件だけを投入し、成果物と詳細時系列ログを監査する。
4. 同時にHR4Cの双腕構成、左右同期API、安全資料を確認する。
5. seed 0の監査後に、残り18条件とRobustnessの投入可否を判断する。
