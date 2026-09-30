# Prompt2Motion 実験実行計画

更新日: 2026-09-29

## 原則

- 当初の順序で Phase B0 → Phase B → Phase C → Phase D → Phase E と進める。
- 固定validation/test manifest、10,000 optimizer-step、seed 0/1/2を変更しない。
- 既存成果物を削除・上書きしない。未完了成果物と復旧成果物は別namespaceに置く。
- filesystemには常に2 GiBを残す。実行中、queue待ち、依存hold中のPBSジョブも容量予約へ含める。
- gate監査が通るまで後続phaseを投入しない。

## 実行順序

1. **Phase B0（seed 0）**
   - 25-demo top/side/multiのheld-out testを完了する。
   - Google Drive退避物から5/10-demoの6 runを`phase_b_rebuild`へ1 runずつ復帰する。
   - 各runで決定論的checkpoint選定後、固定100状態のheld-out testを行う。
   - 40-demo top/side/multiを別namespaceで再構築する。これはB0と並行可能。
   - 5/10/25-demoの9 runを監査し、immutable B0 gateを作る。

2. **Phase B（seed 1/2）**
   - B0 gate後、5/10/25/40-demo × top/side/multi × seed 1/2を学習する。
   - 各runを決定論的選定→held-out testまで進め、全36 runのPhase B gateを作る。

3. **Phase C（naive baseline robustness）**
   - 10/40-demoの全view・全seedについて、同一100状態でclean、25/50% occlusion、camera missing、
     camera pose shiftをpaired評価する。
   - top/side外乱を個別に記録し、Phase C gateを作る。

4. **Phase D（robust training）**
   - modality dropout、ViewDrop-ACTを10/40-demo × seed 0/1/2で学習する。
   - Phase Cと同じ条件・状態で評価し、paired bootstrap CIとexact McNemarを集計する。
   - Phase D gateを作る。

5. **Phase E（HR4C移行）**
   - mock、ACT互換収録、readiness auditまでは継続する。
   - 納入構成、SDK、安全資料、E-stop試験が確定するまで実機通信はロックする。

## 2026-09-28時点

- 25-demo seed 0のtop/side/multi held-out testは完了。
- 5-demo single_top seed 0は復帰・決定論的選定済みで、held-out test（105225）を実行中。
- 5-demo single_side seed 0は復帰済み。決定論的選定（105227）と依存test（105228）を投入済み。
- 未開始だった40-demo single_top系列（105012/105019/105020）は、B0を優先して容量を確保するため
  取り下げた。成果物は未作成であり、容量確保後に同じ設定で再投入する。
- 旧探索run約486 MiBはSHA検証済みの分割アーカイブとしてDriveへ退避し、ホームから外した。
  詳細は`results/act/OFFLOAD_RECEIPT_20260929.md`に記録した。
- 25-demoの非選択checkpoint 9本（約3 GiB）もDriveへ退避し、選択済み3本はローカルに保持した。
- 5-demo multiと10-demo single_topを復帰し、選定（105235/105237）と依存test（105236/105238）を投入済み。
- 5-demo single_topは選定・test完了後、非選択3 checkpointを検証済みDrive原本へ退避した。
- 10-demo single_sideを復帰し、選定（105317）と依存test（105318）を投入済み。
- 選定済みの5-demo side/multiと10-demo top/sideは、非選択checkpointをDrive-backed receipt付きで退避した。
- 10-demo multiを復帰し、選定（106193）と依存test（106194）を投入済み。
- holdしていた10-demo top/side testは106191/106192として再投入した。
- 40-demo single_top seed 0の再学習（106195）、選定（106196）、test（106197）を依存投入した。
- B0完了監査（106198）を10-demo test群の成功依存で投入した。
- 40-demo single_side seed 0の再学習（106199）と選定（106200）を投入した。testは容量gate解除後に投入する。
- 次のcheckpoint復帰・学習は2 GiB reserveを下回るため、上記ジョブ完了後に再判定する。
- Phase B–D完走には既存成果物を保持したまま追加約65 GBが必要。
- PBSの`afterok` system holdが、親ジョブのExit status 0確定後も自動解除されない事象を確認した。
  完了済み選定105227/105235に依存して停止していたtest 105228/105236は、未実行・評価成果物なしを
  確認して取り下げ、`phase_b_rebuild`を明示した独立test 105449/105450として再投入した。
- 10-demo single_top選定105237は実行中、single_side選定105317はqueue待ち。これらの依存test
  105238/105318も同じhold事象が起きる可能性があるため、親のExit status 0と成果物を確認後に
  未実行なら同じ手順で独立再投入する。
- pipeline plannerが、退避済みcheckpointを持つ完了runを新規学習候補と誤判定しないよう、
  `completed canonical run requires checkpoint restore`としてブロックする防護を追加した。
- 2026-09-29のローカル検証は30 tests成功（2件は任意依存`h5py`なしでskip）、Python compile、
  全shell/PBSの`bash -n`、`git diff --check`が成功した。
