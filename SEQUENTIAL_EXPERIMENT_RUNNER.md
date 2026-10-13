# Phase B の逐次実験と Windows ローカル退避

`EXPERIMENT_PLAN_REVISED.md` 第5節のうち、既存 Phase B の学習・validation 選択・探索用
test・成果物退避を順番に進める CLI。実行計画は
`configs/sequential_phase_b_windows.json` に固定した。退避先は Windows PC の
`C:\Users\chibi\Research\result`。HPC からこのドライブはマウントされていないため、
Windows 側 CLI が SSH 経由で成果物を引き取る。

## Windows PC から起動

Windows に Python 3 と OpenSSH クライアントを用意し、PowerShell で実行する。
`your-ssh-alias` は VS Code Remote SSH で使っているホスト名または SSH 設定の別名に置き換える。
まず `ssh -o BatchMode=yes your-ssh-alias true` がパスワード入力なしで成功することを確認する。
PowerShell で次のように CLI を Windows PC にコピーして起動する。
保存先フォルダは CLI が作成する。

## 継続運転の設定（推奨）

2026-10-03 に、実行中の runner が返す `busy: true, next: null` を「完了」と
誤判定して controller が終了する不具合を修正した。既に Windows へコピーした
古い controller は停止し、下記で最新版を取得する。

Windows PowerShell で一度実行すると、常駐タスクを登録して直ちに開始する。
`your-ssh-alias` は実際の SSH 設定の別名へ置き換える。

```powershell
$ExperimentHost = 'your-ssh-alias'
scp "${ExperimentHost}:/home/home0/y263357/Prompt2Motion/scripts/windows_experiment_controller.py" .\windows_experiment_controller.py
scp "${ExperimentHost}:/home/home0/y263357/Prompt2Motion/scripts/install_windows_experiment_task.ps1" .\install_windows_experiment_task.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_windows_experiment_task.ps1 -Remote $ExperimentHost
```

タスク名は `Prompt2Motion-Experiments`。通常は管理者権限不要で、現在のユーザーが
ログインしている間に動作する。端末を閉じても継続し、次回ログイン時にも再開する。
controller 実行中は Windows の自動スリープを抑止する。手動スリープ・電源断・
ログアウト中は進行しない。Windows 退避先の空き容量も継続運転に必要。

実行中の PBS ジョブを待ち、学習→選定→探索用 test→Windows 退避→次 run の順で
120秒ごとに進める。SSH接続障害は再試行する。実験失敗・ハッシュ不一致・容量不足は
停止して記録し、checkpoint を保持する。失敗した学習を自動的に何度も回し直さない。
固定計画の全 run を検証・退避した時点で終了する。

ログは `C:\Users\chibi\Research\result\.controller\controller.log`。
状態確認・停止・再開は次のとおり。SSH設定修正などの後も再開できる。

```powershell
Get-ScheduledTaskInfo -TaskName 'Prompt2Motion-Experiments'
Get-Content 'C:\Users\chibi\Research\result\.controller\controller.log' -Tail 30
Stop-ScheduledTask -TaskName 'Prompt2Motion-Experiments'
Start-ScheduledTask -TaskName 'Prompt2Motion-Experiments'
```

タスクを解除する場合は
`Unregister-ScheduledTask -TaskName 'Prompt2Motion-Experiments' -Confirm:$false`。
失敗したPBS段階は成果物を確認してからHPC runnerの `--retry-failed` で明示的に再試行する。

## 手動で起動する場合

```powershell
$ExperimentHost = 'your-ssh-alias'
scp "${ExperimentHost}:/home/home0/y263357/Prompt2Motion/scripts/windows_experiment_controller.py" .\windows_experiment_controller.py
py -3 .\windows_experiment_controller.py --remote $ExperimentHost --loop
```

`--destination` の既定値は `C:\Users\chibi\Research\result`。
別の場所に保存する場合のみ `--destination 'D:\Research\result'` を指定する。
`--loop` では 120 秒ごとに PBS 状態を確認し、順番に **1件だけ** ジョブを投入する。
ジョブが終了したら結果を Windows に転送し、サイズ・SHA-256 を全件検査する。
検査後に HPC へ受領を返し、非選択 checkpoint だけを HPC から削除して次へ進む。
選択 checkpoint は HPC に残す。SSH切断は再接続し、検査・空き容量・ジョブが失敗した場合は停止し、
HPC の元データを保持する。途中までの Windows 転送は再試行時に破棄する。
PC が停止・スリープした間は進まず、同じ CLI を
再実行すると再開する。安定して継続させるには Windows PC と SSH 接続を維持する。

計画の先頭は `phase_b/40demos/single_top/train_seed0`。受領済みrunは再実行時に
スキップする。2026-10-03 時点で40-demo seed 0 top/side、10-demo seed 1 top/sideの
退避は完了済み。新規学習はhome空き容量がcheckpoint見積1.5 GBと余裕1 GiBを
満たす場合にだけ開始する。
Drive に以前転送したコピーは今回の受領・削除判断に使わない。

## HPC で状態を見る

```bash
python3 scripts/sequential_experiment_runner.py status \
  --plan configs/sequential_phase_b_windows.json
```

`next` が `offload` の間は Windows 側の検査待ち。`train`/`select`/`test` なら次の
PBS 段階を投入できる。`active_jobs` が true の場合は既存ジョブの終了待ち。
Windows PC からの `--loop` が停止した場合は、同じコマンドで再開する。

`scripts/windows_offload.py prepare` は完了済み run の全ファイルを inventory し、
manifest を固定する。Windows CLI は SSH の tar ストリームを一時フォルダへ受け、
各ファイルをハッシュ照合してから確定フォルダへ原子的に切り替える。
`commit` は Windows からの受領内容と manifest を照合してから HPC 側 receipt を書く。
再開時には保存済み Windows ファイルを再ハッシュする。HPC 側の runner は receipt が
manifest と一致する場合だけ次の run に進む。

## 計画の範囲

2026-10-06に先行11 runの完了・Windows退避受領を確認し、欠けていた
`phase_b/40demos/multi/train_seed0` を同じ計画の末尾に追加した。計画完了で
Windows controllerが終了していた場合、PC側で
`Start-ScheduledTask -TaskName 'Prompt2Motion-Experiments'` を実行すると追加分を処理する。
20 runの探索結果集計は `results/act/phase_b/analysis/summary_20261006/README.md` に保存した。
2026-10-07に研究計画を再確認し、固定subsetのseed感度を測る補助実験として
25/40デモ×top/side/multiのseed 2（6 run）を末尾に登録した。10デモの床効果を踏まえ
5デモは追加しない。各runはWindows退避受領・容量判定を満たしてから投入される。
独立subsetを要求する改訂計画のGate 2完了とは区別する。再確認記録は
`reports/RESEARCH_REVIEW_20261007.md`。選択checkpointは引き続き保持する。

現在の計画は既存の 40 デモ seed 0 の top/side を先に退避し、10・25・40 デモの
top/side/multi seed 1 を順番に回す。seed 0 の 10/25 デモには削除済み checkpoint があり、
完全な run 退避の対象に含めていない。runner が使うデモ subset は既存コードの固定
`0..N-1` で、seed 1200 の test は探索用。改訂計画の独立 subset、工程別診断、
頑健性評価、未閲覧の確認用 test、Phase C/D、HR4C 実機はこの CLI の対象外。
これらを seed 1200 の結果で代用しない。

HPC 上に直接マウントされた退避先を使う場合は、別計画で `offload_backend: "local"` と
`offload_destination` を設定できる。Windows の `C:` パスは HPC の
`offload_destination` に設定しない。

## ファイル読取・PBS接続障害への対応（2026-10-13）

読取時の `Operation not permitted`、I/Oエラー、stale handle、タイムアウトは
同じファイルで最大3回、1秒・2秒間隔で再試行する。PBS照会も最大3回、
1回15秒で打ち切る。これは一時的な障害への対応であり、ストレージの根本原因を
修復するものではない。通常の権限不足、ファイル欠落、JSON不正、ハッシュ不一致は
自動的に無視しない。

読取・PBS照会が再試行後も失敗した場合、変更前のHPC runnerは
`[WAIT_INFRA]` と終了コード75を返す。最新版Windows controllerの `--loop` は
120秒後に状態確認から再開する。PBSに接続できない状態を「ジョブなし」や「完了」と
扱わない。`qsub` 開始後の失敗は自動再試行しない。ジョブの投入状況が不明な場合に
二重投入を避けるため、状態を確認してから明示的に再開する。

この対応をWindows側で有効にするには、常駐タスクを停止し、最新版の
`windows_experiment_controller.py` を同じ保存場所にコピーして再開する必要がある。
既にWindowsへコピー済みのファイルはHPC側の編集だけでは更新されない。

checkpoint選定・testのPBSスクリプトは、元のmanifestに記録されたSHA-256で
重みをジョブの作業領域にコピーする。コピー後にディスクから再読込して照合し、
一致したものだけを確定する。失敗した一時コピーは削除し、元の重みと既存の
検証済みコピーを保持する。保存先にはファイルサイズに加えて1 GiBの余裕を要求する。
選定・test開始にもhomeの1 GiBの余裕が必要。容量不足は停止し、退避してから再開する。

同じ方式をHR4C等の単独checkpointに使う場合は、保存時のstatus/manifestに
記録されたSHA-256と、明示した保存先を指定する。

```bash
python3 scripts/experiment_io.py /path/to/policy_best.ckpt \
  /path/to/job-work/policy_best.ckpt --sha256 RECORDED_SHA256
```

`RECORDED_SHA256` は保存時に記録した64桁のSHA-256に置き換える。
これは実行用コピーであり、永続バックアップの代わりにはならない。
`/tmp` はそのノードでのみ使う一時領域で、Windows controllerの検証済み退避先を
Phase Bの永続保存先とする。HR4C等の計画外の重みはこのcontrollerでは退避されないため、
別途永続保存先へ検証済みコピーを作成する。退避コピーを復元するときも記録された
SHA-256で照合する。権限を緩める変更や、異なるcheckpointへの自動置換は行わない。
