# EE_FL プロジェクト 引き継ぎサマリー（更新版 v3）

> 次回チャットの冒頭にこのファイルを貼り付け／再アップロードすれば、これまでの経緯を引き継げます。
> **v3 の主な更新**: 研究方針を C-1（正規化・スケール不変性の動力学）に事実上寄せ、C-1 の実験計画を E0〜E3 に具体化。**E0（M1+M2）は完了**。文献ポジショニングを再調査し、看板の立て方と査読者の反論への備えを整理。§8 の非対称記述の誤りを訂正。

---

## 0. この版で必ず押さえるべき要点（忙しい人向け）

- **確定した現象**: スケール不変 conv 重みのノルムは、中央集権(K=1)では平衡するが、FedAvg(K=5)では**ほぼIIDでも毎ラウンド約1.78倍で幾何級数的に発散**する。E0 実験（無対処・IID・厳密条件）で再現・定量化済み。
- **重要な誤解の解消**: この**ノルム発散はそれ自体では精度を落とさない**。SI conv はスケール不変なので、‖w‖ が 9億まで膨れても出力＝損失＝精度は不変。過去に精度が 10% に崩壊したのは、**非スケール不変**な GN affine（γ,β）が暴走していたためで、`affine=False` で除去済み。
- **したがって研究の看板は「暴走＝精度崩壊」に置いてはいけない**。本当の害は潜在的：(i) 有効学習率 η/‖w‖² の崩壊（バックボーンの早期凍結）、(ii) float32 overflow による長期学習の破綻。
- **新規性の核**: 「ノルムが増える」は集中学習で既知（Arora, Van Laarhoven, AdamP, Kosson 等）。「Du を反証」も単体では弱い。**空いている交差点＝(1) FedAvg の平均化が平衡を壊す（IIDでも／不均一性ではない）、(2) multi-exit の深度・Exit数依存の非対称、(3) どの対処が multi-exit 精度を保つか**。
- **次の一手**: E1（同期頻度 τ を振り、averaging が原因だと分離）。その前に E0 の残タスク3点（seed分散・lr非依存・handoff整合の明記）。

---

## 1. プロジェクト概要

- **テーマ**: Early-Exit ネットワーク × Federated Learning（EE_FL / FedAvg）
- **モデル**: `eenet32`（6n+2 型 ResNet32 ベース、BasicBlock、中間Exit3個＋最終Exit1個＝計4Exit）
- **データ**: CIFAR10（10クラス）、IID設定（Dirichlet α = 1e9〜1e10）
- **集約**: FedAvg（`server.py`）
- **損失**: `loss_v1`（全Exitの CrossEntropy を等重み平均）
- **当面のマイルストーン**: 11月の情報処理学会（IPSJ）研究会（SIG未定）＋修士論文。
- **研究方針（更新）**: 文献調査の結果、**案2: C-1（正規化・スケール不変性の学習動力学）を主軸**に据える方向で固まりつつある（§8）。案1（A-3→A-1 セキュリティ系）は修論の応用の山場として温存し、**C-1 →（A-3）→ A-1** の縦串にする。**教授と重心を最終確認する段階**。
- **現在地**: 学習不安定性は根治済み。C-1 の中心実験 **E0 が完了**し、E1 に進む直前。

---

## 2. これまでのデバッグ経緯（時系列）※v2から変更なし

### 症状①：損失が数百〜数千に爆発 → 全Exitが10%（＝ランダム）
- **対策**: `local.py` の `train()` に勾配クリッピングを追加 `clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **結果**: 急性爆発は止まったが、多クライアントでは依然10%へ崩壊。

### 症状②：多クライアントでのみ崩壊。`num_clients=1`（中央集権）は正常
- → **FL特有の問題**と確定。

### 症状③：BatchNorm の平均化が原因と特定
- `bn_axis_check` で BN の running 統計をクライアント平均すると壊れることを実証。
- **対策**: 全 `BatchNorm2d` を `GroupNorm` に置換 → 序盤は精度上昇。

### 症状④：GN化後、後半で損失が緩やかに幾何級数的増加（発散）
- lr を100倍振ってもほぼ同じラウンドで発散（**lr非依存**）。momentum=0.5で即爆発、0.9で安定。
- 5クライアントで ‖w‖ が毎ラウンド約1.78倍の指数増大、1クライアントでは平衡（≈57）に張り付く。
- **根本原因**: GN直前の conv 重みはスケール不変。勾配は重みと直交し、有効学習率 ∝ η/‖w‖²。中央集権では膨張項と収縮項(WD)が釣り合い ‖w‖ が平衡するが、**FedAvgの平均化がこの釣り合いを壊す**。

### 症状④の対処①：スケール不変convの集約後ノルム再射影
- `norm_tools.py` を新規作成。`find_scale_invariant_conv_weights()`（eenet32で36個検出）＋ `reproject_model_()` ＋ 群別診断 `norm_report()`。
- **結果**: `SI_conv` は平坦化したが `total` は依然指数増大 → **暴走源が別に残存**と判明。

### 症状④の対処②：GN affine（γ, β）が残りの暴走源
- 群別ログで `GN_affine` が約1.78倍/ラウンドで暴走、`total` とほぼ一致。
- **原因**: 中間ブロックGNの (γ,β) も「下流GNが吸収」でスケール不変。conv とまったく同じ機構で暴走。
- **対策**: `convert_bn_to_gn()` に `affine=False`（γ,β 自体を除去）。
- **結果**: `GN_affine`＝0、`total` 平坦化、損失単調減少、Exit4 最良（健全パターン復帰）。→ **根治確認**。

---

## 3. 確定した根本原因（最終版）

**スケール不変性 × FedAvg平均化による重みノルムの平衡破壊。スケール不変パラメータは2種類あった**：(1) GN直前の bias無し conv 重み、(2) 下流に別のGNが続く GN の affine (γ,β)。
中央集権では膨張項 η²‖∇L‖² と収縮項 −2ηλ‖w‖²(WD) が釣り合い平衡。FedAvgの平均化がこれを壊し指数増大。対処は conv 再射影＋GN affine 除去の2段階が必要だった。

> **v3補足（重要）**: (1)(2) の暴走は**それ自体は損失不変**（＝精度に無害）。過去の 10% 崩壊は、正確には **Exit分岐の終端GN affine のような「非スケール不変」な暴走**が損失を壊したケースが混在していた可能性が高い。純粋なスケール不変暴走（現在の `affine=False` 構成）では、後述 E0 の通り**精度は落ちない**。この区別が C-1 の主張の立て方を左右する（§9）。

---

## 4. 現在の config・実装（要点）

**通常構成（対処あり・本来の学習用）**
```yaml
model_name: eenet32
dataset: CIFAR10
num_ee: 4
loss_func: v1
optim_name: SGD   # lr=0.1, momentum=0.9, weight_decay=1e-4
FL:
  num_clients: 5
  cohort: 5
  global_rounds: 100
  dirichlet: 1e10   # ほぼIID
```

- `convert_bn_to_gn()` は `affine=False`（GN affine除去）。
- `norm_tools.py`：SI conv 検出・基準ノルム記録・再射影・群別診断。
- `server.py::federated_learning()` は `si_param_names`/`ref_norms` を渡すと集約後に conv 再射影。**両方 None で素のFedAvg**（＝E0/E1 の無対処条件）。

**E0/E1（診断実験）の条件**
- 再射影 **OFF**（`federated_learning(client_updates, global_model)` で呼ぶ）。
- clip **OFF**（`clip_grad_norm_` をコメントアウト。有効学習率測定を汚さないため）。
- `norm_report()` を毎ラウンド回収 → `norm_report.csv`（生ログ, `ite`列）と `norm_report_mean.csv`（seed平均）。
- 出力パス: `results/E0/M1/K={num_clients}/`。

---

## 5. C-1 実験計画（E0〜E3）※これが現在のロードマップ

### 目標（一文）
「スケール不変重みのノルム平衡が FedAvg では破れる。これは (i) IIDでも起きる、(ii) データ不均一ではなく“ローカル学習後に平均する”操作の帰結、(iii) multi-exit では深度・Exit数に依存して非対称に強まる。そしてどの対処が multi-exit 精度を最もよく保つか」を定量で示す。位置づけは **Du の「FLではノルムは初期値近傍」への反例**、かつ**集中の回転平衡（Kosson ら ICLR2024）が崩れる領域の提示**。

### 共通固定
`eenet32`/`num_ee=4`/`distribution=pareto`/`loss_func=v1`、`batch_size=64`、`val_ratio=0.2`。SGD `lr=0.1`, `momentum=0.9`, `weight_decay=1e-4`。ローカルエポック `epochs(E)=1`、`ite_num=3`（seed 0,1,2）、`cohort=num_clients`（フル参加）、`dirichlet=1e10`（IID）。測定ランは再射影OFF・clip OFF。

### 測定量（各ラウンド、集約直後のグローバルモデルで記録）
- **M1** 群別ノルム＝`norm_report()`（SI_conv/GN_affine/linear_heads/other/total）【実装済】
- **M2** ラウンド比 r_t = SI_conv_t / SI_conv_{t-1}（平衡→1、幾何級数→一定）
- **M3** conv別ノルム ‖w_i‖
- **M4** conv別勾配ノルム ‖∇w_i‖（固定プローブバッチで1回backward）
- **M5** conv別有効学習率 e_i = lr·‖∇w_i‖/‖w_i‖²
- **M6** conv別ラウンド回転角 γ_i = arccos(⟨w_i^t, w_i^{t+1}⟩/(‖·‖‖·‖))（Kosson の平衡指標）
- **M7** Exit別精度＝`evaluate_all_exits()`【実装済】

M4/M5 用プリミティブ（固定プローブ `probe=next(iter(val_loader))` を毎ラウンド使い回す）:
```python
def probe_effective_lr(model, probe, si_names, lr, device):
    from src.loss_functions import loss as loss_fn
    model.train()
    x, y = probe; x, y = x.to(device), y.to(device)
    model.zero_grad()
    preds = model(x)
    loss_fn('v1', device, preds, y).backward()
    out = {}
    for name, p in model.named_parameters():
        if name in si_names and p.grad is not None:
            wn = p.detach().float().norm().item()
            gn = p.grad.detach().float().norm().item()
            out[name] = (wn, gn, lr*gn/(wn*wn + 1e-12))
    model.zero_grad()
    return out
```

### E0：集中 vs FedAvg の対照（**完了**。§6 に結果）
- 変数: `num_clients=cohort` ∈ {1,5}。無対処・IID。
- 図A(M1): 横軸round／縦軸 SI_conv ノルム(対数)。→ K=1平衡・K=5指数増大。
- 図B(M2): 横軸round／縦軸 r_t（線形）。→ K=1→1・K=5→一定。

### E1：原因が“平均化”で不均一性でないと分離（次にやる）
- 本当の knob は「ラウンドあたりローカルステップ数 τ」＝同期頻度。`train()` に「τバッチで break」を追加して制御。
- 変数: τ ∈ {1,5,25,125}、`num_clients=cohort=5`、IID固定、無対処。
- 図: 横軸 τ(対数)／縦軸 定常ラウンド比 r（発散前 round後半の r_t 平均）。→ τ→1 で r→1（頻繁同期＝集中に漸近）、τ増で r増。
- 補助: 横軸 `num_clients`∈{1,2,5,10}／縦軸 定常r。

### E2：multi-exit 深度・Exit数依存（差別化の要）
- **E2a**: 横軸 conv位置(stem→stage1→2→3→final)／縦軸 e_i(M5) or γ_i(M6)。点を「逆伝播で通るExit数」で色分け（stages[j]のconvは num_ee−j 個、Exit分岐内convは1個）。→ 前半共有convほど大、後半・分岐conv小。
- **E2b**: 横軸round／縦軸 r_t、曲線を `num_ee`∈{2,4,8} で3本。→ Exit数が多いほど発散が速い／強い（multi-exit固有の看板結果）。

### E3：対処法比較（実用貢献＋反論潰し）
- 比較法: (a)conv再射影【実装済】/(b)global weight shrinking/(c)AdamP的接空間射影/(d)FedBN型(GN affineローカル保持=affine=Trueに戻す)/**(e)SI convをWD対象から除外**（`select_optimizer` を param group 2分割、SI群 weight_decay=0）。
- **(e) は査読者の帰無仮説なので必須**。
- 変数: 5手法 ×（IID: 1e10 / 非IID: 0.1）、K=5、E1で決めた τ、rounds=100。
- 図: 横軸round／縦軸 最終Exit精度・全Exit平均精度。表: 手法→Exit4精度／全Exit平均／r平均(平坦性)／NaN有無。

### IPSJ 最小構成
背骨は **E0→E1→E2b**（反例→機序→multi-exit固有性）。E2a・E3 は修論の肉付け。

---

## 6. E0 実験結果（**完了**・3 seed平均）

### E0/M1（図A）: SI_conv ノルム, K=1 vs K=5, IID, 無対処
- **K=1（中央集権）**: SI_conv は 65 前後で横ばい（**平衡**。既観測の≈57と整合）。
- **K=5（FedAvg）**: 65.0 → **9.29×10⁸**（30ラウンド、対数軸で直線＝幾何級数）。
- `GN_affine` は全ラウンド **0.0**（affine除去が効いている）。`total ≈ SI_conv`（暴走は SI_conv 部分空間に**局在**）。`linear_heads` は 3.8→14.7 の緩増、`other` は約18→16で安定。
- **精度は K=1・K=5 とも問題なく上昇**。→ SI_conv がスケール不変なので、‖w‖ が9億でも出力＝損失＝精度は不変。**発散＝精度崩壊ではない**ことを実証。

### E0/M2（図B）: ラウンド比 r_t, K=1 vs K=5
- **K=1**: 1.0 のすぐ上で微振動（0.98〜1.02）。WD と勾配ノイズによる平衡点周りの揺らぎで、**正常**（厳密に1.000ならむしろ要バグ疑い）。
- **K=5**: round1〜5 で 1.53→1.68→1.75 と立ち上がり、round10以降 **1.779 で完全に一定**（漸近成長率）。
- この 1.779 は handoff §2 の「毎ラウンド約1.78倍」と**ぴったり一致** → デバッグ時の経験的観測を、無対処・IID・厳密条件下で再現・定量化したことになる。

### E0 の解釈（§9 と連動・最重要）
- **反例としては成立**: Du の「FLではノルムは初期値近傍に留まる」に対し 65→9.3億 で明確な反証。
- **ただし精度に出ない**ので「so what?」への答えを用意する必要:
  - **有効学習率の崩壊**: e_i ∝ η/‖w‖²。round30 では round1 の約 (65/9.29e8)² ≈ **5×10⁻¹⁵ 倍** → SI conv は事実上**凍結**。短期IIDでは凍結前に特徴が学べているので精度が上がるが、非IID・長ラウンド・難タスクでは効いてくる。→ E2 の M5 で可視化。
  - **数値破綻**: 1.779倍/ラウンドなら round30 の 9.3億から約120ラウンド後、**round≈150 で float32 overflow → NaN**。長期連合学習では確実に落ちる。

### E0 の残タスク（教授に見せる前に）
1. **seed分散の確認**: `norm_report.csv`（生ログ）から seed ごとに r を計算し、K=5 の 1.78 が3 seedとも収束しているか。3本重なれば「構造的現象」の強い証拠。
2. **lr非依存の確認**: `lr`∈{0.01, 1.0} でも回し、漸近比 r が lr によらずほぼ同じ（発散開始ラウンドだけずれる）ことを見る。崩れたら「スケール不変性由来」の説明を再検討。
3. **handoff整合の明記**: r=1.779 が §2 の 1.78 と一致する旨を、結果の締めに書く（「過去の崩壊はこの機構で説明できる」）。

---

## 7. ファイル構成と役割

- `exp_05.py` … 実験本体。EENet定義／FLループ／`evaluate_all_exits`／`convert_bn_to_gn`（`affine=False`）。
- `E0_M1_図A.py` … E0 の実行スクリプト（FLループ＋群別ノルムCSV出力＋SI_conv対数プロット）。再射影OFFで呼ぶ。出力は `results/E0/M1/K={K}/`。
- `E0_M2_図B.py` … 既存の2つの `norm_report_mean.csv`(K=1,K=5) から r_t を計算して重ねる**描画専用**スクリプト（再学習不要）。出力 `results/E0/M2/`。
- `local.py` … `train`（clipは測定時コメントアウト）/`test`/`early_test`。
- `loss_functions.py` … `loss`/`loss_v1`(等重み)/`loss_v2`(線形)。⚠潜在バグ: 非Exitモデルで3要素タプルを返す不整合（eenet32では未発症）。
- `server.py` … `average_models`/`federated_learning`（si_param_names/ref_norms を渡せば集約後に conv 再射影）。
- `norm_tools.py` … `find_scale_invariant_conv_weights`/`capture_reference_norms`/`reproject_model_`/`norm_report`。
- `utils.py` … seed/optimizer選択/`calculate_percentile_thresholds`/`society_thresholds`。
- `config.yaml` … ハイパラ。

---

## 8. 研究方針と文献ポジショニング（再調査済み）

**共通の武器**: 動く「グローバル FedAvg × 4-Exit ResNet32(CIFAR-10)」実装＋E0で定量化した「スケール不変性×FedAvgのノルム暴走」。

### 案2: C-1（正規化・スケール不変性の動力学）※最有力・主軸
- **問い**:
  1. ノルム平衡破壊の **conv位置・Exit構造依存**（§9で訂正した正しい非対称）。
  2. 有効学習率 η/‖w‖² が **conv位置・Exit数**でどう分布するか。
  3. どの対処が **multi-exit精度を最もよく保つか**（E3 の (a)〜(e)）。
- **文献（この会話で再確認）**:
  - **Du et al. "Rethinking Normalization Methods in FL"**（arXiv:2210.03277）: **DistributedML 2022 のワークショップ論文(6pp)**。「同じ初期値から始めるFLではスケール不変性ゆえノルムは初期値から大きく逸脱しない」と主張。ただし表現は控えめ（"some guarantee ... will not diverge far"）で、彼ら自身の式（WD無し Eq.7）は単調増加を示す。→ **我々の反例のターゲット**だが、的が小さい点に注意。
  - **集中学習では既知**（＝「ノルムが増える」だけでは新規性なし）: Arora et al. 2018（スケール不変損失でノルム単調増加・有効LR η/‖w‖²）、Van Laarhoven 2017、Li & Arora 2019（exponential/intrinsic LR）、**AdamP**（arXiv:2006.08217、接空間射影でノルム増大抑制＝E3(c)のベース）、**Kosson et al. "Rotational Equilibrium"**（ICLR 2024, arXiv:2305.17212、WDが角度更新で有効LRを均衡させる平衡像）。
  - **FLのノルム制御**: Revisiting Weighted Aggregation / adaptive GWS（arXiv:2302.10911、E3(b)のベース）。
  - **反証の有無**: 「FLでDuの平衡が破れる」と正面から実証した論文は**見つからず**（＝交差点は空いている）。ただし上記の通り集中側は埋まっているので、看板は「Du反証」単体ではなく **FedAvg平均化×multi-exit** に置く。

### 案1: A-3 → A-1（セキュリティ系）※修論の応用の山場として温存
- A-3（良性non-IIDでのExit単位ドリフト測定）→ A-1（exit-consistency での backdoor/Byzantine 検知）。
- 攻撃面: 悪性クライアントが**浅いExitだけにトリガー**を仕込むと、最終層しか見ない既存防御(Krum等)をすり抜けうる。Exit間不整合を検知指紋に。
- 文献: CAFEDistill（arXiv:2601.10015、PFL-EEの手法論文。目的が精度向上のPFLなのでGFL+検知とは差分）、disagreement-OOD 系。「multi-exit の exit-consistency を**クライアント単位**で検知」の交差点は空き。

### 縦串と暫定プラン
- **C-1 →（A-3）→ A-1** に一本化（C-1 が A-3 の機序を与える）。
- 11月 IPSJ は **C-1 の分析を先行**、修論は C-1 を土台・A-1 を山場。
- **教授と決めること**: (1)修論の重心（C-1寄り/A-1寄り）、(2)IPSJをC-1先行でよいか、(3)狙うSIG（CSEC/MPS/DPS等）。

---

## 9. §8旧C-1問1の訂正（重要・概念）

旧版に「各Exit分岐が独自の正規化終端サブグラフを持つため、**深いExitほど下流GN数が多く暴走機構が増える**」と書いたが、これは**誤り**。正しくは:

- **暴走の自由度は conv 1本につき1個**。あるSI conv W を定数倍すると**直後の1個のGN**が吸収するので、W の平坦方向は1本。下流にGNがいくつあっても W の自由度は増えない。暴走機構の総数＝SI conv 本数＝36本で固定であり、Exit深度で増えるものではない。
- **各Exit分岐の内部GN数は同数**（Exit1〜3は各1個、最終Exitは分岐内0個）。「深いExitほど分岐内GNが多い」ことはない。
- **conv基準で下流GN数を数えると、むしろ浅いトランクのconvほど下流GNが多い**（後続の全ブロック＋全Exit分岐のGNが下流）。方向が逆。

**正しい非対称（C-1で測るべきもの）**:
1. **勾配の重畳数がconv位置で違う**。`loss_v1` は4Exitの等重み和なので、前半共有convは Exit1〜4 全部の逆伝播経路上にあり複数Exit損失の勾配が重畳。後半トランク（stages[-1]）は最終Exitのみ、Exit分岐内convは自Exitのみ。→ 膨張項 η²‖∇L‖² の駆動がconv位置で構造的に非対称（浅い共有convほど多Exitで駆動＝「深いExitほど暴走」とはむしろ逆向き）。
2. **各Exitの予測経路に乗るSI conv本数**は深いExitほど多い（Exit1: stem+stage0+exit1conv、最終: 全トランク）。旧記述が言いたかったのはおそらくこれだが、正しくは「下流」でなく「上流・共有」で、Exit専有でなく共有。

**混乱の根**: 暴走は**パラメータ（conv）ごとの性質**で、「Exitごと」に割り当てられる量ではない。トランクのconvは複数Exitに共有されるので単一Exitに帰属できない。→ E2 は「Exit分岐別」に切るより、**(i) conv位置別、(ii) そのconvが逆伝播で受け取るExitの集合、の2軸**で切る方が筋が通る。

---

## 10. 査読者の反論への備え（先に潰す）

1. **「精度は落ちないなら何が問題？」** → 害は潜在的（§6の解釈）。有効学習率の凍結（M5で可視化）と長期overflow。E2で「バックボーンが学び続ける必要のある設定（非IID・長ラウンド）で効く」ことを示す。
2. **「SI paramをWD対象から外せば済む既知の話（＝設定バグ）では？」** → 正規化重みをWD除外するのは標準的助言。だから主張を「**FLは集中の対処（WD除外/再射影）だけで足りるのか、averagingのせいで追加で何か要るのか**」に立て直す。E3(e) をベースラインに必ず入れ、(a)〜(d)が(e)を上回るかで答える。どちらの結論でもC-1の貢献になる。
3. **「ノルムが増えるのは既知（Arora等）」** → その通りなので「増える」を主張しない。**FedAvg平均化が平衡を壊す点（IIDでも）＋multi-exit依存性**に限定する。

---

## 11. 次にやること（次チャットの冒頭タスク）

1. **E0 残タスク3点**（§6末尾: seed分散・lr非依存・handoff整合の明記）を片付け、E0 図A/図B を教授提示レベルに。
2. **E1 に着手**: `train()` に τ-break を実装し、τ∈{1,5,25,125} で K=5・IID・無対処を回す。図（横軸τ／縦軸定常r）で「averaging頻度が駆動因子」を示す。
3. 余力があれば **E2b**（`num_ee`∈{2,4,8} で r_t）に着手。

---

（v3 追記の元になった検討: E0 の M1/M2 完了と解釈、文献の再ポジショニング、§8問1の訂正、査読反論への備え。次版では E1 の結果を §6 に追記予定。）
