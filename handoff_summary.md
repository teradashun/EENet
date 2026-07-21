# EE_FL プロジェクト 引き継ぎサマリー（更新版）

> 次回チャットの冒頭にこのファイルを貼り付け／再アップロードすれば、これまでの経緯を引き継げます。

---

## 1. プロジェクト概要

- **テーマ**: Early-Exit ネットワーク × Federated Learning（EE_FL / FedAvg）
- **モデル**: `eenet32`（6n+2 型 ResNet32 ベース、BasicBlock、中間Exit3個＋最終Exit1個＝計4Exit）
- **データ**: CIFAR10（10クラス）、IID設定（Dirichlet α = 1e9）
- **集約**: FedAvg（`server.py`）
- **損失**: `loss_v1`（全Exitの CrossEntropy を等重み平均）
- **最終的な研究テーマ**: Multi-Exitの一貫性（exit-consistency）を用いたFLにおけるバックドア／Byzantine検知
- **当面のマイルストーン**: 11月の情報処理学会（IPSJ）の研究会（具体的なSIG名は未定）に向けて、まず「non-IIDによるExit単位のクライアントドリフト分析」を先に行い、その後上記のセキュリティ研究に進む計画
- **現在地**: 多クライアントFLの学習不安定性（後半での損失発散）を根治し、STEP4（閾値ベースのEarly-Exit評価）に進む直前の段階。
- **研究方針（重要・更新）**: 引き継ぎ後の文献調査を踏まえ、今後の方向性を **2案（案1: A-3→A-1 セキュリティ系 / 案2: C-1 正規化・スケール不変性の動力学）** に整理した。**§8 を参照。教授と相談して重心を決める段階。** §5 の当初プラン（STEP4→A-3）は案1の一部として残るが、A-3 を*単体の看板*にはしない方針に変更済み。

---

## 2. これまでのデバッグ経緯（時系列）

### 症状①：損失が数百〜数千に爆発 → 全Exitが10%（＝ランダム）
- **対策**: `local.py` の `train()` に勾配クリッピングを追加
  `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **結果**: 数千への急性爆発は止まったが、多クライアントでは依然10%へ崩壊。

### 症状②：多クライアントでのみ崩壊。`num_clients=1`（中央集権）は正常（80%程度まで学習）
- → **FL特有の問題**と確定。

### 症状③：BatchNorm の平均化が原因と特定
- 自作の `bn_axis_check`（(A)eval凍結統計 / (B)trainモードBN / (C)BN再計算後）を崩壊前のRound1〜2で実行し、BNのrunning統計をクライアント平均すると壊れることを実証。
- **対策**: 全 `BatchNorm2d` を `GroupNorm` に置換。
- **結果**: 序盤（〜Round12あたり）はきれいに精度上昇するようになった。

### 症状④：GN化後、後半で損失が緩やかに幾何級数的増加（発散）
- lr を100倍振ってもほぼ同じラウンドで発散（lr非依存）。momentum=0.5で即爆発、0.9で安定。
- 重みノルム `‖w‖` をラウンドごとにログし、5クライアントで毎ラウンド約1.78倍の指数増大、1クライアントでは平衡（≈57）に張り付くことを確認。
- **根本原因の仮説**: GN直前の conv 重みはスケール不変（定数倍しても損失不変）。このとき勾配は重みと直交し、有効学習率 ∝ η / ‖w‖²。中央集権では膨張項と収縮項（weight decay）が釣り合い ‖w‖ が平衡するが、**FedAvgの平均化がこの釣り合いを壊し**、各クライアントの膨張成分が平均後も残る一方で収縮が追いつかず指数増大する、と推定。

### 症状④の対処①：スケール不変convの集約後ノルム再射影を実装
- `src/norm_tools.py` を新規作成。`find_scale_invariant_conv_weights()` で「bias無しConv2dの直後にGroupNormが続く」箇所を自動検出（eenet32では該当36個）。
- 集約直後にこれらconv weightのノルムだけを初期値（基準ノルム）へリスケールする `reproject_model_()` を実装し、`server.py` の `federated_learning()` から呼び出すよう変更。
- あわせて `norm_report()` で ‖w‖ を `SI_conv`（スケール不変conv）／`GN_affine`（GroupNormのγ,β）／`linear_heads`（各Exit fcと最終fc）／`other`（final_adapter等）の4群に分けてログする診断機能を追加。
- **結果（5クライアント・15ラウンドで実行）**: `SI_conv` は48.5で完全に平坦化。しかし `total` は162551まで従来同様に指数増大 → **暴走源が別の場所に残っていることが判明。**

### 症状④の対処②：GN affine（γ, β）が残りの暴走源と特定
- 群別ログの結果、`GN_affine` が 56.5 → 162551 と**約1.78倍/ラウンドで指数増大**しており、これが `total` の暴走とほぼ完全に一致（`heads`と`other`は安定）。
- **原因**: 中間ブロックのGN（下流に別のGNが続くもの）は、出力が `γ·x̂+β → ReLU → conv → 次のGN` と流れるため、(γ, β) を同時に定数倍しても次のGNが吸収し損失は不変。conv weightとまったく同じ「スケール不変×FedAvg平均化」の機構で暴走していた。
- **対策**: `convert_bn_to_gn()` に `affine=False` を追加し、GroupNormのγ,βパラメータ自体を除去（conv側の再射影はそのまま維持）。
- **結果（5クライアント・15ラウンドで再実行）**:
  - `GN_affine` は常に0.00、`total` は51.1〜51.7で完全に平坦化（暴走消失）。
  - 損失は2.4→1.3台まで単調減少し、後半での再発散なし。
  - 各Exit精度（Round15）: Exit1=37.91%, Exit2=43.51%, Exit3=46.02%, **Exit4(最終)=48.60%**。Exit4が全ラウンドを通じて最良となり、1クライアント（中央集権）で見られた「深いExitほど良い」という健全なパターンと一致。
- → **不安定性の根治を確認。**

---

## 3. 確定した根本原因（症状④・最終版）

**スケール不変性 × FedAvg平均化による重みノルムの平衡破壊。ただしスケール不変パラメータは2種類あった。**

1. GN直前の conv 重み（bias無し）
2. 下流に別のGNが続く GroupNorm の affine パラメータ（γ, β）

いずれも「定数倍しても下流のGNが吸収し損失が不変」という意味でスケール不変であり、勾配が重みと直交するため有効学習率 ∝ η/‖w‖² となる。中央集権では膨張項（η²‖∇L‖²）と収縮項（weight decay, −2ηλ‖w‖²）が釣り合い ‖w‖ は平衡するが、FedAvgの平均化がこの釣り合いを壊し、各クライアントの膨張成分は平均後も残る一方で収縮が追いつかず指数増大する。

**対処は2段階必要だった**：conv側だけ再射影しても、GN affine側が同じ機構で暴走を引き継いだため、両方を同時に止めて初めて根治した。

---

## 4. 現在の config・実装（要点）

```yaml
model_name: eenet32
dataset: CIFAR10
num_ee: 4
loss_func: v1
optim_name: SGD   # lr=0.01, momentum=0.9, weight_decay=1e-4
FL:
  num_clients: 5
  cohort: 5
  global_rounds: 15   # ← 次段階でここを30〜50程度に拡大予定
  dirichlet: 1e9   # ほぼIID
```

- `local.py` の `train()` に `clip_grad_norm_(max_norm=1.0)` 追加済み。
- `exp_05.py` の `convert_bn_to_gn()` は `affine=False` に変更済み（GN affine除去）。
- `src/norm_tools.py`（新規）: スケール不変conv検出・基準ノルム記録・再射影・群別ノルム診断。
- `server.py` の `federated_learning()` は `si_param_names` / `ref_norms` を受け取り、集約直後にconv側の再射影を実行するよう変更済み。

---

## 5. 次にやること

> ⚠ 注記（更新）: 以下は当初の技術ロードマップ（STEP4→A-3）。研究の*看板*としては §8 の2案検討に移行済み。案2（C-1）を主軸にする場合、下記1〜3は「案1へ橋渡しする補助実験」または「C-1の測定基盤」として再利用する位置づけになる。

1. **`global_rounds` を30〜50程度に拡大して再実行**：現状15ラウンドではExit4がまだ48.6%で、1クライアント基準の70%には未到達。GN affineありでは9ラウンド以降で崩壊していたため長く回せていなかっただけの可能性が高く、伸び続けるか頭打ちになるかをまず確認する。
2. 精度上昇（またはプラトー）を確認できたら、**本来のSTEP4（閾値ベースのEarly-Exit評価）へ進む**：`utils.py` の `calculate_percentile_thresholds` / `society_thresholds` は実装済みなので、`evaluate_all_exits` を `exit_threshold` 付きで呼ぶ評価ループを組む。
3. STEP4完了後、当面のマイルストーンである「non-IIDによるExit単位のクライアントドリフト分析」へ進む（Dirichlet αを1e9から下げてnon-IID化する想定）。

---

## 6. 検証の定番手順

- 集約直後に `norm_report()` で `SI_conv` / `GN_affine` / `linear_heads` / `other` を群別にログする（全体合算の `‖w‖` だけでは暴走源を誤認する。実際に conv側だけ再射影した際、GN affine側の暴走を見落としかけた）。
- 損失が後半で増加に転じないこと、Exit4（最終）が全Exit中で最良に戻ることを確認する。

```python
def global_weight_norm(model):
    import torch
    with torch.no_grad():
        sq = sum(p.detach().float().pow(2).sum() for p in model.parameters())
    return sq.sqrt().item()
```

---

## 7. ファイル構成と役割

- `exp_05.py` … 実験本体。EENet定義（BasicBlock/ExitBlock/EENet/forward）、FLループ、`evaluate_all_exits`、`convert_bn_to_gn`（`affine=False`に変更済み）。
- `local.py` … `train`（clip追加済み）/ `test` / `early_test`。
- `loss_functions.py` … `loss` / `loss_v1`（等重み）/ `loss_v2`（線形重み）。
  - ⚠ 潜在バグ：非Exitモデル（pred が tensor）のとき3要素タプルを返し、`.item()`/`.backward()` 前提の呼び出し側と不整合。eenet32では未発症。
- `server.py` … `average_models` / `federated_learning`（**更新済み**：`si_param_names`/`ref_norms`を受け取り集約後にconv再射影を実行）。
- `src/norm_tools.py`（**新規**）… `find_scale_invariant_conv_weights` / `capture_reference_norms` / `reproject_model_` / `norm_report`。
- `utils.py` … seed / optimizer選択 / `calculate_percentile_thresholds` / `society_thresholds`（STEP4用の閾値計算、GN化後も未変更で使用可）。
- `config.yaml` … ハイパラ。

---

## 8. 研究方針の選択肢（2候補・文献ポジショニング）※要・教授判断

**背景**: 引き継ぎ後の文献調査で、当初の「A-3: non-IIDによるExit単位ドリフト分析」を*単体の看板*にするのは新規性が弱いと判明（設定を CAFEDistill が、"Exit不一致=信号" の発想を disagreement-OOD 系がカバー）。そこで研究方針を2案に整理し、教授と相談して重心を決める段階。目標は **11月 IPSJ 研究会（SIG未定）＋修士論文**。共通の武器は、既に動く「グローバル FedAvg × 4-Exit ResNet32（CIFAR-10）」実装と、§2〜§3 で根治した**スケール不変性×FedAvgのノルム暴走という独自発見**。

### 案1: A-3 → A-1（セキュリティ系）
- **内容**: A-3（良性 non-IID での Exit 単位ドリフト／不整合ベースライン測定）を土台に、A-1（exit-consistency を用いた backdoor/Byzantine 検知）へ。
- **攻撃面の主張**: 悪性クライアントが**浅いExitだけにトリガー**を仕込むと、最終出力・最終層しか見ない既存防御（Krum, Multi-Krum, Trimmed Mean, FLAME）をすり抜けうる。逆に**Exit間不整合を指紋**として検知に転用（Exit-Consistency-Aware Aggregation）。
- **文献ポジショニング**:
  - **CAFEDistill**（arXiv:2601.10015）: PFL-EE の**手法**論文。設定は近いが目的は精度向上で PFL、こちらは **GFL(FedAvg)＋検知**。A-3の記述分析だけだと食われる。深度間干渉≠Exit単位クライアントドリフト、である点も差分。
  - 「不一致=異常信号」は一般的（logit disagreement, multi-exit CAM OOD 等）。FLのクライアント更新相違での検知も混雑（Representational Dissimilarity Analysis arXiv:2503.04473、Energy/Wasserstein 構造チェック）。
  - ただし「**multi-exit の exit-consistency を*クライアント単位*で backdoor/Byzantine 検知**」の正確な交差点は空き。差別化の核＝攻撃面の指摘＋global regime＋クライアント単位の検知特徴量。
- **実装**: BranchyNet風多分岐＋Dirichlet non-IID＋BadNets/DBA 注入＋既存防御と比較（ASR／主タスク精度／偽陽性率）。**やや重い**（攻撃・防御・複数ベースライン）。

### 案2: C-1（正規化・スケール不変性の学習動力学）※手持ち資産を再利用・最有力候補
- **内容**: §2〜§3 で根治した「スケール不変性×FedAvg平均化によるノルム暴走」を、**multi-exit特有の問題**として体系化する研究に昇格。
- **問い**:
  1. ノルム平衡破壊は **Exit深度に依存するか**（各Exit分岐が独自の正規化終端サブグラフ＝スケール不変パラメータ群を持つため、深いExitほど下流GN数が多く暴走機構が増える等の非対称が予想される）。
  2. 有効学習率 ∝ η/‖w‖² は **Exitごと・深度ごと**にどう分布するか。
  3. どの対処が **multi-exit精度を最もよく保つか**: (a) 集約後ノルム再射影【実装済＝`norm_tools.py`】, (b) global weight shrinking, (c) AdamP的な接空間射影optimizer, (d) 正規化affineをローカル保持（FedBN型）。
- **文献ポジショニング**:
  - **Du et al. "Rethinking Normalization Methods in FL"**（arXiv:2210.03277）: スケール不変性ゆえ FL では重みノルムが**初期値付近に留まる**と主張。→ 我々の「multi-exit＋GN＋WD で指数暴走する実測反例」が**この主張の破れる領域**を示す（＝主張の核・最重要フック）。
  - **Revisiting Weighted Aggregation / adaptive GWS**（arXiv:2302.10911）: スケール不変性を根拠に global weight shrinking を提案 → **対処法比較のベースライン**。
  - **AdamP**: スケール不変重みの過剰成長を接空間射影で抑制 → **対処法比較のベースライン**。
  - 一般の norm 動力学は占有領域なので、**multi-exit の深度依存性**と**Du らへの反例**で締めることが必須。
- **実装**: 動く実装と実測データが既にあり**半分完了**。測定は既存FLループに群別・Exit別のスカラーログ（‖w‖、有効学習率 ‖∇w‖/‖w‖²、Exit分岐ごとのスケール不変サブグループ別ノルム）を足すだけ。**IPSJ研究会に分析として出しやすい**。

### 2案の関係と暫定プラン
- **地続きの関係**: C-1（どのExit・層がノルム／有効学習率で不安定か）は A-3（Exit単位ドリフト）の**機序**を与える。つまり **C-1 →（A-3）→ A-1** と一本化でき、修論の縦串になる。
- **暫定**: 11月 IPSJ は **C-1 の分析を先行**（手持ちで確実）。修論は **C-1 を機序解析の土台**に、**A-1（セキュリティ）を応用の山場**に据える。
- **教授と決めること**: (1) 修論の重心＝A-1（セキュリティ）寄り or C-1（最適化・正規化）寄り、(2) 11月 IPSJ を C-1 先行でよいか、(3) 狙う SIG（CSEC / MPS / DPS 等）。

### C-1 を進める場合の直近タスク（次チャット用）
1. `norm_report()` を **Exit分岐別**に拡張（現状の SI_conv / GN_affine / heads / other の4群を、さらに「どのExitに属すか」で細分）。§6 の診断を Exit 軸で切り直すイメージ。
2. **有効学習率の実測**: 各ラウンドで層ごとに ‖∇w‖/‖w‖²（またはローカル1ステップの回転角）を記録し、Exit深度に対してプロット。
3. **対処法の比較実験**: (a)再射影 /(b)GWS /(c)射影optimizer /(d)affineローカル保持 を同一 non-IID グリッドで回し、最終Exit精度・全Exit精度・‖w‖平坦性を比較。
4. **Du らへの反例の主張整理**: 「どの条件（multi-exit×GN×WD×FedAvg平均化×ラウンド数）で平衡が破れるか」を最小構成で切り分け、反例として提示できる図表を用意。

