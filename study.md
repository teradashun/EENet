A. 最重要：正規化層のスケール不変性と「有効学習率」
今回の発散の核心。ここを理解すると「なぜlrを100倍振っても発散ラウンドが動かないか」が腑に落ちます。

van Laarhoven (2017) "L2 Regularization versus Batch and Weight Normalization" (arXiv:1706.05350) — 出発点。正規化層があると、正則化がないと重みが無限に増大し有効学習率が0に向かうため、L2正則化（weight decay）は依然として有益という主張。まずこれを読む。 arXiv
Arora et al. (2019) "Theoretical Analysis of Auto Rate-Tuning by Batch Normalization" (arXiv:1812.03981) — 勾配が常に現在のパラメータベクトルと直交し、その結果スケール不変なパラメータ群のノルムが単調増加してオートチューニング効果を持つことを理論化。今回の「‖w‖増大」の理論的裏づけ。 arXiv
Li & Arora (2019) "An Exponential Learning Rate Schedule for Deep Learning" (arXiv:1910.07454) — weight decay は指数増加LRスケジュールと等価であり、この等価性はGroupNorm/LayerNorm等の他の正規化層でも成立する。GN化で問題が顕在化した理由に直結。 arxiv
(直感的解説) inference.vc ブログ — https://www.inference.vc/exponentially-growing-learning-rate-implications-of-scale-invariance-induced-by-batchnorm/ — スケール不変な損失では原点から離れるほど勾配が小さくなることを図で説明。論文の前に読むと理解が速い。 arxiv
Li, Wang et al. (2023) "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks" (arXiv:2305.17212) — weight decay の正則化効果は主に重みノルムの減少による有効学習率の増加で説明できる。「平衡ノルム」の概念が今回の‖w‖≈57の話そのもの。 arxiv

B. Federated Learning の基礎とクライアントドリフト

McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks from Decentralized Data" (arXiv:1602.05629) — FedAvg 原論文。まずこれ。
Karimireddy et al. (2020) SCAFFOLD (arXiv:2003.00295) と Li et al. (2020) FedProx (arXiv:1812.06127) — FedAvgは各クライアントが複数ステップ進んだ後の「パラメータ」を平均するため、損失地形が非凸だとどのクライアントの解より悪い点に着地しうる（クライアントドリフト）。FedProxの近接項などで緩和する。IIDでも非凸ゆえドリフトは残る点が今回の学びの要。 ACM Digital Library
Hsu et al. (2019) "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification" (arXiv:1909.06335) — FedAvgM（サーバ側モメンタム）の出典。次の一手の候補。

C. FLにおける正規化層（BN問題とGN）
症状③④に直結。今回あなたが独力で辿り着いた結論が、そのまま文献の主流と一致します。

Hsieh et al. (2020) "The Non-IID Quagmire of Decentralized Machine Learning" (arXiv:1910.00189) — GroupNormがFLの収束を改善できることを最初に示した論文。BN→GN置換の根拠。 arxiv
Li et al. (2021) "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" (arXiv:2102.07623) — BN層をアップロード・平均せずローカル保持することで、平均化前の feature shift を緩和する。GNを使わずBNを残す代替案。 arxiv
Du et al. (2022) "Rethinking Normalization Methods in Federated Learning" (arXiv:2210.03277) — 独立なローカル学習が引き起こす外部共変量シフトを明示的に指摘し、正規化層がFLで不可欠である理由を論じる。BNがなぜFLで劣化するかの整理に良い。 arXiv

D. 次の一手（ノルム制御）の技術

Wu & He (2018) "Group Normalization" (arXiv:1803.08494) — GN本体。今使っている層の原理。
Qiao et al. (2019) "Weight Standardization" (arXiv:1903.10520) — 対策候補②。GNと組み合わせる正規化。
Brock et al. (2021) "High-Performance Large-Scale Image Recognition Without Normalization"（NFNets） (arXiv:2102.06171) — 正規化を使わず適応的勾配クリッピングで安定化する手法。あなたが最初に入れた勾配クリッピングの発展形として示唆に富む。

E. Early-Exit ネットワーク（研究の土台）

Teerapittayanon et al. (2016) "BranchyNet" (arXiv:1709.01686) — 側枝分類器を追加し、高い確信度で推論できるサンプルは早期の枝から出力させる元祖。閾値ベース早期退出の原点でSTEP4に直結。
Huang et al. (2018) MSDNet (arXiv:1703.09844) と Kaya et al. (2019) "Shallow-Deep Networks" (arXiv:1810.07052) — Shallow-Deep Networks は各内部分類器のsoftmaxの最大確率を確信度スコアとして使う。あなたの forward の閾値判定と同じ枠組み。
(サーベイ) "Early-Exit Deep Neural Network — A Comprehensive Survey" (ACM Computing Surveys, 2024) — https://dl.acm.org/doi/10.1145/3698767 — 早期退出型DNNは推論の高速化に加え、勾配消失の緩和・過学習や overthinking の抑制といった利点も持つ。全体像の把握に。

おすすめの学習順序
まず A（inference.vcブログ → van Laarhoven → Li&Arora） で「なぜlr非依存か」を腹落ちさせ、次に B（FedAvg → FedProx/クライアントドリフト） でFLの基本、そして C で両者の交差点（BN/GNのFL問題）に進むのが、今回のバグを最短で完全理解できる道筋です。D・Eは対策実装とSTEP4本体に着手するときの参照用に回して構いません。
Aの3本を押さえた時点で、今回の「‖w‖ログが指数増大していた」現象を自分の言葉で第三者に説明できるようになるはずです。そこが研究ノートや面接での説明力に直結します。