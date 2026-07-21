"""
norm_tools.py

スケール不変性 × FedAvg平均化による重みノルム発散への対処ツール。

役割:
  1) find_scale_invariant_conv_weights: GN直前の bias無し Conv2d の weight を自動検出。
     GN はスケールを吸収するため、これらの weight を定数倍しても損失は不変（スケール不変）。
  2) capture_reference_norms: 学習開始時(初期)の基準ノルムを記録。
  3) reproject_model_: 集約直後、スケール不変 conv の weight ノルムだけを基準値へ戻す。
     スケール不変なので損失・精度は厳密に不変。暴走成分（ノルムの指数増大）だけが消える。
  4) norm_report: ノルムを「SI conv / GN affine / linear heads / other」に分けて記録。
     ‖w‖ の増大がどの部分空間で起きているかを局在化するための診断。

すべて named_parameters / named_modules の正準名で統一しているのでキー不整合は起きない。
"""

import math
import torch
import torch.nn as nn


def find_scale_invariant_conv_weights(model):
    """
    「bias無し Conv2d の直後に GroupNorm が続く」箇所の conv.weight のパラメータ名集合を返す。

    検出方法:
      各モジュールの named_children を順序どおりに走査し、
      child[i] が Conv2d(bias=None) かつ child[i+1] が GroupNorm のとき、child[i].weight を採用。

    これで拾える例（eenet32 / GN化後）:
      - stem の conv → GN
      - BasicBlock の conv1 → bn1(GN), conv2 → bn2(GN)
      - downsample の conv1x1 → GN
      - ExitBlock.layers の conv → GN
    拾わない（＝再射影しない、で正しい）もの:
      - final_adapter (→ fc に直結, GNなし → スケール不変でない)
      - ExitBlock.adapter (→ conv → GN の連鎖。原理上は不変だが直後GNではないので対象外)
      - fc(Linear), GN の affine(γ, β)
    """
    si = set()
    for module_name, module in model.named_modules():
        children = list(module.named_children())
        prefix = (module_name + ".") if module_name else ""
        for i in range(len(children) - 1):
            cur_name, cur = children[i]
            _, nxt = children[i + 1]
            if isinstance(cur, nn.Conv2d) and isinstance(nxt, nn.GroupNorm):
                if cur.bias is None:  # bias有りだと厳密なスケール不変ではないので除外
                    si.add(f"{prefix}{cur_name}.weight")
    return si


@torch.no_grad()
def capture_reference_norms(model, si_param_names):
    """スケール不変 conv weight の基準ノルム(初期値)を記録して返す。"""
    return {
        name: p.detach().float().norm().item()
        for name, p in model.named_parameters()
        if name in si_param_names
    }


@torch.no_grad()
def reproject_model_(model, si_param_names, ref_norms, eps=1e-12):
    """
    集約直後に呼ぶ。スケール不変 conv weight のノルムだけを基準値へリスケール(in-place)。
    方向はそのまま、大きさだけ固定。スケール不変なので損失・精度は不変。
    named_parameters 上で直接操作するので state_dict のエイリアス問題を回避できる。
    """
    for name, p in model.named_parameters():
        if name in si_param_names:
            cur = p.data.float().norm()
            if cur > eps:
                p.data.mul_(ref_norms[name] / cur.item())
    return model


def _classify(model):
    """GN affine と Linear(head) のパラメータ名集合を返す（診断用の分類）。"""
    gn_names, lin_names = set(), set()
    for mname, m in model.named_modules():
        prefix = (mname + ".") if mname else ""
        if isinstance(m, nn.GroupNorm):
            for pn, _ in m.named_parameters(recurse=False):
                gn_names.add(prefix + pn)
        elif isinstance(m, nn.Linear):
            for pn, _ in m.named_parameters(recurse=False):
                lin_names.add(prefix + pn)
    return gn_names, lin_names


@torch.no_grad()
def norm_report(model, si_param_names):
    """
    ‖w‖ を4群に分けて返す:
      SI_conv      : スケール不変 conv weight（再射影の対象。ONなら平坦になるはず）
      GN_affine    : GroupNorm の γ, β
      linear_heads : 各Exit fc / 最終 fc の weight, bias
      other        : final_adapter / ExitBlock.adapter など（残り。スケール不変だが未固定）
    total は全体ノルム（従来の global_weight_norm と一致）。

    使い方: 再射影ONで SI_conv が平坦なのに total が伸び続けるなら、
            発散源は GN_affine か linear_heads 側（＝素直な損失発散）で、再射影では治らない。
    """
    gn_names, lin_names = _classify(model)
    sq = {"SI_conv": 0.0, "GN_affine": 0.0, "linear_heads": 0.0, "other": 0.0}
    for name, p in model.named_parameters():
        v = p.detach().float().pow(2).sum().item()
        if name in si_param_names:
            sq["SI_conv"] += v
        elif name in gn_names:
            sq["GN_affine"] += v
        elif name in lin_names:
            sq["linear_heads"] += v
        else:
            sq["other"] += v
    rep = {k: math.sqrt(v) for k, v in sq.items()}
    rep["total"] = math.sqrt(sum(sq.values()))
    return rep