"""
utils.py — FuidAI training utilities.
Cung cấp các hàm helper cần thiết cho finetuning.py và inference.py
mà không phụ thuộc vào các package ngoài không tồn tại.
"""

from __future__ import annotations

import functools
import os

import torch


# ─── XPU (Intel GPU) detection ───────────────────────────────────────────────

def is_xpu_available() -> bool:
    """Kiểm tra Intel XPU (IPEX) có sẵn không."""
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return torch.xpu.is_available()
    except (ImportError, AttributeError):
        return False


# ─── FSDP auto-wrap policy ───────────────────────────────────────────────────

def fsdp_auto_wrap_policy(model, transformer_layer_cls: list):
    """
    Xây dựng FSDP auto-wrap policy bao quanh từng lớp transformer
    được chỉ định trong `transformer_layer_cls`.

    Args:
        model: mô hình PyTorch (chỉ dùng để kiểm tra, không thay đổi).
        transformer_layer_cls: danh sách các class lớp transformer cần wrap.

    Returns:
        callable: policy truyền vào FSDP(auto_wrap_policy=...).
    """
    # _or_policy là private API — thử import, fallback sang size-based
    try:
        from torch.distributed.fsdp.wrap import (
            _or_policy,
            lambda_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )
        def _leaf_with_grad(module: torch.nn.Module) -> bool:
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )
        leaf_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=_leaf_with_grad)
        xfm_policy  = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_layer_cls),
        )
        return functools.partial(_or_policy, policies=[leaf_policy, xfm_policy])
    except ImportError:
        # Fallback an toàn: chỉ dùng transformer wrap policy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_layer_cls),
        )
