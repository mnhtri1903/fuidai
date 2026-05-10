"""
model_checkpointing.py — FuidAI FSDP checkpoint utilities.
Cung cấp các hàm load/save checkpoint cho single-GPU và FSDP multi-GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


# ─── Load sharded FSDP checkpoint vào single-GPU model ───────────────────────

def load_sharded_model_single_gpu(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    """
    Nạp FSDP sharded checkpoint từ `model_path` vào model trên single GPU.

    Hỗ trợ hai định dạng:
    1. PyTorch Distributed Checkpoint (DCP) — có file `.metadata`.
    2. Checkpoint thông thường (*.bin / *.pt / *.pth).

    Args:
        model:      Instance model rỗng (đã khởi tạo đúng kiến trúc).
        model_path: Thư mục chứa checkpoint.

    Returns:
        model với trọng số đã nạp.
    """
    model_path = str(model_path)
    metadata_file = os.path.join(model_path, ".metadata")

    # ── Thử DCP format (PyTorch >= 2.0) ──────────────────────────────────────
    if os.path.exists(metadata_file):
        try:
            from torch.distributed.checkpoint import FileSystemReader
            from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

            state_dict = {"model": model.state_dict()}
            dcp_load(
                state_dict=state_dict,
                storage_reader=FileSystemReader(model_path),
            )
            model.load_state_dict(state_dict["model"])
            print(f"[model_checkpointing] Đã nạp DCP checkpoint từ: {model_path}")
            return model
        except Exception as e:
            print(f"[model_checkpointing] DCP load thất bại ({e}), thử fallback...")

        # Fallback cũ hơn (PyTorch 1.x)
        try:
            from torch.distributed._shard.checkpoint import (
                FileSystemReader,
                load_state_dict,
            )
            state_dict = {"model": model.state_dict()}
            load_state_dict(state_dict=state_dict, storage_reader=FileSystemReader(model_path))
            model.load_state_dict(state_dict["model"])
            print(f"[model_checkpointing] Đã nạp (legacy DCP) từ: {model_path}")
            return model
        except Exception as e2:
            print(f"[model_checkpointing] Legacy DCP cũng thất bại ({e2}), thử file thường...")

    # ── Fallback: tìm file .bin / .pt / .pth ─────────────────────────────────
    search_path = Path(model_path)
    for pattern in ("pytorch_model.bin", "*.bin", "*.pt", "*.pth"):
        candidates = list(search_path.glob(pattern))
        if candidates:
            ckpt_file = candidates[0]
            ckpt = torch.load(ckpt_file, map_location="cpu")
            if isinstance(ckpt, dict):
                # Có thể là {"model": state_dict} hoặc raw state_dict
                state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            else:
                state = ckpt
            model.load_state_dict(state, strict=False)
            print(f"[model_checkpointing] Đã nạp checkpoint từ: {ckpt_file}")
            return model

    raise FileNotFoundError(
        f"Không tìm thấy checkpoint hợp lệ trong: {model_path}\n"
        "Định dạng hỗ trợ: DCP sharded (.metadata), pytorch_model.bin, *.bin, *.pt, *.pth"
    )


# ─── Save checkpoint ─────────────────────────────────────────────────────────

def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer,
    rank: int,
    cfg,
    epoch: int | None = None,
) -> None:
    """
    Lưu checkpoint model.
    - FSDP model: chỉ rank 0 lưu full state dict.
    - Non-FSDP model: lưu trực tiếp.
    """
    if not getattr(cfg, "save_model", True):
        return

    out_dir = Path(getattr(cfg, "output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_epoch{epoch}" if epoch is not None else ""
    ckpt_path = out_dir / f"model{tag}.bin"

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

        if isinstance(model, FSDP):
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                cpu_state = model.state_dict()
            if rank == 0:
                torch.save(cpu_state, ckpt_path)
                print(f"[model_checkpointing] Đã lưu FSDP checkpoint: {ckpt_path}")
            return
    except Exception:
        pass

    # Non-FSDP
    torch.save(model.state_dict(), ckpt_path)
    print(f"[model_checkpointing] Đã lưu checkpoint: {ckpt_path}")


# ─── Save PEFT / LoRA adapter ────────────────────────────────────────────────

def save_peft_checkpoint(model, output_dir: str, rank: int = 0) -> None:
    """Lưu PEFT/LoRA adapter (chỉ rank 0)."""
    if rank != 0:
        return
    try:
        model.save_pretrained(output_dir)
        print(f"[model_checkpointing] Đã lưu PEFT adapter: {output_dir}")
    except Exception as e:
        print(f"[model_checkpointing] Lỗi lưu PEFT: {e}")
