"""
train_utils.py — FuidAI training utilities.
Thay thế hoàn toàn llama_cookbook.utils.* — không cần cài llama-cookbook.
"""
from __future__ import annotations

import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


# ─── Distributed setup ───────────────────────────────────────────────────────

def setup():
    """Khởi tạo torch.distributed process group."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)


def setup_environ_flags(rank: int):
    """Đặt biến môi trường cho distributed training."""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if rank == 0:
        print("--> Distributed environment ready.")


# ─── GPU cache ───────────────────────────────────────────────────────────────

def clear_gpu_cache(rank: Optional[int] = None):
    """Xóa GPU cache và chạy GC."""
    if rank == 0 or rank is None:
        print("Clearing GPU cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ─── Model helpers ───────────────────────────────────────────────────────────

def print_model_size(model, cfg, rank: int = 0):
    if rank != 0:
        return
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n[Model] Tổng params: {total / 1e6:.2f}M | "
        f"Trainable: {trainable / 1e6:.2f}M ({100 * trainable / max(total, 1):.1f}%)\n"
    )


def print_frozen_model_status(model, cfg, rank: int = 0):
    if rank != 0:
        return
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Frozen: {frozen / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M")


def freeze_transformer_layers(model, num_layer: int):
    """Đóng băng `num_layer` lớp transformer đầu tiên (HF-style model)."""
    try:
        layers = model.model.layers
    except AttributeError:
        try:
            layers = model.cac_khoi  # fuidai custom model
        except AttributeError:
            print("[freeze] Không tìm được danh sách layers, bỏ qua freeze.")
            return
    for i, layer in enumerate(layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False
    print(f"[freeze] Đã đóng băng {num_layer} lớp đầu tiên.")


def freeze_LLM_only(model):
    """Đóng băng LLM, chỉ train vision encoder (dành cho mllama)."""
    for name, param in model.named_parameters():
        if "vision_model" not in name and "multi_modal_projector" not in name:
            param.requires_grad = False
    print("[freeze] Đã đóng băng LLM (chỉ train vision encoder).")


# ─── FSDP activation checkpointing ──────────────────────────────────────────

def apply_fsdp_checkpointing(model: torch.nn.Module, check_fn=None) -> None:
    """
    Áp dụng activation checkpointing cho FSDP model.
    Canonical version — inference.py và finetuning.py đều có thể import từ đây.

    Args:
        model:    FSDP-wrapped model.
        check_fn: Callable(module) → bool. Nếu None, tự detect:
                  - HF LLaMA  → LlamaDecoderLayer
                  - fuidai    → KhoiTransformer
    """
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from functools import partial as _partial
    except ImportError:
        print("[train_utils] checkpoint_wrapper không khả dụng. Bỏ qua activation checkpointing.")
        return

    if check_fn is None:
        # Thử detect tự động
        try:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            check_fn = lambda m: isinstance(m, LlamaDecoderLayer)
        except ImportError:
            try:
                from mo_hinh import KhoiTransformer
                check_fn = lambda m: isinstance(m, KhoiTransformer)
            except ImportError:
                print("[train_utils] Không tìm được layer class để checkpointing.")
                return

    print("--> Đang áp dụng activation checkpointing (FSDP)...")
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=_partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=check_fn,
    )


# ─── FSDP helpers ────────────────────────────────────────────────────────────

def hsdp_device_mesh(replica_group_size: int, sharding_group_size: int):
    """Tạo HSDP device mesh cho FSDP hybrid sharding."""
    try:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh(
            "cuda",
            (replica_group_size, sharding_group_size),
            mesh_dim_names=["replicate", "shard"],
        )
        return mesh
    except Exception as e:
        print(f"[hsdp_device_mesh] Lỗi tạo device mesh: {e}")
        return None


def get_policies(fsdp_cfg, rank: int):
    """
    Trả về (mixed_precision_policy, wrapping_policy) cho FSDP.
    """
    import functools
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    # Mixed precision policy
    mp_policy = None
    if fsdp_cfg.pure_bf16:
        mp_policy = None  # pure_bf16 không dùng MixedPrecision wrapper
    elif fsdp_cfg.mixed_precision and fsdp_cfg.use_fp16:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif fsdp_cfg.mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

    # Wrapping policy mặc định theo kích thước tham số
    wrapping_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=int(1e6)
    )

    return mp_policy, wrapping_policy


# ─── Dataset utils ───────────────────────────────────────────────────────────

def get_preprocessed_dataset(tokenizer, dataset_config, split: str = "train"):
    """
    Tải và trả về dataset theo dataset_config.dataset.
    Ưu tiên: fuidai_dataset → alpaca_dataset → custom_dataset → HF datasets.
    """
    from data import FuidAIDataset, InstructionDataset, get_custom_dataset

    name = getattr(dataset_config, "dataset", "fuidai_dataset")

    if name in ("fuidai_dataset", "fuidai_chat_dataset"):
        partition = "train" if split in ("train", "training") else "val"
        return FuidAIDataset(dataset_config, tokenizer, partition=partition)

    elif name == "alpaca_dataset":
        partition = "train" if split in ("train", "training") else "val"
        return InstructionDataset(dataset_config, tokenizer, partition=partition)

    elif name == "custom_dataset":
        return get_custom_dataset(dataset_config, tokenizer, split=split)

    elif name == "samsum_dataset":
        # Cần HF datasets
        try:
            from data import get_preprocessed_samsum
            return get_preprocessed_samsum(dataset_config, tokenizer, split)
        except Exception as e:
            raise RuntimeError(f"Không thể tải samsum dataset: {e}")

    else:
        # Fallback: HF datasets.load_dataset
        try:
            import datasets as hf_datasets
            data_path = getattr(dataset_config, "data_path", None)
            if data_path and Path(str(data_path)).exists():
                ds = hf_datasets.load_dataset("json", data_files=str(data_path), split=split)
            else:
                ds = hf_datasets.load_dataset(name, split=split)
            return ds
        except Exception as e:
            raise ValueError(
                f"Không thể tải dataset '{name}': {e}\n"
                "Kiểm tra lại dataset trong configs.py."
            )


def get_custom_data_collator(tokenizer, dataset_config):
    """Trả về custom collator nếu có, ngược lại None."""
    name = getattr(dataset_config, "dataset", "")
    if name == "custom_dataset":
        from data import get_data_collator
        try:
            return get_data_collator(tokenizer, dataset_config)
        except Exception:
            return None
    return None


# ─── Training loop ───────────────────────────────────────────────────────────

def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    scheduler,
    gradient_accumulation_steps: int,
    train_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
    wandb_run=None,
):
    """
    Main training loop — hỗ trợ single GPU và FSDP multi-GPU.
    Tương thích với cả HuggingFace model và custom fuidai model.
    """
    use_fsdp = fsdp_config is not None and getattr(train_config, "enable_fsdp", False)
    is_rank0 = rank is None or rank == 0

    # ── Device ───────────────────────────────────────────────────────────────
    if local_rank is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── FP16 scaler ──────────────────────────────────────────────────────────
    use_fp16 = getattr(train_config, "use_fp16", False)
    # torch.cuda.amp.GradScaler deprecated trong PyTorch >= 2.4
    try:
        scaler = torch.amp.GradScaler("cuda") if use_fp16 and torch.cuda.is_available() else None
    except TypeError:
        scaler = torch.cuda.amp.GradScaler() if use_fp16 and torch.cuda.is_available() else None

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path(getattr(train_config, "output_dir", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics: dict = {"train_loss": [], "val_loss": [], "epoch": []}
    best_val_loss = float("inf")

    num_epochs = getattr(train_config, "num_epochs", 3)
    grad_clip = getattr(train_config, "gradient_clipping", True)
    grad_clip_thresh = getattr(train_config, "gradient_clipping_threshold", 1.0)
    max_train_step = getattr(train_config, "max_train_step", 0)
    run_validation = getattr(train_config, "run_validation", True)

    if is_rank0:
        print(f"\n{'='*60}")
        print(f"  Bắt đầu training: {num_epochs} epochs")
        print(f"  Device: {device} | FP16: {use_fp16} | FSDP: {use_fsdp}")
        print(f"  Grad accum steps: {gradient_accumulation_steps}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0

        # Distributed sampler epoch
        if use_fsdp and hasattr(train_dataloader, "sampler"):
            sampler = train_dataloader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        optimizer.zero_grad()

        # Progress bar
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(
                total=len(train_dataloader) // gradient_accumulation_steps,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                disable=not is_rank0,
            )
        except ImportError:
            pbar = None

        for step, batch in enumerate(train_dataloader):
            # Max step guard
            if max_train_step > 0:
                global_step = epoch * len(train_dataloader) + step
                if global_step >= max_train_step:
                    break

            # Move to device
            batch = _batch_to_device(batch, device)

            # Autocast context
            ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if use_fp16 and torch.cuda.is_available()
                else nullcontext()
            )

            # Forward pass
            with ctx:
                loss = _forward(model, batch)
                loss = loss / gradient_accumulation_steps

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps
            n_steps += 1

            # Optimizer step
            if (step + 1) % gradient_accumulation_steps == 0:
                if grad_clip:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if use_fsdp:
                        model.clip_grad_norm_(grad_clip_thresh)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip_thresh
                        )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

                if pbar is not None:
                    pbar.update(1)
                    try:
                        lr_now = scheduler.get_last_lr()[0]
                    except Exception:
                        lr_now = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{total_loss / n_steps:.4f}",
                        lr=f"{lr_now:.2e}",
                    )

        if pbar is not None:
            pbar.close()

        scheduler.step()

        avg_train_loss = total_loss / max(n_steps, 1)
        metrics["train_loss"].append(avg_train_loss)
        metrics["epoch"].append(epoch + 1)

        if is_rank0:
            print(f"\n[Epoch {epoch+1}] Train loss: {avg_train_loss:.4f}")

        # ── Validation ───────────────────────────────────────────────────────
        val_loss = None
        if eval_dataloader is not None and run_validation:
            val_loss = _evaluate(model, eval_dataloader, device, train_config)
            metrics["val_loss"].append(val_loss)

            if is_rank0:
                print(f"[Epoch {epoch+1}] Val loss:   {val_loss:.4f}")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if is_rank0:
                    _save_checkpoint(model, optimizer, rank, train_config, epoch + 1)

        else:
            if is_rank0:
                _save_checkpoint(model, optimizer, rank, train_config, epoch + 1)

        # ── WandB ────────────────────────────────────────────────────────────
        if wandb_run is not None and is_rank0:
            log = {"epoch": epoch + 1, "train_loss": avg_train_loss}
            if val_loss is not None:
                log["val_loss"] = val_loss
            wandb_run.log(log)

        # ── Sync ─────────────────────────────────────────────────────────────
        if dist.is_initialized():
            dist.barrier()

    # ── Final ─────────────────────────────────────────────────────────────────
    if is_rank0:
        print(f"\n{'='*60}")
        print(f"  Training hoàn tất! Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoint tại: {out_dir}")
        print(f"{'='*60}\n")

        if getattr(train_config, "save_metrics", False):
            mp = out_dir / "train_metrics.json"
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"Đã lưu metrics: {mp}")

    return metrics


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _batch_to_device(batch, device: torch.device) -> dict:
    """Chuyển tất cả tensor trong batch sang device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _forward(model, batch: dict) -> torch.Tensor:
    """
    Forward pass linh hoạt — hỗ trợ:
    - HuggingFace model: outputs.loss
    - fuidai custom model: (logits, loss)
    - batch có 'labels' key hoặc không
    """
    if "labels" in batch:
        # HuggingFace-style: model(**batch) → outputs với .loss
        try:
            outputs = model(**batch)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return outputs.loss
            # Fallback: outputs là tuple (logits, loss)
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                return outputs[1]
            raise ValueError("Model không trả về loss. Kiểm tra forward() của model.")
        except TypeError:
            # fuidai model: forward(ids, nhan)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            _, loss = model(input_ids, labels)
            return loss
    else:
        # Không có labels: tự tạo từ input_ids (causal LM shift)
        input_ids = batch["input_ids"]
        labels = input_ids[:, 1:].contiguous()
        input_ids_in = input_ids[:, :-1].contiguous()
        try:
            _, loss = model(input_ids_in, labels)
        except TypeError:
            outputs = model(input_ids=input_ids_in, labels=labels)
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                loss = outputs[1]
        return loss


def _evaluate(model, eval_dataloader, device, train_config) -> float:
    """Đánh giá model trên validation set, trả về average loss."""
    model.eval()
    total_loss = 0.0
    n = 0
    max_eval_step = getattr(train_config, "max_eval_step", 0)
    use_fp16 = getattr(train_config, "use_fp16", False)

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if max_eval_step > 0 and step >= max_eval_step:
                break
            batch = _batch_to_device(batch, device)
            ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if use_fp16 and torch.cuda.is_available()
                else nullcontext()
            )
            with ctx:
                loss = _forward(model, batch)
            total_loss += loss.item()
            n += 1

    model.train()
    return total_loss / max(n, 1)


def _save_checkpoint(model, optimizer, rank, cfg, epoch=None):
    """Lưu checkpoint qua model_checkpointing.py."""
    try:
        from model_checkpointing import save_model_checkpoint
        save_model_checkpoint(model, optimizer, rank or 0, cfg, epoch)
    except Exception as e:
        print(f"[train_utils] Không thể lưu checkpoint: {e}")