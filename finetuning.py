"""
finetuning.py — FuidAI fine-tuning entry point.
Hỗ trợ:
  • LLaMA / Mllama + LoRA/PEFT + FSDP + quantization (4bit/8bit) — khi có transformers
  • Custom fuidai model từ mo_hinh.py — khi không có transformers
Chạy qua train.py hoặc trực tiếp: python finetuning.py --model-name ...

KHÔNG yêu cầu llama_cookbook. Tất cả utilities được cung cấp bởi train_utils.py.
"""

from __future__ import annotations

import dataclasses
import os
import random
import sys
from pathlib import Path
from warnings import warn

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# ─── Thêm thư mục hiện tại vào sys.path ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ─── is_xpu_available ────────────────────────────────────────────────────────
try:
    from accelerate.utils import is_xpu_available
except ImportError:
    try:
        from utils import is_xpu_available
    except ImportError:
        def is_xpu_available() -> bool:
            try:
                import intel_extension_for_pytorch  # noqa: F401
                return torch.xpu.is_available()
            except (ImportError, AttributeError):
                return False

# ─── Local configs ────────────────────────────────────────────────────────────
from configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
    check_fsdp_config,
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)
from data import ConcatDataset

# ─── Local training utilities (thay thế llama_cookbook) ──────────────────────
from train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,
    freeze_LLM_only,
    print_model_size,
    print_frozen_model_status,
    setup,
    setup_environ_flags,
    train,
    hsdp_device_mesh,
    get_policies,
    get_preprocessed_dataset,
    get_custom_data_collator,
    apply_fsdp_checkpointing,
)

# ─── fsdp_auto_wrap_policy ───────────────────────────────────────────────────
try:
    from utils import fsdp_auto_wrap_policy
except ImportError:
    import functools as _functools

    def fsdp_auto_wrap_policy(model, transformer_layer_cls: list):
        from torch.distributed.fsdp.wrap import (
            _or_policy,
            lambda_auto_wrap_policy,
            transformer_auto_wrap_policy,
        )

        def _leaf(m):
            return (
                len(list(m.named_children())) == 0
                and getattr(m, "weight", None) is not None
                and m.weight.requires_grad
            )

        return _functools.partial(
            _or_policy,
            policies=[
                _functools.partial(lambda_auto_wrap_policy, lambda_fn=_leaf),
                _functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=set(transformer_layer_cls),
                ),
            ],
        )

# ─── Inference utilities ─────────────────────────────────────────────────────
try:
    from inference import AnyPrecisionAdamW
except Exception as _ie:
    print(f"[finetuning] AnyPrecisionAdamW không khả dụng ({_ie}), fallback AdamW.")
    class AnyPrecisionAdamW(torch.optim.AdamW):  # type: ignore[misc]
        pass

# ─── PyTorch distributed / FSDP ──────────────────────────────────────────────
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

# ─── peft (tuỳ chọn) ─────────────────────────────────────────────────────────
try:
    from peft import get_peft_model, PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    get_peft_model = PeftModel = None
    _PEFT_AVAILABLE = False
    print("[finetuning] peft chưa cài. LoRA/PEFT sẽ bị tắt.")
    print("             Cài: pip install peft")

# ─── transformers (tuỳ chọn) ─────────────────────────────────────────────────
try:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        LlamaForCausalLM,
        MllamaForConditionalGeneration,
    )
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.mllama.modeling_mllama import (
        MllamaCrossAttentionDecoderLayer,
        MllamaSelfAttentionDecoderLayer,
        MllamaVisionEncoderLayer,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    (
        AutoConfig, AutoProcessor, AutoTokenizer, BitsAndBytesConfig,
        LlamaForCausalLM, MllamaForConditionalGeneration,
        LlamaDecoderLayer, MllamaCrossAttentionDecoderLayer,
        MllamaSelfAttentionDecoderLayer, MllamaVisionEncoderLayer,
    ) = (None,) * 11
    _TRANSFORMERS_AVAILABLE = False
    print("[finetuning] transformers chưa cài. Sẽ dùng custom fuidai model.")
    print("             Cài: pip install transformers")

# ─── fire (tuỳ chọn) ─────────────────────────────────────────────────────────
try:
    import fire
    _FIRE_AVAILABLE = True
except ImportError:
    fire = None
    _FIRE_AVAILABLE = False


# ─── WandB setup ─────────────────────────────────────────────────────────────

def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError("Cần cài wandb: pip install wandb")
    from configs import wandb_config as WANDB_CONFIG

    wandb_cfg = WANDB_CONFIG()
    update_config(wandb_cfg, **kwargs)
    init_dict = dataclasses.asdict(wandb_cfg)
    run = wandb.init(**init_dict)
    run.config.update(dataclasses.asdict(train_config) if dataclasses.is_dataclass(train_config) else train_config)
    run.config.update(dataclasses.asdict(fsdp_config) if dataclasses.is_dataclass(fsdp_config) else fsdp_config, allow_val_change=True)
    return run


# ─── Load custom fuidai model ─────────────────────────────────────────────────

def _load_fuidai_model(train_config):
    """
    Tải hoặc khởi tạo custom fuidai model từ mo_hinh.py.
    Cần: vocab.json tại thư mục model_name.
    """
    from mo_hinh import fuidai, TokenizerTV
    from configs import fuidai_model_config

    model_dir = Path(train_config.model_name)
    vocab_path = model_dir / "vocab.json"

    tokenizer = TokenizerTV()

    if vocab_path.exists():
        tokenizer = TokenizerTV.tai(str(vocab_path))
        print(f"[finetuning] Đã tải TokenizerTV: {vocab_path} ({tokenizer.vocab_size} ký tự)")
    else:
        raise FileNotFoundError(
            f"Không tìm thấy vocab.json tại {vocab_path}.\n"
            "Hãy tạo vocab trước hoặc chỉ định đúng đường dẫn model_name."
        )

    mcfg = fuidai_model_config()

    model = fuidai(
        kich_thuoc_tu_vung=tokenizer.vocab_size,
        d_model=mcfg.hidden_size,
        so_lop=mcfg.num_hidden_layers,
        so_dau=mcfg.num_attention_heads,
        kich_thuoc_khoi=train_config.context_length,
        ty_le_bo_qua=mcfg.attention_dropout,
    )

    # Thử nạp checkpoint đã lưu
    ckpt_path = model_dir / "model.bin"
    if ckpt_path.exists():
        try:
            from model_checkpointing import load_sharded_model_single_gpu
            model = load_sharded_model_single_gpu(model, str(model_dir))
            print(f"[finetuning] Đã nạp checkpoint: {model_dir}")
        except Exception as e:
            print(f"[finetuning] Không thể nạp checkpoint ({e}). Khởi tạo model mới.")

    return model, tokenizer


# ─── Main training function ───────────────────────────────────────────────────

def main(**kwargs):
    """Điểm vào chính cho fine-tuning FuidAI."""

    # ── Config ───────────────────────────────────────────────────────────────
    train_config = TRAIN_CONFIG()
    fsdp_config = FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    # Nếu peft không có: tắt use_peft
    if not _PEFT_AVAILABLE and train_config.use_peft:
        print("[finetuning] Tắt use_peft vì peft chưa cài.")
        train_config.use_peft = False

    # ── Seed ─────────────────────────────────────────────────────────────────
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)

    # ── Distributed init ─────────────────────────────────────────────────────
    local_rank = rank = world_size = None
    if train_config.enable_fsdp:
        setup()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # ── WandB ────────────────────────────────────────────────────────────────
    wandb_run = None
    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    # ── Detect model type ─────────────────────────────────────────────────────
    use_hf_model = False
    is_vision = False
    model_path = Path(train_config.model_name)

    if _TRANSFORMERS_AVAILABLE and model_path.exists():
        config_file = model_path / "config.json"
        if config_file.exists():
            use_hf_model = True

    if not use_hf_model and not _TRANSFORMERS_AVAILABLE:
        print("[finetuning] Dùng custom fuidai model (transformers không khả dụng).")
    elif not use_hf_model:
        print(
            f"[finetuning] Không tìm thấy HF config.json tại {model_path}.\n"
            "             Dùng custom fuidai model."
        )

    # ── Quantization config ───────────────────────────────────────────────────
    bnb_config = None
    if use_hf_model and train_config.quantization and train_config.quantization != "none":
        if isinstance(train_config.quantization, bool):
            warn("quantization nên là '4bit'/'8bit', không phải bool.", FutureWarning)
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError("8bit quantization không hỗ trợ với FSDP. Dùng 4bit.")

        quant_cfg = QUANTIZATION_CONFIG()
        update_config(quant_cfg, **kwargs)
        bnb_config = quant_cfg.create_bnb_config(train_config.quantization)

        if train_config.enable_fsdp and train_config.quantization == "4bit":
            try:
                bnb_config.bnb_4bit_quant_storage = bnb_config.bnb_4bit_compute_dtype
            except AttributeError:
                pass  # bitsandbytes cũ không có field này

    # ── Load model ────────────────────────────────────────────────────────────
    processor = None

    if use_hf_model:
        # ── HuggingFace model path ────────────────────────────────────────────
        config = AutoConfig.from_pretrained(train_config.model_name)
        use_cache = False if train_config.enable_fsdp else None

        if config.model_type == "mllama":
            is_vision = True
            model = MllamaForConditionalGeneration.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map=(
                    "auto" if train_config.quantization and not train_config.enable_fsdp else None
                ),
                torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
            )
            processor = AutoProcessor.from_pretrained(
                train_config.model_name
                if train_config.tokenizer_name is None
                else train_config.tokenizer_name
            )
            processor.tokenizer.padding_side = "right"
            model.supports_gradient_checkpointing = True
            model.language_model.supports_gradient_checkpointing = True

        elif config.model_type == "llama":
            is_vision = False
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map=(
                    "auto" if train_config.quantization and not train_config.enable_fsdp else None
                ),
                torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
            )

        else:
            raise ValueError(
                f"Model type '{config.model_type}' không được hỗ trợ trực tiếp.\n"
                "Hỗ trợ: llama, mllama. Với model khác, hãy dùng custom fuidai model."
            )

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.model_name
            if train_config.tokenizer_name is None
            else train_config.tokenizer_name
        )
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
            print("CẢNH BÁO: Resize embedding matrix để khớp tokenizer vocab size.")
            model.resize_token_embeddings(len(tokenizer))

    else:
        # ── Custom fuidai model path ──────────────────────────────────────────
        model, tokenizer = _load_fuidai_model(train_config)
        # fuidai dùng c_attn/c_proj — đảm bảo target_modules đúng nếu PEFT bật
        if train_config.use_peft and "target_modules" not in kwargs:
            kwargs["target_modules"] = ["c_attn", "c_proj"]

    print_model_size(model, train_config, rank or 0)

    # ── Convert sang bf16 nếu cần ────────────────────────────────────────────
    if use_hf_model and train_config.enable_fsdp and fsdp_config.pure_bf16 and not train_config.quantization:
        model.to(torch.bfloat16)

    # ── PEFT / LoRA ──────────────────────────────────────────────────────────
    if train_config.use_peft and _PEFT_AVAILABLE:
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(
                model, train_config.from_peft_checkpoint, is_trainable=True
            )
            peft_config = model.peft_config
        else:
            # Auto-set target_modules đúng theo loại model nếu chưa override
            if "target_modules" not in kwargs:
                kwargs["target_modules"] = (
                    ["q_proj", "k_proj", "v_proj", "o_proj"]
                    if use_hf_model
                    else ["c_attn", "c_proj"]
                )
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        if wandb_run:
            wandb_run.config.update(peft_config)
        model.print_trainable_parameters()

    # ── HSDP device mesh ─────────────────────────────────────────────────────
    hsdp_device_mesh_plan = None
    if (
        train_config.enable_fsdp
        and fsdp_config.hsdp
        and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
    ):
        hsdp_device_mesh_plan = hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print("HSDP device mesh đã sẵn sàng.")

    # ── FSDP setup ───────────────────────────────────────────────────────────
    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)

        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)
            print_frozen_model_status(model, train_config, rank or 0)

        if (
            not train_config.use_peft
            and train_config.freeze_LLM_only
            and use_hf_model
        ):
            freeze_LLM_only(model)
            print_frozen_model_status(model, train_config, rank or 0)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank or 0)

        # Chọn wrap policy theo loại model
        if use_hf_model and is_vision:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(
                model,
                [
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    MllamaVisionEncoderLayer,
                ],
            )
        elif use_hf_model:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
        else:
            # Custom fuidai model — wrap theo KhoiTransformer
            from mo_hinh import KhoiTransformer
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [KhoiTransformer])

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        use_orig_params = train_config.freeze_LLM_only or train_config.use_peft

        model = FSDP(
            model,
            auto_wrap_policy=(
                my_auto_wrapping_policy if train_config.use_peft else wrapping_policy
            ),
            cpu_offload=(
                CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None
            ),
            mixed_precision=(
                mixed_precision_policy if not fsdp_config.pure_bf16 else None
            ),
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(
                (lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
                if train_config.low_cpu_fsdp and rank != 0
                else None
            ),
            use_orig_params=use_orig_params,
        )

        if fsdp_config.fsdp_activation_checkpointing:
            # enable_input_require_grads() và gradient_checkpointing_enable()
            # chỉ tồn tại trên HuggingFace PreTrainedModel, không gọi trên FSDP wrapper.
            # Với custom fuidai model, activation checkpointing được xử lý hoàn toàn
            # bởi apply_fsdp_checkpointing (dùng torch.utils.checkpoint bên dưới).
            if use_hf_model and train_config.use_peft:
                # Cần enable input grads để PEFT + gradient checkpointing hoạt động
                # (phải gọi trên model HF gốc, nhưng lúc này đã bọc trong FSDP)
                # → dùng FSDP module accessor
                try:
                    model.module.enable_input_require_grads()
                except AttributeError:
                    pass

            # Xác định check_fn phù hợp với loại model
            if use_hf_model and is_vision:
                _layer_classes = [cls for cls in [
                    LlamaDecoderLayer,
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    MllamaVisionEncoderLayer,
                ] if cls is not None]
                _check_fn = lambda m: isinstance(m, tuple(_layer_classes))
            elif use_hf_model:
                _check_fn = (
                    (lambda m: isinstance(m, LlamaDecoderLayer))
                    if LlamaDecoderLayer is not None
                    else None
                )
            else:
                from mo_hinh import KhoiTransformer
                _check_fn = lambda m: isinstance(m, KhoiTransformer)

            apply_fsdp_checkpointing(model, check_fn=_check_fn)

    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_processer = processor if (use_hf_model and is_vision) else tokenizer

    dataset_train = get_preprocessed_dataset(dataset_processer, dataset_config, split="train")
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(dataset_processer, dataset_config, split="test")
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        if use_hf_model and is_vision:
            raise ValueError("Packing không hỗ trợ vision dataset.")
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, dataset_processer, "train")

    custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
    if custom_data_collator:
        train_dl_kwargs["collate_fn"] = custom_data_collator

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Số batch train: {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            if use_hf_model and is_vision:
                raise ValueError("Packing không hỗ trợ vision dataset.")
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, dataset_processer, "val")
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        if len(eval_dataloader) == 0:
            raise ValueError(
                f"Validation set quá nhỏ, không thể tạo batch. "
                f"Hãy tăng dữ liệu hoặc tắt run_validation."
            )
        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Số batch validation: {len(eval_dataloader)}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # ── Train ─────────────────────────────────────────────────────────────────
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
    )

    if not train_config.enable_fsdp or rank == 0:
        for k, v in results.items():
            print(f"  {k}: {v}")
        if train_config.use_wandb and wandb_run:
            for k, v in results.items():
                if not isinstance(v, list):
                    wandb_run.summary[k] = v


if __name__ == "__main__":
    if _FIRE_AVAILABLE:
        import fire as _fire
        _fire.Fire(main)
    else:
        # Fallback: parse sys.argv thủ công
        import argparse
        parser = argparse.ArgumentParser(description="FuidAI Finetuning")
        parser.add_argument("--model-name", required=True)
        parser.add_argument("--dataset", default="fuidai_dataset")
        parser.add_argument("--num-epochs", type=int, default=3)
        parser.add_argument("--batch-size-training", type=int, default=4)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--enable-fsdp", action="store_true", default=False)
        parser.add_argument("--use-peft", action="store_true", default=False)
        parser.add_argument("--quantization", default="none")
        parser.add_argument("--output-dir", default="output")
        args = parser.parse_args()
        main(**{k.replace("-", "_"): v for k, v in vars(args).items()})