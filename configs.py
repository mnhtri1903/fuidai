from dataclasses import dataclass, field, fields
from typing import List, Optional
import torch

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

FUIDAI_MODEL_NAME = "fuidai"
FUIDAI_VERSION = "0.05"
FUIDAI_PARAMS = "300M"
FUIDAI_MODEL_ID = f"{FUIDAI_MODEL_NAME}-{FUIDAI_VERSION}-{FUIDAI_PARAMS}"


# ─── Dataset configs ──────────────────────────────────────────────────────────

@dataclass
class fuidai_dataset:
    dataset: str = "fuidai_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = "data/fuidai_sft/train.jsonl"
    val_path: str = "data/fuidai_sft/val.jsonl"


@dataclass
class fuidai_chat_dataset:
    dataset: str = "fuidai_chat_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = "data/fuidai_sft/train.jsonl"
    val_path: str = "data/fuidai_sft/val.jsonl"
    prompt_style: str = "alpaca"
    mask_prompt: bool = True


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_cookbook/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_cookbook/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_cookbook/datasets/alpaca_data.json"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "getting-started/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""


@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"


# ─── FSDP config ──────────────────────────────────────────────────────────────

@dataclass
class fsdp_config:
    mixed_precision: bool = True
    use_fp16: bool = True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    hsdp: bool = False
    sharding_group_size: int = 0
    replica_group_size: int = 0
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"


# ─── LoRA / PEFT configs ──────────────────────────────────────────────────────

@dataclass
class lora_config:
    r: int = 16
    lora_alpha: int = 32
    # fuidai custom model dùng tên GPT-style: c_attn (QKV fused), c_proj (output proj)
    # HuggingFace LLaMA dùng: q_proj, k_proj, v_proj, o_proj
    # Mặc định là fuidai — override khi dùng HF model
    target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False


@dataclass
class fuid_adapter_config:
    adapter_len: int = 10
    adapter_layers: int = 30
    task_type: str = "CAUSAL_LM"


@dataclass
class prefix_config:
    num_virtual_tokens: int = 30
    task_type: str = "CAUSAL_LM"


# ─── Quantization config ──────────────────────────────────────────────────────

@dataclass
class quantization_config:
    quant_type: str = "nf4"
    compute_dtype: str = "float16"
    use_double_quant: bool = True

    def create_bnb_config(self, quantization: str):
        """Trả về BitsAndBytesConfig (yêu cầu transformers + bitsandbytes)."""
        if quantization not in {"4bit", "8bit"}:
            raise ValueError("quantization phải là '4bit' hoặc '8bit'")
        try:
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.quant_type,
                    bnb_4bit_use_double_quant=self.use_double_quant,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            return BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            # Fallback khi không cài transformers
            if quantization == "4bit":
                return {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": self.quant_type,
                    "bnb_4bit_use_double_quant": self.use_double_quant,
                }
            return {"load_in_8bit": True}


# ─── Train config ─────────────────────────────────────────────────────────────

@dataclass
class train_config:
    model_name: str = f"PATH/to/{FUIDAI_MODEL_ID}"
    tokenizer_name: Optional[str] = None
    enable_fsdp: bool = True
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    batching_strategy: str = "packing"
    context_length: int = 2048
    gradient_accumulation_steps: int = 4
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 3
    max_train_step: int = 0
    max_eval_step: int = 0
    num_workers_dataloader: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = True
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset: str = "fuidai_dataset"
    peft_method: str = "lora"
    use_peft: bool = True
    from_peft_checkpoint: str = ""
    output_dir: str = f"output/{FUIDAI_MODEL_ID}/peft"
    freeze_layers: bool = False
    num_freeze_layers: int = 0
    freeze_LLM_only: bool = False
    quantization: str = "4bit"
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str = f"output/{FUIDAI_MODEL_ID}/fsdp"
    dist_checkpoint_folder: str = "fine-tuned"
    save_optimizer: bool = False
    use_fast_kernels: bool = True
    use_wandb: bool = False
    save_metrics: bool = True
    flop_counter_start: int = 3
    use_profiler: bool = False
    profiler_dir: str = f"output/{FUIDAI_MODEL_ID}/profiler"


# ─── Misc configs ─────────────────────────────────────────────────────────────

@dataclass
class t4_dual_gpu_config:
    num_gpus: int = 2
    gpu_type: str = "T4"
    vram_per_gpu_gb: int = 16
    total_vram_gb: int = 32
    nccl_backend: str = "nccl"
    master_port: int = 29500
    master_addr: str = "localhost"
    fp16: bool = True
    bf16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = True
    pin_memory: bool = True
    num_workers: int = 2
    prefetch_factor: int = 2
    persistent_workers: bool = True
    compile_model: bool = False
    find_unused_parameters: bool = False


@dataclass
class fuidai_model_config:
    model_id: str = FUIDAI_MODEL_ID
    model_name: str = FUIDAI_MODEL_NAME
    version: str = FUIDAI_VERSION
    num_parameters: str = FUIDAI_PARAMS
    # ── ~0.5B: 12 × d² × L ≈ 12 × 1280² × 26 ≈ 511M params ──
    hidden_size: int = 1280
    num_hidden_layers: int = 26
    num_attention_heads: int = 16           # d_head = 1280/16 = 80
    num_key_value_heads: int = 16           # fuidai dùng full attention (không GQA)
    intermediate_size: int = 5120           # = 4 × hidden_size (khớp MangDaTang)
    max_position_embeddings: int = 4096
    # vocab_size này chỉ mang tính tài liệu tham khảo cho HF-style config.
    # Model thực tế dùng tokenizer.vocab_size (char-level, ~5K ký tự).
    vocab_size: int = 32000
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    model_type: str = "fuidai"
    torch_dtype: str = "float16"
    tie_word_embeddings: bool = False
    use_cache: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0


@dataclass
class wandb_config:
    project: str = f"{FUIDAI_MODEL_NAME}-{FUIDAI_VERSION}"
    entity: Optional[str] = None
    job_type: Optional[str] = "finetune"
    tags: Optional[List[str]] = field(
        default_factory=lambda: [FUIDAI_MODEL_ID, "lora", "t4-x2", "fsdp"]
    )
    group: Optional[str] = None
    notes: Optional[str] = f"FuidAI {FUIDAI_VERSION} {FUIDAI_PARAMS} LoRA finetune on T4 x2"
    mode: Optional[str] = None


# ─── Helper functions ─────────────────────────────────────────────────────────

_DATASET_MAP = {
    "fuidai_dataset": fuidai_dataset,
    "fuidai_chat_dataset": fuidai_chat_dataset,
    "samsum_dataset": samsum_dataset,
    "grammar_dataset": grammar_dataset,
    "alpaca_dataset": alpaca_dataset,
    "custom_dataset": custom_dataset,
    "llamaguard_toxicchat_dataset": llamaguard_toxicchat_dataset,
}

_PEFT_MAP = {
    "lora": lora_config,
    "llama_adapter": fuid_adapter_config,
    "prefix": prefix_config,
}


def update_config(config, **kwargs):
    """Cập nhật một hoặc nhiều dataclass config từ kwargs.
    
    Nhận vào: single dataclass, tuple, hoặc list dataclass.
    Chỉ cập nhật các field tồn tại trong dataclass — bỏ qua kwargs không khớp.
    """
    if isinstance(config, (tuple, list)):
        for cfg in config:
            update_config(cfg, **kwargs)
        return
    for f in fields(config):
        if f.name in kwargs:
            setattr(config, f.name, kwargs[f.name])


def check_fsdp_config(fsdp_cfg: fsdp_config):
    """Kiểm tra tính hợp lệ của FSDP config."""
    if fsdp_cfg.hsdp:
        if fsdp_cfg.sharding_group_size == 0 or fsdp_cfg.replica_group_size == 0:
            raise ValueError(
                "Khi bật HSDP, phải đặt sharding_group_size và replica_group_size > 0."
            )
    if fsdp_cfg.pure_bf16 and fsdp_cfg.use_fp16:
        raise ValueError("Không thể bật đồng thời pure_bf16 và use_fp16.")


def generate_dataset_config(train_cfg: train_config, kwargs: dict):
    """Tạo dataset config tương ứng với train_config.dataset."""
    name = train_cfg.dataset
    if name not in _DATASET_MAP:
        raise ValueError(
            f"Dataset '{name}' chưa được đăng ký. "
            f"Các giá trị hợp lệ: {list(_DATASET_MAP)}"
        )
    cfg = _DATASET_MAP[name]()
    update_config(cfg, **kwargs)
    return cfg


def generate_peft_config(train_cfg: train_config, kwargs: dict):
    """Tạo PEFT config (peft.LoraConfig, v.v.) từ train_config.peft_method."""
    from peft import LoraConfig, PrefixTuningConfig

    method = train_cfg.peft_method
    if method == "lora":
        cfg = lora_config()
        update_config(cfg, **kwargs)
        # Tự động chọn target_modules đúng nếu user không override:
        # HF LLaMA dùng q_proj/k_proj/v_proj/o_proj
        # fuidai custom model dùng c_attn/c_proj
        # → giữ nguyên giá trị trong cfg (mặc định đã là fuidai style)
        return LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            bias=cfg.bias,
            task_type=cfg.task_type,
            lora_dropout=cfg.lora_dropout,
            inference_mode=cfg.inference_mode,
        )
    elif method == "prefix":
        cfg = prefix_config()
        update_config(cfg, **kwargs)
        return PrefixTuningConfig(
            num_virtual_tokens=cfg.num_virtual_tokens,
            task_type=cfg.task_type,
        )
    elif method == "llama_adapter":
        # Dùng LoRA với adapter_len làm r
        cfg = fuid_adapter_config()
        update_config(cfg, **kwargs)
        return LoraConfig(r=cfg.adapter_len, task_type=cfg.task_type)
    else:
        raise ValueError(
            f"peft_method '{method}' không hỗ trợ. Dùng: lora, prefix, llama_adapter."
        )


def get_dataloader_kwargs(train_cfg: train_config, dataset, tokenizer, mode: str) -> dict:
    """Trả về kwargs cho torch.utils.data.DataLoader."""
    import torch
    import torch.distributed as dist
    from data import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler

    batch_size = train_cfg.batch_size_training if mode == "train" else train_cfg.val_batch_size
    kwargs: dict = {}

    if train_cfg.batching_strategy == "padding":
        if train_cfg.enable_fsdp and dist.is_initialized():
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=(mode == "train"),
            )
        else:
            kwargs["batch_sampler"] = LengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                shuffle=(mode == "train"),
            )
    elif train_cfg.batching_strategy == "packing":
        if train_cfg.enable_fsdp and dist.is_initialized():
            kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=(mode == "train"),
                seed=train_cfg.seed,
            )
        else:
            kwargs["shuffle"] = (mode == "train")
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
    else:
        raise ValueError(
            f"batching_strategy '{train_cfg.batching_strategy}' không hợp lệ. "
            "Dùng 'padding' hoặc 'packing'."
        )

    return kwargs