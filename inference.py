"""
inference.py — FuidAI inference & model loading utilities.
Hỗ trợ: FSDP checkpoint consolidation, OpenAI API wrapper, FSDP helpers.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from string import Template
from typing import Callable, List, Union
from warnings import warn

try:
    import fire
    _FIRE_AVAILABLE = True
except ImportError:
    fire = None
    _FIRE_AVAILABLE = False

import torch

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    yaml = None
    _YAML_AVAILABLE = False

# typing_extensions.override (tuỳ chọn)
try:
    from typing_extensions import override
except ImportError:
    def override(f):
        return f

from configs import quantization_config as QUANT_CONFIG
from configs import update_config

# ─── openai (tuỳ chọn) ───────────────────────────────────────────────────────
try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

# ─── peft (tuỳ chọn) ─────────────────────────────────────────────────────────
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[assignment,misc]

# ─── transformers (tuỳ chọn) ─────────────────────────────────────────────────
try:
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        MllamaProcessor,
        AutoModelForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        MllamaConfig,
        MllamaForConditionalGeneration,
    )
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.mllama.modeling_mllama import (
        MllamaSelfAttentionDecoderLayer,
        MllamaCrossAttentionDecoderLayer,
        MllamaVisionEncoderLayer,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoConfig = AutoTokenizer = MllamaProcessor = AutoModelForCausalLM = None  # type: ignore
    LlamaConfig = LlamaForCausalLM = MllamaConfig = MllamaForConditionalGeneration = None  # type: ignore
    LlamaDecoderLayer = None  # type: ignore
    MllamaSelfAttentionDecoderLayer = MllamaCrossAttentionDecoderLayer = MllamaVisionEncoderLayer = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

# ─── activation checkpointing (tuỳ chọn) ────────────────────────────────────
try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    _CHECKPOINT_WRAPPER_AVAILABLE = True
except ImportError:
    checkpoint_wrapper = CheckpointImpl = apply_activation_checkpointing = None  # type: ignore
    _CHECKPOINT_WRAPPER_AVAILABLE = False

from torch.optim.optimizer import Optimizer
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

# ─── model_checkpointing: thử import local, fallback inline ──────────────────
# Thêm thư mục hiện tại vào sys.path để tìm model_checkpointing.py cục bộ
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from model_checkpointing import load_sharded_model_single_gpu
except ImportError:
    # Fallback inline nếu model_checkpointing.py không tồn tại
    def load_sharded_model_single_gpu(model: torch.nn.Module, model_path: str) -> torch.nn.Module:  # type: ignore[misc]
        """Fallback: nạp checkpoint đơn giản từ *.bin / *.pt."""
        from pathlib import Path
        model_path_obj = Path(str(model_path))

        # Thử DCP format
        metadata = model_path_obj / ".metadata"
        if metadata.exists():
            try:
                from torch.distributed.checkpoint import FileSystemReader
                from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
                state_dict = {"model": model.state_dict()}
                dcp_load(state_dict=state_dict, storage_reader=FileSystemReader(str(model_path)))
                model.load_state_dict(state_dict["model"])
                return model
            except Exception:
                pass

        # Thử file thông thường
        for pat in ("pytorch_model.bin", "*.bin", "*.pt", "*.pth"):
            found = list(model_path_obj.glob(pat))
            if found:
                ckpt = torch.load(found[0], map_location="cpu")
                state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state, strict=False)
                print(f"[inference] Đã nạp checkpoint từ: {found[0]}")
                return model

        raise FileNotFoundError(f"Không tìm thấy checkpoint trong: {model_path}")


# ─── Đọc file dialogs ────────────────────────────────────────────────────────

def read_dialogs_from_file(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── main: consolidate FSDP checkpoint ───────────────────────────────────────

def main(
    fsdp_checkpoint_path: str = "",
    consolidated_model_path: str = "",
    HF_model_path_or_name: str = "",
):
    if AutoConfig is None:
        raise ImportError("Cần cài transformers: pip install transformers")

    try:
        train_params_path = os.path.join(fsdp_checkpoint_path, "train_params.yaml")
        with open(train_params_path, "r") as f:
            if yaml is not None:
                data = yaml.safe_load(f)
            else:
                import json, re
                text = f.read()
                # fallback: parse simple key: value
                data = {}
                for m in re.finditer(r'^(\w+):\s*(.+)$', text, re.MULTILINE):
                    data[m.group(1)] = m.group(2).strip()
            HF_model_path_or_name = data.get("model_name")
            print(f"Tên model: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"Không tìm thấy train_params.yaml trong {fsdp_checkpoint_path}.")
        HF_model_path_or_name = input("Nhập tên model: ")
        print(f"Model đã chọn: {HF_model_path_or_name}")
    except Exception as e:
        print(f"Lỗi: {e}")

    model_def = load_llama_from_config(HF_model_path_or_name)
    print("Đã tải config model.")
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("Đã nạp FSDP checkpoint vào model.")

    config = AutoConfig.from_pretrained(HF_model_path_or_name)
    if config.model_type == "mllama":
        processor = MllamaProcessor.from_pretrained(HF_model_path_or_name)
        processor.save_pretrained(consolidated_model_path)
        print(f"Đã lưu mllama processor tại: {consolidated_model_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
        tokenizer.save_pretrained(consolidated_model_path)
        print(f"Đã lưu tokenizer tại: {consolidated_model_path}")

    model.save_pretrained(consolidated_model_path)
    print(f"Đã lưu model HF tại: {consolidated_model_path}")


# ─── LLM constants ───────────────────────────────────────────────────────────

NUM_LLM_RETRIES = 10
MAX_TOKENS = 1000
TEMPERATURE = 0.1
TOP_P = 0.9
LOG = logging.getLogger(__name__)


# ─── LLM abstract base ───────────────────────────────────────────────────────

class LLM(ABC):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        if model not in self.valid_models():
            LOG.warning(
                f"Cảnh báo: {model} không có trong danh sách hỗ trợ của {type(self).__name__}."
            )
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def query(self, prompt: str) -> str: ...

    def query_with_system_prompt(self, system_prompt: str, prompt: str) -> str:
        return self.query(system_prompt + "\n" + prompt)

    def _query_with_retries(self, func, *args, retries=NUM_LLM_RETRIES, backoff_factor=0.5):
        last_exc = None
        for retry in range(retries):
            try:
                return func(*args)
            except Exception as e:
                last_exc = e
                sleep_time = backoff_factor * (2 ** retry)
                time.sleep(sleep_time)
                LOG.debug(f"Truy vấn thất bại: {e}. Thử lại sau {sleep_time}s...")
        raise RuntimeError(f"Không kết nối được LLM sau {retries} lần: {last_exc}")

    def valid_models(self) -> list[str]:
        return []


# ─── OpenAI wrapper ──────────────────────────────────────────────────────────

class OPENAI(LLM):
    def __init__(self, model: str, api_key: str) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError("Cần cài openai: pip install openai")
        super().__init__(model, api_key)
        self.client = openai.OpenAI(api_key=api_key)  # type: ignore[union-attr]

    @override
    def query(self, prompt: str) -> str:
        level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
        )
        logging.getLogger().setLevel(level)
        return response.choices[0].message.content

    def valid_models(self) -> list[str]:
        return ["gpt-3.5-turbo", "gpt-4"]


# ─── Load model ──────────────────────────────────────────────────────────────

def load_model(model_name: str, quantization, use_fast_kernels: bool, **kwargs):
    if AutoModelForCausalLM is None:
        raise ImportError("Cần cài transformers: pip install transformers")

    if isinstance(quantization, bool):
        warn("Dùng '4bit' hoặc '8bit' thay cho boolean.", FutureWarning)
        quantization = "8bit"

    bnb_config = None
    if quantization:
        quant_config = QUANT_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(quantization)

    conf: dict = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "return_dict": True,
    }
    if bnb_config:
        conf["quantization_config"] = bnb_config
    if use_fast_kernels:
        conf["attn_implementation"] = "sdpa"

    return AutoModelForCausalLM.from_pretrained(model_name, **conf)


def load_llama_from_config(config_path: str):
    if AutoConfig is None:
        raise ImportError("Cần cài transformers: pip install transformers")
    config = AutoConfig.from_pretrained(config_path)
    if config.model_type == "mllama":
        return MllamaForConditionalGeneration(config=config)
    elif config.model_type == "llama":
        return LlamaForCausalLM(config=config)
    raise ValueError(
        f"Loại model không hỗ trợ: {config.model_type}. Dùng 'llama' hoặc 'mllama'."
    )


# ─── LlamaGuard / agent enums ────────────────────────────────────────────────

class LlamaGuardVersion(Enum):
    LLAMA_GUARD_1 = "Llama Guard 1"
    LLAMA_GUARD_2 = "Llama Guard 2"
    LLAMA_GUARD_3 = "Llama Guard 3"


class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"


@dataclass
class ConversationTurn:
    message: str
    agent_type: AgentType


# ─── FSDP activation checkpointing ──────────────────────────────────────────

def apply_fsdp_checkpointing(model: torch.nn.Module, check_fn=None) -> None:
    """
    Áp dụng activation checkpointing cho FSDP model.

    Args:
        model:    FSDP-wrapped model.
        check_fn: Callable(module) → bool xác định module nào được wrap.
                  Nếu None, tự detect theo loại model:
                  - HF LLaMA  → LlamaDecoderLayer
                  - fuidai    → KhoiTransformer
    """
    if not _CHECKPOINT_WRAPPER_AVAILABLE:
        print("[inference] Cảnh báo: checkpoint_wrapper không khả dụng. Bỏ qua activation checkpointing.")
        return

    if check_fn is None:
        if LlamaDecoderLayer is not None:
            # HF LLaMA / Mllama
            _layer_classes = [LlamaDecoderLayer]
            for cls in [
                MllamaSelfAttentionDecoderLayer,
                MllamaCrossAttentionDecoderLayer,
                MllamaVisionEncoderLayer,
            ]:
                if cls is not None:
                    _layer_classes.append(cls)
            check_fn = lambda m: isinstance(m, tuple(_layer_classes))
        else:
            # Custom fuidai model
            try:
                from mo_hinh import KhoiTransformer
                check_fn = lambda m: isinstance(m, KhoiTransformer)
            except ImportError:
                print("[inference] Cảnh báo: Không tìm được layer class để checkpointing.")
                return

    print("--> Đang áp dụng activation checkpointing (FSDP)...")
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=check_fn,
    )


# ─── Custom AdamW ─────────────────────────────────────────────────────────────

class AnyPrecisionAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"] = torch.zeros_like(
                        p, dtype=group.get("momentum_dtype", torch.bfloat16)
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, dtype=group.get("variance_dtype", torch.bfloat16)
                    )
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                if group["weight_decay"]:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                step_size = group["lr"] / (1 - beta1 ** state["step"])
                denom = (exp_avg_sq.sqrt() / (1 - beta2 ** state["step"]) ** 0.5).add_(
                    group["eps"]
                )
                p.data.addcdiv_(exp_avg, denom, value=-step_size)


# ─── Mixed precision presets ─────────────────────────────────────────────────

fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)


# ─── FSDP wrap policy helper ─────────────────────────────────────────────────

def get_llama_wrapper():
    layer_classes = {LlamaDecoderLayer}
    for cls in [
        MllamaSelfAttentionDecoderLayer,
        MllamaCrossAttentionDecoderLayer,
        MllamaVisionEncoderLayer,
    ]:
        if cls is not None:
            layer_classes.add(cls)
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )


if __name__ == "__main__":
    if _FIRE_AVAILABLE:
        fire.Fire(main)
    else:
        import sys
        fsdp = sys.argv[1] if len(sys.argv) > 1 else ""
        consolidated = sys.argv[2] if len(sys.argv) > 2 else ""
        hf_model = sys.argv[3] if len(sys.argv) > 3 else ""
        main(fsdp_checkpoint_path=fsdp, consolidated_model_path=consolidated, HF_model_path_or_name=hf_model)