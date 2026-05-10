from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from configs import (
    FUIDAI_MODEL_NAME,
    FUIDAI_VERSION,
    FUIDAI_PARAMS,
    FUIDAI_MODEL_ID,
)

PROJECT_ROOT = Path(__file__).resolve().parent

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "fuidai_sft"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / FUIDAI_MODEL_ID
FINETUNING_SCRIPT = PROJECT_ROOT / "finetuning.py"
INFERENCE_SCRIPT = PROJECT_ROOT / "inference.py"
LEGACY_SCRIPT = PROJECT_ROOT / "huan_luyen_legacy.py"

T4_NUM_GPUS = 2
T4_CONTEXT_LENGTH = 2048
T4_BATCH_SIZE = 4
T4_GRAD_ACCUM = 4
T4_LR = 2e-4
T4_LORA_R = 16
T4_LORA_ALPHA = 32

NEW_COMMANDS = {"doctor", "prepare", "finetune", "continue", "chat", "export", "auto", "legacy"}
LEGACY_FLAGS = {"--ckpt", "--epochs", "--data", "--lr"}

DATA_EXTS = {".json", ".jsonl", ".txt", ".md"}
MODEL_MARKER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}

KAGGLE_MODEL_SEARCH_DIRS = [
    KAGGLE_INPUT,
    KAGGLE_WORKING / "models",
]

LOCAL_DATA_SEARCH_DIRS = [
    PROJECT_ROOT / "data_hf_da_chia",
    PROJECT_ROOT / "data" / "DATATOTRAIN",
    PROJECT_ROOT / "data_train",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "DATATOTRAIN",
]

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{input_section}.\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


class AppError(RuntimeError):
    pass


def info(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def fail(msg: str) -> None:
    raise AppError(msg)


def ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_cmd_string(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(cmd)


def is_kaggle() -> bool:
    return KAGGLE_WORKING.exists() and KAGGLE_INPUT.exists()


def output_root() -> Path:
    if is_kaggle():
        return KAGGLE_WORKING
    return PROJECT_ROOT


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.8"
    return env


def run_command(cmd: list[str], *, dry_run: bool = False, cwd: Path | None = None) -> int:
    info(f"$ {to_cmd_string(cmd)}")
    if dry_run:
        info("[dry-run] skipping execution")
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT), env=build_env())
    return proc.returncode


def should_delegate_to_legacy(argv: list[str]) -> bool:
    if not argv:
        return True
    if any(a in ("-h", "--help") for a in argv):
        return False
    if argv[0] in NEW_COMMANDS:
        return False
    return any(flag in argv for flag in LEGACY_FLAGS)


def detect_module(modname: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(modname)
    if spec is None:
        return False, "missing"
    try:
        version = importlib.metadata.version(modname)
    except importlib.metadata.PackageNotFoundError:
        version = "installed"
    return True, version


def detect_gpu_info() -> list[dict[str, Any]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024 ** 3, 2),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
            })
        return gpus
    except Exception:
        return []


def is_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return any((p / m).exists() for m in MODEL_MARKER_FILES)


def find_model_auto() -> Path | None:
    search_roots: list[Path] = []
    if is_kaggle():
        search_roots += KAGGLE_MODEL_SEARCH_DIRS
    search_roots += [
        PROJECT_ROOT / "models",
        PROJECT_ROOT.parent / "models",
        Path.home() / "models",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        if is_model_dir(root):
            return root
        for child in sorted(root.iterdir()):
            if is_model_dir(child):
                return child
            for grandchild in sorted(child.iterdir()) if child.is_dir() else []:
                if is_model_dir(grandchild):
                    return grandchild

    return None


def find_data_files_auto(extra_inputs: list[str] | None = None) -> list[Path]:
    search_roots: list[Path] = []

    if extra_inputs:
        for s in extra_inputs:
            p = Path(s)
            if p.exists():
                search_roots.append(p)

    if is_kaggle():
        if KAGGLE_INPUT.exists():
            search_roots.append(KAGGLE_INPUT)

    search_roots += LOCAL_DATA_SEARCH_DIRS

    found: list[Path] = []
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        target = root if root.is_file() else None
        files_iter = [root] if root.is_file() else root.rglob("*")
        for p in files_iter:
            if p.is_file() and p.suffix.lower() in DATA_EXTS and str(p) not in seen:
                seen.add(str(p))
                found.append(p)

    return sorted(found)


def discover_candidate_checkpoints(root: Path) -> list[Path]:
    candidates: list[Path] = []
    search_roots = [root / "output", root / "runs", root / "checkpoints", root]
    if is_kaggle():
        search_roots += [KAGGLE_WORKING / "runs", KAGGLE_WORKING / "output"]
    seen: set[str] = set()

    for base in search_roots:
        if not base.exists():
            continue
        for config_file in base.rglob("train_params.yaml"):
            parent = config_file.parent
            has_weight = any(parent.glob("*.bin")) or any(parent.glob("*.safetensors"))
            if has_weight and str(parent) not in seen:
                seen.add(str(parent))
                candidates.append(parent)
        for config_file in base.rglob("adapter_config.json"):
            parent = config_file.parent
            if str(parent) not in seen:
                seen.add(str(parent))
                candidates.append(parent)

    return sorted(candidates)


def read_val_loss(ckpt_dir: Path) -> float:
    for name in ("train_metrics.json", "metrics.json", "eval_results.json"):
        p = ckpt_dir / name
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for key in ("eval_loss", "val_loss", "validation_loss"):
                    if key in data:
                        return float(data[key])
            except Exception:
                pass

    for csv_p in ckpt_dir.glob("*.csv"):
        try:
            lines = csv_p.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                continue
            header = [h.strip().lower() for h in lines[0].split(",")]
            for key in ("eval_loss", "val_loss"):
                if key in header:
                    idx = header.index(key)
                    last_val = lines[-1].split(",")[idx].strip()
                    return float(last_val)
        except Exception:
            pass

    try:
        mtime = ckpt_dir.stat().st_mtime
        return -mtime
    except Exception:
        return float("inf")


def find_best_checkpoint(search_root: Path) -> Path | None:
    candidates = discover_candidate_checkpoints(search_root)
    if not candidates:
        return None
    best = min(candidates, key=lambda p: read_val_loss(p))
    return best


def zip_checkpoint(ckpt_path: Path, zip_out: Path, prefix: str = "") -> Path:
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        base = ckpt_path.parent
        for f in sorted(ckpt_path.rglob("*")):
            if not f.is_file():
                continue
            arc = Path(prefix) / f.relative_to(base) if prefix else f.relative_to(base)
            zf.write(f, arc)
    return zip_out


def install_dependencies_if_needed() -> None:
    mods = ["transformers", "peft", "accelerate", "datasets", "bitsandbytes", "llama_cookbook"]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if not missing:
        return
    info(f"Cai dat dependencies con thieu: {missing}")
    pip_names = {
        "llama_cookbook": "llama-cookbook",
        "bitsandbytes": "bitsandbytes",
    }
    pkgs = [pip_names.get(m, m) for m in missing]
    run_command([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


def messages_to_text(messages: list, tokenizer=None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            pass
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _detect_schema(sample: dict) -> str:
    if "messages" in sample and isinstance(sample["messages"], list):
        return "messages"
    if "text" in sample and isinstance(sample["text"], str):
        return "text"
    if "instruction" in sample or "output" in sample:
        return "alpaca"
    return "unknown"


def _normalize_sample(sample: dict, tokenizer=None) -> dict | None:
    schema = _detect_schema(sample)
    if schema == "text":
        return {"text": sample["text"]}
    if schema == "messages":
        return {"text": messages_to_text(sample["messages"], tokenizer)}
    if schema == "alpaca":
        instruction = sample.get("instruction", "")
        inp = sample.get("input", "").strip()
        output = sample.get("output", "")
        input_section = f", taking the following as input\n### Input:\n{inp}" if inp else ""
        text = ALPACA_TEMPLATE.format(
            input_section=input_section,
            instruction=instruction,
            output=output,
        )
        return {"text": text}
    return None


def _load_local(path: str) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            return [data]
    raise ValueError(f"Khong ho tro dinh dang: {suffix!r}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def iter_input_files(inputs: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            warn(f"Path khong ton tai, bo qua: {p}")
            continue
        if p.is_file() and p.suffix.lower() in DATA_EXTS:
            out.append(p)
            continue
        if p.is_dir():
            for fp in p.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in DATA_EXTS:
                    out.append(fp)
    return sorted(set(out))


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                warn(f"JSONL decode loi tai {path}:{idx}")
                continue
            if isinstance(obj, list):
                yield from obj
            else:
                yield obj


def iter_json(path: Path):
    text = safe_read_text(path).strip()
    if not text:
        return
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        warn(f"JSON decode loi: {path}")
        return
    if isinstance(data, list):
        yield from data
    else:
        yield data


def iter_records(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
    elif suffix == ".json":
        yield from iter_json(path)
    elif suffix in {".txt", ".md"}:
        text = safe_read_text(path)
        if text.strip():
            yield {"text": text}


_DIALOG_RE = re.compile(
    r"(?:^|\n)\s*(?:user|human|q|prompt)\s*:\s*(.*?)\s*\n\s*(?:assistant|fuid|fuidai|bot|a|response)\s*:\s*(.*?)(?=(?:\n\s*(?:user|human|q|prompt)\s*:)|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def flatten_to_text(obj: Any) -> str:
    parts: list[str] = []

    def walk(x: Any, prefix: str = "") -> None:
        if isinstance(x, str):
            s = normalize_text(x)
            if s:
                parts.append(f"{prefix}{s}" if prefix else s)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{k}: ")
            return
        if isinstance(x, list):
            for v in x:
                walk(v, prefix)

    walk(obj)
    return normalize_text("\n".join(parts))


def split_for_continuation(text: str, max_chars: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        pivot = remaining.rfind("\n", int(max_chars * 0.45), max_chars)
        if pivot == -1:
            pivot = remaining.rfind(". ", int(max_chars * 0.45), max_chars)
        if pivot == -1:
            pivot = max_chars
        chunk = normalize_text(remaining[:pivot])
        if chunk:
            chunks.append(chunk)
        remaining = normalize_text(remaining[pivot:])
    return chunks


def continuation_sample_from_text(text: str, min_chars: int, max_chars: int) -> list[dict[str, str]]:
    text = normalize_text(text)
    if len(text) < min_chars:
        return []

    dialog_matches = list(_DIALOG_RE.finditer(text))
    if dialog_matches:
        out: list[dict[str, str]] = []
        for m in dialog_matches:
            prompt = normalize_text(m.group(1))
            answer = normalize_text(m.group(2))
            if len(answer) < min_chars:
                continue
            out.append({"instruction": prompt or "Tra loi cho nguoi dung.", "output": answer})
        if out:
            return out

    samples: list[dict[str, str]] = []
    for chunk in split_for_continuation(text, max_chars=max_chars):
        if len(chunk) < (min_chars * 2):
            continue
        split_at = int(len(chunk) * 0.60)
        for sep in ("\n", ". ", "? ", "! "):
            cand = chunk.find(sep, split_at)
            if cand != -1:
                split_at = cand + len(sep)
                break
        prefix = normalize_text(chunk[:split_at])
        suffix = normalize_text(chunk[split_at:])
        if len(prefix) < min_chars or len(suffix) < min_chars:
            continue
        samples.append({
            "instruction": "Viet tiep noi dung theo dung ngu canh va van phong da cho.",
            "input": prefix,
            "output": suffix,
        })
    if samples:
        return samples

    return [{"instruction": "Viet lai doan van sau ro rang va mach lac.", "output": text[:max_chars]}]


def records_to_sft_samples(record: Any, *, min_chars: int, max_chars: int) -> list[dict[str, str]]:
    if record is None:
        return []
    if isinstance(record, str):
        return continuation_sample_from_text(record, min_chars=min_chars, max_chars=max_chars)
    if isinstance(record, list):
        out: list[dict[str, str]] = []
        for item in record:
            out.extend(records_to_sft_samples(item, min_chars=min_chars, max_chars=max_chars))
        return out
    if not isinstance(record, dict):
        return []

    k = {str(x).lower() for x in record.keys()}

    def clean(v: Any) -> str:
        if not isinstance(v, str):
            return ""
        return normalize_text(v)

    if "instruction" in k and "output" in k:
        instruction = clean(record.get("instruction", ""))
        output = clean(record.get("output", ""))
        input_text = clean(record.get("input", ""))
        if len(output) >= min_chars:
            sample: dict[str, str] = {"instruction": instruction or "Tra loi cau hoi cua nguoi dung.", "output": output}
            if input_text:
                sample["input"] = input_text
            return [sample]

    if "question" in k and "answer" in k:
        question = clean(record.get("question", ""))
        answer = clean(record.get("answer", ""))
        if len(answer) >= min_chars:
            return [{"instruction": question or "Tra loi cau hoi.", "output": answer}]

    user = clean(record.get("user", "") or record.get("prompt", ""))
    assistant = clean(record.get("assistant", "") or record.get("response", ""))
    if assistant and len(assistant) >= min_chars:
        return [{"instruction": user or "Tra loi nguoi dung.", "output": assistant}]

    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        out: list[dict[str, str]] = []
        prev_user = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", msg.get("from", ""))).lower().strip()
            content = clean(msg.get("content", msg.get("value", "")))
            if not content:
                continue
            if role in {"user", "human", "prompt", "question"}:
                prev_user = content
            elif role in {"assistant", "bot", "gpt", "fuid", "fuidai", "answer"}:
                if len(content) >= min_chars:
                    out.append({"instruction": prev_user or "Tra loi nguoi dung.", "output": content})
        if out:
            return out

    text_value = clean(record.get("text", ""))
    if text_value:
        return continuation_sample_from_text(text_value, min_chars=min_chars, max_chars=max_chars)

    fallback_text = flatten_to_text(record)
    if fallback_text:
        return continuation_sample_from_text(fallback_text, min_chars=min_chars, max_chars=max_chars)

    return []


def sanitize_sample(sample: dict[str, str], *, min_chars: int, max_chars: int) -> dict[str, str] | None:
    instruction = normalize_text(sample.get("instruction", ""))
    output = normalize_text(sample.get("output", ""))
    input_text = normalize_text(sample.get("input", ""))
    if len(output) < min_chars:
        return None
    if len(output) > max_chars:
        output = output[:max_chars]
    cleaned: dict[str, str] = {"instruction": instruction or "Tra loi nguoi dung.", "output": output}
    if input_text:
        cleaned["input"] = input_text
    return cleaned


def run_prepare_pipeline(
    source_inputs: list[str | Path],
    out_dir: Path,
    *,
    val_ratio: float = 0.02,
    seed: int = 42,
    min_output_chars: int = 24,
    max_chars: int = 6000,
    max_samples: int = 0,
    log_every: int = 2000,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    files = iter_input_files([str(s) for s in source_inputs])
    if not files:
        fail(f"Khong tim thay file data nao trong: {source_inputs}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    stats_path = out_dir / "prepare_stats.json"

    if overwrite:
        for p in (train_path, val_path, stats_path):
            if p.exists():
                p.unlink()

    rnd = random.Random(seed)
    total_records = total_samples_raw = total_samples_kept = total_train = total_val = dropped = 0

    info(f"Input files: {len(files)}")

    with train_path.open("a", encoding="utf-8") as f_train, val_path.open("a", encoding="utf-8") as f_val:
        stop = False
        for file_idx, fp in enumerate(files, 1):
            if stop:
                break
            info(f"[{file_idx}/{len(files)}] {fp.name}")
            try:
                for rec in iter_records(fp):
                    total_records += 1
                    samples = records_to_sft_samples(rec, min_chars=min_output_chars, max_chars=max_chars)
                    total_samples_raw += len(samples)
                    for s in samples:
                        cleaned = sanitize_sample(s, min_chars=min_output_chars, max_chars=max_chars)
                        if cleaned is None:
                            dropped += 1
                            continue
                        line = json.dumps(cleaned, ensure_ascii=False)
                        if rnd.random() < val_ratio:
                            f_val.write(line + "\n")
                            total_val += 1
                        else:
                            f_train.write(line + "\n")
                            total_train += 1
                        total_samples_kept += 1
                        if max_samples > 0 and total_samples_kept >= max_samples:
                            stop = True
                            break
                    if total_records % log_every == 0:
                        info(f"  records={total_records:,} kept={total_samples_kept:,} train={total_train:,} val={total_val:,}")
                    if stop:
                        break
            except Exception as ex:
                warn(f"Bo qua file loi: {fp} ({ex})")

    if total_samples_kept == 0:
        fail("Khong co sample nao hop le. Kiem tra format input hoac giam min-output-chars.")

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": FUIDAI_MODEL_ID,
        "input_files": [str(x) for x in files],
        "records_seen": total_records,
        "samples_raw": total_samples_raw,
        "samples_kept": total_samples_kept,
        "samples_train": total_train,
        "samples_val": total_val,
        "dropped": dropped,
        "val_ratio": val_ratio,
        "seed": seed,
        "min_output_chars": min_output_chars,
        "max_chars": max_chars,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    info(f"Prepare xong: train={total_train:,} val={total_val:,} dropped={dropped:,}")
    return train_path, val_path


def build_torchrun_finetune_cmd(
    args: argparse.Namespace,
    *,
    model_name: str,
    data_path: Path,
    out_dir: Path,
) -> list[str]:
    num_gpus = getattr(args, "num_gpus", T4_NUM_GPUS)
    master_port = getattr(args, "master_port", 29500)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={master_port}",
        str(FINETUNING_SCRIPT),
        f"--model_name={model_name}",
        f"--output_dir={out_dir}",
        f"--dist_checkpoint_root_folder={out_dir / 'fsdp'}",
        f"--dataset=fuidai_dataset",
        f"--data_path={data_path}",
        f"--context_length={getattr(args, 'context_length', T4_CONTEXT_LENGTH)}",
        f"--batch_size_training={getattr(args, 'batch_size', T4_BATCH_SIZE)}",
        f"--val_batch_size=1",
        f"--gradient_accumulation_steps={getattr(args, 'grad_accum', T4_GRAD_ACCUM)}",
        f"--num_epochs={getattr(args, 'epochs', 3)}",
        f"--lr={getattr(args, 'learning_rate', T4_LR)}",
        f"--weight_decay={getattr(args, 'weight_decay', 0.01)}",
        f"--gamma={getattr(args, 'gamma', 0.85)}",
        f"--seed={getattr(args, 'seed', 42)}",
        f"--num_workers_dataloader={getattr(args, 'num_workers', 2)}",
    ]

    if getattr(args, "enable_fsdp", True):
        cmd += ["--enable_fsdp=True", "--low_cpu_fsdp=False"]
    if getattr(args, "use_fp16", True):
        cmd += ["--use_fp16=True"]
    if getattr(args, "use_peft", True):
        cmd += [
            "--use_peft=True",
            f"--peft_method={getattr(args, 'peft_method', 'lora')}",
            f"--lora_r={getattr(args, 'lora_r', T4_LORA_R)}",
            f"--lora_alpha={getattr(args, 'lora_alpha', T4_LORA_ALPHA)}",
            f"--lora_dropout={getattr(args, 'lora_dropout', 0.05)}",
        ]

    quantization = getattr(args, "quantization", "4bit")
    if quantization and quantization != "none":
        cmd += [f"--quantization={quantization}"]
    if getattr(args, "gradient_clipping", True):
        cmd += ["--gradient_clipping=True", "--gradient_clipping_threshold=1.0"]
    if getattr(args, "use_fast_kernels", True):
        cmd += ["--use_fast_kernels=True"]
    if getattr(args, "run_validation", True):
        cmd += ["--run_validation=True"]
    if getattr(args, "save_metrics", True):
        cmd += ["--save_metrics=True"]
    if getattr(args, "use_wandb", False):
        cmd += [
            "--use_wandb=True",
            f"--wandb_project={getattr(args, 'wandb_project', f'{FUIDAI_MODEL_NAME}-{FUIDAI_VERSION}')}",
        ]

    max_train_step = getattr(args, "max_steps", 0)
    if max_train_step > 0:
        cmd += [f"--max_train_step={max_train_step}"]

    from_peft_checkpoint = getattr(args, "from_peft_checkpoint", "")
    if from_peft_checkpoint:
        cmd += [f"--from_peft_checkpoint={from_peft_checkpoint}"]

    return cmd


def save_train_params(out_dir: Path, params: dict[str, Any]) -> None:
    try:
        import yaml
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "train_params.yaml"
        p.write_text(yaml.dump(params, allow_unicode=True, default_flow_style=False), encoding="utf-8")
        info(f"Train params: {p}")
    except ImportError:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "train_params.json"
        p.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        info(f"Train params: {p}")


def do_zip_best(out_dir: Path, export_dir: Path) -> Path | None:
    info("\n=== Xuat checkpoint tot nhat ===")
    best = find_best_checkpoint(out_dir)
    if best is None:
        best_candidates = list(out_dir.rglob("adapter_config.json"))
        if best_candidates:
            best = sorted(best_candidates, key=lambda p: p.stat().st_mtime)[-1].parent
    if best is None:
        warn("Khong tim thay checkpoint nao de zip.")
        return None

    info(f"Checkpoint tot nhat: {best}")
    zip_name = f"{FUIDAI_MODEL_ID}_best_{now_tag()}.zip"
    zip_path = export_dir / zip_name
    zip_checkpoint(best, zip_path, prefix=FUIDAI_MODEL_ID)
    size_mb = zip_path.stat().st_size / 1024 / 1024
    info(f"Da zip: {zip_path}  ({size_mb:.1f} MB)")
    return zip_path


def cmd_doctor(args: argparse.Namespace) -> int:
    info(f"\n=== FuidAI Doctor [{FUIDAI_MODEL_ID}] ===")
    info(f"Project root : {PROJECT_ROOT}")
    info(f"Python       : {sys.executable}")
    info(f"Version      : {sys.version.split()[0]}")
    info(f"Kaggle env   : {is_kaggle()}")

    info("\n[GPU Info]")
    gpus = detect_gpu_info()
    if not gpus:
        warn("Khong tim thay GPU CUDA.")
    else:
        for g in gpus:
            info(f"  GPU {g['index']}: {g['name']} | {g['total_memory_gb']} GB | CC {g['major']}.{g['minor']}")
        if len(gpus) >= T4_NUM_GPUS:
            info(f"  T4 x{T4_NUM_GPUS} config: READY")
        else:
            warn(f"  Chi co {len(gpus)} GPU, khuyen nghi {T4_NUM_GPUS}")

    core_mods = ["torch", "transformers", "peft", "accelerate", "datasets"]
    optional_mods = ["bitsandbytes", "wandb", "fire", "sentencepiece", "yaml", "llama_cookbook"]
    missing_core = []

    info("\n[Core modules]")
    for name in core_mods:
        ok, ver = detect_module(name)
        info(f"  {'OK' if ok else 'MISSING'}  {name:<25} {ver}")
        if not ok:
            missing_core.append(name)

    info("\n[Optional modules]")
    for name in optional_mods:
        ok, ver = detect_module(name)
        info(f"  {'OK' if ok else 'MISSING'}  {name:<25} {ver}")

    info("\n[FuidAI scripts]")
    for label, p in {
        "finetuning.py": FINETUNING_SCRIPT,
        "inference.py": INFERENCE_SCRIPT,
        "configs.py": PROJECT_ROOT / "configs.py",
        "data.py": PROJECT_ROOT / "data.py",
        "huan_luyen_legacy.py": LEGACY_SCRIPT,
    }.items():
        info(f"  {'OK' if p.exists() else 'MISSING'}  {label}")

    info("\n[Auto-detect model]")
    model = find_model_auto()
    if model:
        info(f"  Tìm thấy model: {model}")
    else:
        warn("  Chua tim thay model weights.")

    info("\n[Auto-detect data]")
    data_files = find_data_files_auto()
    if data_files:
        for fp in data_files[:5]:
            info(f"  {fp}")
        if len(data_files) > 5:
            info(f"  ... va {len(data_files) - 5} files khac")
    else:
        warn("  Chua tim thay file data nao.")

    info("\n[Checkpoint candidates]")
    ckpts = discover_candidate_checkpoints(output_root())
    if not ckpts:
        warn("Chua tim thay checkpoint nao.")
    else:
        for c in ckpts[:5]:
            loss = read_val_loss(c)
            info(f"  {c}  (val_loss={loss:.4f})" if loss != float("inf") and loss >= 0 else f"  {c}")

    if missing_core:
        info(f"\nDoctor: INCOMPLETE — thieu: {', '.join(missing_core)}")
        return 1

    info(f"\nDoctor: READY")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    source_inputs = args.input or [str(d) for d in LOCAL_DATA_SEARCH_DIRS]
    if is_kaggle() and not args.input:
        source_inputs = [str(KAGGLE_INPUT)] + source_inputs

    out_dir = Path(args.out_dir or DEFAULT_DATA_DIR)
    info(f"\n=== FuidAI Prepare [{FUIDAI_MODEL_ID}] ===")
    info(f"Output dir: {out_dir}")

    run_prepare_pipeline(
        source_inputs,
        out_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_output_chars=args.min_output_chars,
        max_chars=args.max_chars,
        max_samples=args.max_samples,
        log_every=args.log_every,
        overwrite=args.overwrite,
    )
    return 0


def cmd_finetune(args: argparse.Namespace) -> int:
    if not FINETUNING_SCRIPT.exists():
        fail(f"finetuning.py khong tim thay: {FINETUNING_SCRIPT}")

    data_path = Path(args.data) if getattr(args, "data", None) else DEFAULT_DATA_DIR / "train.jsonl"
    if not data_path.exists():
        fail(f"Data khong ton tai: {data_path}. Chay 'prepare' truoc.")

    model_name = args.model_name
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else (output_root() / "runs" / f"run_{now_tag()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_train_params(out_dir, {
        "model_name": model_name,
        "model_id": FUIDAI_MODEL_ID,
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "num_gpus": getattr(args, "num_gpus", T4_NUM_GPUS),
        "context_length": getattr(args, "context_length", T4_CONTEXT_LENGTH),
        "batch_size_training": getattr(args, "batch_size", T4_BATCH_SIZE),
        "gradient_accumulation_steps": getattr(args, "grad_accum", T4_GRAD_ACCUM),
        "num_epochs": getattr(args, "epochs", 3),
        "lr": getattr(args, "learning_rate", T4_LR),
        "peft_method": getattr(args, "peft_method", "lora"),
        "quantization": getattr(args, "quantization", "4bit"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })

    info(f"\n=== FuidAI Finetune [{FUIDAI_MODEL_ID}] ===")
    info(f"Model     : {model_name}")
    info(f"Data      : {data_path}")
    info(f"Output    : {out_dir}")
    info(f"GPUs      : {getattr(args, 'num_gpus', T4_NUM_GPUS)}")
    info(f"Context   : {getattr(args, 'context_length', T4_CONTEXT_LENGTH)}")
    info(f"Batch     : {getattr(args, 'batch_size', T4_BATCH_SIZE)}")
    info(f"Grad accum: {getattr(args, 'grad_accum', T4_GRAD_ACCUM)}")
    eff = getattr(args, "batch_size", T4_BATCH_SIZE) * getattr(args, "num_gpus", T4_NUM_GPUS) * getattr(args, "grad_accum", T4_GRAD_ACCUM)
    info(f"Eff batch : {eff}")
    info(f"LR        : {getattr(args, 'learning_rate', T4_LR)}")
    info(f"LoRA r    : {getattr(args, 'lora_r', T4_LORA_R)}")
    info(f"Quant     : {getattr(args, 'quantization', '4bit')}")

    cmd = build_torchrun_finetune_cmd(args, model_name=model_name, data_path=data_path, out_dir=out_dir)
    code = run_command(cmd, dry_run=getattr(args, "dry_run", False))
    if code != 0:
        fail(f"Finetune that bai: exit code {code}")

    if getattr(args, "zip_best", False):
        do_zip_best(out_dir, output_root())

    info(f"\nFinetune hoan tat. Checkpoint: {out_dir}")
    return 0


def find_resume_checkpoint(from_run: Path) -> Path:
    for c in [from_run / "final", from_run]:
        if (c / "adapter_config.json").is_file() or (c / "train_params.yaml").is_file():
            return c
    for c in from_run.rglob("adapter_config.json"):
        return c.parent
    fail(f"Khong tim thay checkpoint hop le trong: {from_run}")


def cmd_continue(args: argparse.Namespace) -> int:
    from_run = Path(args.from_run)
    if not from_run.exists():
        fail(f"Run directory khong ton tai: {from_run}")

    resume_ckpt = find_resume_checkpoint(from_run)
    info(f"Resume checkpoint: {resume_ckpt}")

    forwarded = argparse.Namespace(**vars(args))
    forwarded.from_peft_checkpoint = str(resume_ckpt)

    if not getattr(forwarded, "out_dir", None):
        forwarded.out_dir = str(from_run.parent / f"{from_run.name}_cont_{now_tag()}")

    if not getattr(forwarded, "model_name", None):
        try:
            import yaml
            params_file = from_run / "train_params.yaml"
            if params_file.exists():
                with params_file.open(encoding="utf-8") as f:
                    params = yaml.safe_load(f)
                forwarded.model_name = params.get("model_name", "")
                info(f"Model name tu checkpoint: {forwarded.model_name}")
        except Exception:
            pass

    if not getattr(forwarded, "model_name", None):
        fail("Khong lay duoc model_name. Truyen --model-name.")

    return cmd_finetune(forwarded)


def cmd_export(args: argparse.Namespace) -> int:
    if not INFERENCE_SCRIPT.exists():
        fail(f"inference.py khong tim thay: {INFERENCE_SCRIPT}")

    fsdp_path = args.fsdp_checkpoint
    output_path = args.output or str(DEFAULT_OUTPUT_DIR / "exported" / now_tag())

    info(f"\n=== FuidAI Export [{FUIDAI_MODEL_ID}] ===")
    info(f"FSDP checkpoint: {fsdp_path}")
    info(f"Output         : {output_path}")

    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        f"--fsdp_checkpoint_path={fsdp_path}",
        f"--consolidated_model_path={output_path}",
    ]
    if getattr(args, "hf_model", None):
        cmd += [f"--HF_model_path_or_name={args.hf_model}"]

    code = run_command(cmd, dry_run=getattr(args, "dry_run", False))
    if code != 0:
        fail(f"Export that bai: exit code {code}")

    if getattr(args, "zip_output", False):
        zip_name = f"{FUIDAI_MODEL_ID}_export_{now_tag()}.zip"
        zip_path = output_root() / zip_name
        zip_checkpoint(Path(output_path), zip_path)
        info(f"Da zip: {zip_path}")

    info(f"\nExport hoan tat: {output_path}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint
    info(f"\n=== FuidAI Chat [{FUIDAI_MODEL_ID}] ===")
    info(f"Checkpoint: {checkpoint}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        info("Dang tai model...")
        tok = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()

        gen_config = {
            "max_new_tokens": getattr(args, "max_new_tokens", 256),
            "temperature": getattr(args, "temperature", 0.8),
            "top_p": getattr(args, "top_p", 0.9),
            "top_k": getattr(args, "top_k", 50),
            "repetition_penalty": getattr(args, "repetition_penalty", 1.1),
            "do_sample": True,
            "pad_token_id": tok.eos_token_id,
        }

        info(f"FuidAI {FUIDAI_MODEL_ID} san sang. Nhap 'exit' de thoat.\n")
        while True:
            try:
                user_input = input("Ban: ").strip()
            except (EOFError, KeyboardInterrupt):
                info("\nThoat.")
                break
            if not user_input or user_input.lower() in {"exit", "quit", "thoat"}:
                break
            prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_config)
            response_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tok.decode(response_ids, skip_special_tokens=True).strip()
            info(f"FuidAI: {response}\n")

    except ImportError as e:
        fail(f"Thieu thu vien: {e}")
    except Exception as e:
        fail(f"Chat loi: {e}")

    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    info(f"\n{'='*60}")
    info(f"  FuidAI Auto Pipeline [{FUIDAI_MODEL_ID}]")
    info(f"  Kaggle: {is_kaggle()}")
    info(f"{'='*60}")

    install_dependencies_if_needed()

    model_name = getattr(args, "model_name", None)
    if not model_name:
        model_path = find_model_auto()
        if model_path:
            model_name = str(model_path)
            info(f"[AUTO] Model: {model_name}")
        else:
            fail(
                "Khong tim thay model weights.\n"
                "Tren Kaggle: them dataset chua model vao /kaggle/input/\n"
                "Hoac truyen: --model-name <ten-hoac-duong-dan>"
            )

    data_dir = output_root() / "data" / "fuidai_sft"
    train_jsonl = data_dir / "train.jsonl"

    if not train_jsonl.exists() or getattr(args, "reprepare", False):
        info(f"\n[AUTO] Chuan bi data...")
        data_files = find_data_files_auto(
            extra_inputs=([args.data] if getattr(args, "data", None) else None)
        )
        if not data_files:
            fail(
                "Khong tim thay file data nao.\n"
                "Tren Kaggle: them dataset chua data vao /kaggle/input/\n"
                "Hoac truyen: --data <duong-dan>"
            )
        info(f"Tim thay {len(data_files)} file data.")
        run_prepare_pipeline(
            [str(f) for f in data_files],
            data_dir,
            val_ratio=getattr(args, "val_ratio", 0.02),
            seed=getattr(args, "seed", 42),
            min_output_chars=getattr(args, "min_output_chars", 24),
            max_chars=getattr(args, "max_chars", 6000),
            max_samples=getattr(args, "max_samples", 0),
            overwrite=True,
        )
    else:
        info(f"[AUTO] Da co data: {train_jsonl}")

    run_tag = now_tag()
    out_dir = output_root() / "runs" / f"auto_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args.model_name = model_name
    args.data = str(train_jsonl)
    args.out_dir = str(out_dir)
    args.zip_best = False

    if not FINETUNING_SCRIPT.exists():
        fail(f"finetuning.py khong tim thay: {FINETUNING_SCRIPT}")

    info(f"\n[AUTO] Bat dau train...")
    save_train_params(out_dir, {
        "model_name": model_name,
        "model_id": FUIDAI_MODEL_ID,
        "data_path": str(train_jsonl),
        "out_dir": str(out_dir),
        "auto_pipeline": True,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    })

    cmd = build_torchrun_finetune_cmd(args, model_name=model_name, data_path=train_jsonl, out_dir=out_dir)
    code = run_command(cmd, dry_run=getattr(args, "dry_run", False))

    if code != 0:
        warn(f"Training ket thuc voi exit code {code}. Van co gang zip checkpoint...")

    info(f"\n[AUTO] Xuat checkpoint tot nhat...")
    export_dir = output_root()
    zip_path = do_zip_best(out_dir, export_dir)

    if zip_path and zip_path.exists():
        size_mb = zip_path.stat().st_size / 1024 / 1024
        info(f"\n{'='*60}")
        info(f"  HOAN TAT!")
        info(f"  Checkpoint zip: {zip_path}")
        info(f"  Kich thuoc    : {size_mb:.1f} MB")
        if is_kaggle():
            info(f"  Tai xuong tai : /kaggle/working/{zip_path.name}")
        info(f"{'='*60}")
    else:
        info(f"\nTraining hoan tat. Checkpoint tai: {out_dir}")

    return 0 if code == 0 else code


def cmd_legacy(args: argparse.Namespace) -> int:
    if not LEGACY_SCRIPT.exists():
        fail(f"Legacy script khong tim thay: {LEGACY_SCRIPT}")
    cmd = [sys.executable, str(LEGACY_SCRIPT)] + list(args.legacy_args or [])
    info("[legacy] Chuyen sang huan_luyen_legacy.py")
    return run_command(cmd)


def add_shared_finetune_args(p: argparse.ArgumentParser, *, include_model: bool = True) -> None:
    if include_model:
        p.add_argument("--model-name", required=True, help="Path den model hoac HF repo")
    p.add_argument("--data", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--num-gpus", type=int, default=T4_NUM_GPUS)
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=T4_BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=T4_GRAD_ACCUM)
    p.add_argument("--context-length", type=int, default=T4_CONTEXT_LENGTH)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=T4_LR)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-peft", action="store_true", default=True)
    p.add_argument("--no-peft", dest="use_peft", action="store_false")
    p.add_argument("--peft-method", default="lora", choices=["lora", "prefix", "llama_adapter"])
    p.add_argument("--lora-r", type=int, default=T4_LORA_R)
    p.add_argument("--lora-alpha", type=int, default=T4_LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--from-peft-checkpoint", default="")
    p.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--use-fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    p.add_argument("--enable-fsdp", action="store_true", default=True)
    p.add_argument("--no-fsdp", dest="enable_fsdp", action="store_false")
    p.add_argument("--use-fast-kernels", action="store_true", default=True)
    p.add_argument("--gradient-clipping", action="store_true", default=True)
    p.add_argument("--run-validation", action="store_true", default=True)
    p.add_argument("--save-metrics", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default=f"{FUIDAI_MODEL_NAME}-{FUIDAI_VERSION}")
    p.add_argument("--zip-best", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=f"FuidAI {FUIDAI_MODEL_ID} Training Orchestrator (T4 x{T4_NUM_GPUS})",
    )
    sub = parser.add_subparsers(dest="command")

    p_doctor = sub.add_parser("doctor", help="Kiem tra environment, GPU, dependencies, auto-detect")
    p_doctor.set_defaults(func=cmd_doctor)

    p_auto = sub.add_parser("auto", help="Pipeline tu dong 100%: tim data, prepare, train, zip best (danh cho Kaggle)")
    p_auto.add_argument("--model-name", default=None, help="Path model (tu dong tim neu de trong)")
    p_auto.add_argument("--data", default=None, help="Path data (tu dong tim neu de trong)")
    p_auto.add_argument("--reprepare", action="store_true", default=False, help="Chay lai buoc prepare du da co train.jsonl")
    p_auto.add_argument("--num-gpus", type=int, default=T4_NUM_GPUS)
    p_auto.add_argument("--master-port", type=int, default=29500)
    p_auto.add_argument("--epochs", type=int, default=3)
    p_auto.add_argument("--batch-size", type=int, default=T4_BATCH_SIZE)
    p_auto.add_argument("--grad-accum", type=int, default=T4_GRAD_ACCUM)
    p_auto.add_argument("--context-length", type=int, default=T4_CONTEXT_LENGTH)
    p_auto.add_argument("--max-steps", type=int, default=0)
    p_auto.add_argument("--learning-rate", type=float, default=T4_LR)
    p_auto.add_argument("--weight-decay", type=float, default=0.01)
    p_auto.add_argument("--gamma", type=float, default=0.85)
    p_auto.add_argument("--seed", type=int, default=42)
    p_auto.add_argument("--lora-r", type=int, default=T4_LORA_R)
    p_auto.add_argument("--lora-alpha", type=int, default=T4_LORA_ALPHA)
    p_auto.add_argument("--lora-dropout", type=float, default=0.05)
    p_auto.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p_auto.add_argument("--use-fp16", action="store_true", default=True)
    p_auto.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    p_auto.add_argument("--enable-fsdp", action="store_true", default=True)
    p_auto.add_argument("--no-fsdp", dest="enable_fsdp", action="store_false")
    p_auto.add_argument("--use-peft", action="store_true", default=True)
    p_auto.add_argument("--no-peft", dest="use_peft", action="store_false")
    p_auto.add_argument("--peft-method", default="lora")
    p_auto.add_argument("--use-fast-kernels", action="store_true", default=True)
    p_auto.add_argument("--gradient-clipping", action="store_true", default=True)
    p_auto.add_argument("--run-validation", action="store_true", default=True)
    p_auto.add_argument("--save-metrics", action="store_true", default=True)
    p_auto.add_argument("--num-workers", type=int, default=2)
    p_auto.add_argument("--val-ratio", type=float, default=0.02)
    p_auto.add_argument("--min-output-chars", type=int, default=24)
    p_auto.add_argument("--max-chars", type=int, default=6000)
    p_auto.add_argument("--max-samples", type=int, default=0)
    p_auto.add_argument("--use-wandb", action="store_true", default=False)
    p_auto.add_argument("--dry-run", action="store_true")
    p_auto.set_defaults(func=cmd_auto)

    p_prepare = sub.add_parser("prepare", help="Chuyen du lieu raw sang SFT JSONL")
    p_prepare.add_argument("--input", nargs="+", default=None)
    p_prepare.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    p_prepare.add_argument("--val-ratio", type=float, default=0.02)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--min-output-chars", type=int, default=24)
    p_prepare.add_argument("--max-chars", type=int, default=6000)
    p_prepare.add_argument("--max-samples", type=int, default=0)
    p_prepare.add_argument("--log-every", type=int, default=2000)
    p_prepare.add_argument("--overwrite", action="store_true")
    p_prepare.set_defaults(func=cmd_prepare)

    p_finetune = sub.add_parser("finetune", help=f"Train FuidAI {FUIDAI_MODEL_ID} tren T4 x{T4_NUM_GPUS}")
    add_shared_finetune_args(p_finetune, include_model=True)
    p_finetune.set_defaults(func=cmd_finetune)

    p_continue = sub.add_parser("continue", help="Tiep tuc train tu checkpoint truoc")
    p_continue.add_argument("--from-run", required=True)
    p_continue.add_argument("--model-name", default=None)
    add_shared_finetune_args(p_continue, include_model=False)
    p_continue.set_defaults(func=cmd_continue)

    p_export = sub.add_parser("export", help="Export FSDP checkpoint sang HuggingFace format")
    p_export.add_argument("--fsdp-checkpoint", required=True)
    p_export.add_argument("--output", default=None)
    p_export.add_argument("--hf-model", default=None)
    p_export.add_argument("--zip-output", action="store_true", default=False)
    p_export.add_argument("--dry-run", action="store_true")
    p_export.set_defaults(func=cmd_export)

    p_chat = sub.add_parser("chat", help="Chat interactive voi FuidAI")
    p_chat.add_argument("--checkpoint", required=True)
    p_chat.add_argument("--max-new-tokens", type=int, default=256)
    p_chat.add_argument("--temperature", type=float, default=0.8)
    p_chat.add_argument("--top-k", type=int, default=50)
    p_chat.add_argument("--top-p", type=float, default=0.9)
    p_chat.add_argument("--repetition-penalty", type=float, default=1.1)
    p_chat.set_defaults(func=cmd_chat)

    p_legacy = sub.add_parser("legacy", help="Chay huan_luyen_legacy.py (fuidai-0.01)")
    p_legacy.add_argument("legacy_args", nargs=argparse.REMAINDER)
    p_legacy.set_defaults(func=cmd_legacy)

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)

    if should_delegate_to_legacy(argv):
        if not argv:
            info(f"[FuidAI {FUIDAI_MODEL_ID}] Khong co command. Dung: python train.py auto")
        if LEGACY_SCRIPT.exists():
            cmd = [sys.executable, str(LEGACY_SCRIPT)] + argv
            return run_command(cmd)
        else:
            info("Chua co legacy script. Dung: python train.py auto --help")
            return 1

    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        info(f"\nFuidAI {FUIDAI_MODEL_ID} | T4 x{T4_NUM_GPUS} | Context {T4_CONTEXT_LENGTH}")
        info("Chay nhanh tren Kaggle: python train.py auto")
        return 1

    try:
        return int(args.func(args) or 0)
    except AppError as ex:
        print(f"[ERROR] {ex}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("Ngat boi nguoi dung.", file=sys.stderr, flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
