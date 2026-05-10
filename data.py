import copy
import json
import torch
import importlib
import importlib.util
import importlib.machinery
import itertools
import ast
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, BatchSampler
from unittest.mock import patch
from tqdm import tqdm
from itertools import chain, islice
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence

try:
    import datasets as _hf_datasets
    _load_dataset = _hf_datasets.load_dataset
except ImportError:
    _hf_datasets = None
    _load_dataset = None

# Định nghĩa nội bộ — không cần llama_cookbook
LLAMA_GUARD_3_CATEGORY = [
    "Violent Crimes",
    "Non-Violent Crimes",
    "Sex Crimes",
    "Child Exploitation",
    "Defamation",
    "Specialized Advice",
    "Privacy",
    "Intellectual Property",
    "Indiscriminate Weapons",
    "Hate",
    "Self-Harm",
    "Sexual Content",
    "Elections",
]

class grammar(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_name=None,
    ):
        if _load_dataset is None:
            raise ImportError("Cần cài HuggingFace datasets: pip install datasets")
        try:
            self.dataset = _load_dataset(
                "csv",
                data_files={"train": [csv_name]},
                delimiter=",",
            )
        except Exception as e:
            print("Loading of grammar dataset failed!")
            raise e
        self.tokenizer = tokenizer
        self.print_text = False

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["text"]))
        input_ = example_batch["input"]
        target_ = example_batch["target"]
        prompt = f"Correct this to standard English: {input_}\n---\nCorrected: "
        prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        label_ids = self.tokenizer.encode(target_ + self.tokenizer.eos_token, add_special_tokens=False)
        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }
        return sample

    def __getitem__(self, index):
        return self.convert_to_features(self.dataset["train"][int(index)])

def get_dataset(dataset_config, tokenizer, csv_name=None):
    if csv_name is None:
        currPath = Path.cwd() / "datasets_grammar" / "grammar_train.csv"
        print(f"Loading dataset {currPath}")
        csv_name = str(currPath)
    dataset = grammar(
        tokenizer=tokenizer,
        csv_name=csv_name,
    )
    return dataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.ann = json.load(open(dataset_config.data_path))
        eval_length = int(len(self.ann)/20)
        if partition == "train":
            self.ann = self.ann[eval_length:]
        else:
            self.ann = self.ann[:eval_length]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }

def load_module_from_py_file(py_file: str) -> object:
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module

def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"
    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")
    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")
    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e

def get_data_collator(dataset_processer, dataset_config):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_data_collator"
    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")
    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")
    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_processer)
    except AttributeError:
        return None

@patch('builtins.input', return_value="N")
def load_samsum(split, _):
    if _load_dataset is None:
        raise ImportError("Cần cài HuggingFace datasets: pip install datasets")
    try:
        ds = _load_dataset("knkarthick/samsum", split=split)
    except ValueError as e:
        if "trust_remote_code" in str(e):
            raise ValueError("HF_DATASETS_TRUST_REMOTE_CODE env variable to True.") from e
        else:
            raise e
    return ds

def get_preprocessed_samsum(dataset_config, tokenizer, split):
    dataset = load_samsum(split)
    prompt = f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n"
    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)
        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }
        return sample
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

def tokenize_prompt_and_labels(full_prompt, tokenizer):
        prompt_tokens = tokenizer.encode(full_prompt)
        combined_tokens = {
            "input_ids": list(prompt_tokens),
            "labels": list(prompt_tokens)
        }
        return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))

@dataclass
class Category:
    name: str
    description: str

@dataclass
class Guidelines:
    categories: Sequence[Category]
    category_code_prefix: str = "S"

class ExplanationPosition(Enum):
    BEFORE_DECISION = 0
    AFTER_DECISION = 1

@dataclass
class LlamaGuardPromptConfigs:
    instructions_format_string: str
    should_include_category_descriptions: bool
    should_shuffle_category_codes: bool = True

@dataclass
class LlamaGuardGenerationConfigs:
    should_list_violated_codes: bool
    explanation_position: Optional[ExplanationPosition]

@dataclass
class AugmentationConfigs:
    should_add_examples_with_dropped_nonviolated_prompt_categories: bool = True
    should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories: bool = False
    explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories: Optional[str] = None

@dataclass
class FormatterConfigs:
    guidelines: Guidelines
    llama_guard_prompt_configs: LlamaGuardPromptConfigs
    llama_guard_generation_configs: LlamaGuardGenerationConfigs
    augmentation_configs: AugmentationConfigs
    random_seed: int = 42

@dataclass
class TrainingExample:
    prompt: str
    response: str
    violated_category_codes: List[str]
    label: Literal["safe", "unsafe"]
    explanation: Optional[str] = None

def mapTcCategoriesToLGCategories(TcCategoriesString):
    TcCategories = ast.literal_eval(TcCategoriesString)
    if(len(TcCategories)==0):
         return None
    ranked = sorted(TcCategories, key=lambda x: x[1], reverse=True)
    primary = ranked[0][0] if len(ranked) else None
    TcMapping = {
        "sexual":"012", "violence":"01", "sexual/minors":"04", "self-harm/intent":"011",
        "hate":"010", "harassment":"010", "self-harm":"011", "self-harm/instructions":"011",
        "violence/graphic":"01", "harassment/threatening":"010", "hate/threatening":"010"
    }
    return TcMapping[primary]

def create_formatted_finetuning_examples(training_examples, formatter_configs):
    _verify_formatter_configs(formatter_configs)
    random.seed(formatter_configs.random_seed)
    indices_of_all_categories = range(len(formatter_configs.guidelines.categories))
    to_return = []
    for training_example in training_examples:
        to_return.append(_create_formatted_finetuning_example(training_example, formatter_configs, list(indices_of_all_categories)))
        _maybe_add_data_augmentations_for_example(training_example, to_return, indices_of_all_categories, formatter_configs)
    return to_return

def _verify_formatter_configs(formatter_configs):
    if (formatter_configs.augmentation_configs.should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories == True
        and formatter_configs.llama_guard_generation_configs.explanation_position is not None
        and formatter_configs.augmentation_configs.explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories is None):
        raise ValueError("Missing explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories.")

def _create_formatted_finetuning_example(training_example, formatter_configs, category_indices):
    if formatter_configs.llama_guard_prompt_configs.should_shuffle_category_codes:
        random.shuffle(category_indices)
    else:
        category_indices = sorted(category_indices)
    llama_guard_prompt = _create_llama_guard_prompt(training_example, category_indices, formatter_configs)
    llama_guard_generation = _create_llama_guard_generation(training_example, category_indices, formatter_configs)
    return f"{llama_guard_prompt} {llama_guard_generation}"

def _create_llama_guard_prompt(training_example, category_indices, formatter_configs):
    full_guidelines_text = ""
    for idx, original_idx in enumerate(category_indices):
        category = formatter_configs.guidelines.categories[original_idx]
        newline = "\n" if idx > 0 else ""
        full_guidelines_text += f"{newline}{formatter_configs.guidelines.category_code_prefix}{idx + 1}: {category.name}. "
        if formatter_configs.llama_guard_prompt_configs.should_include_category_descriptions:
            full_guidelines_text += f"\n{category.description}"
    conversation = {"human": training_example.prompt}
    if training_example.response != "N/A":
        conversation["chatbot"] = training_example.response
    return formatter_configs.llama_guard_prompt_configs.instructions_format_string.format_map({
        "guidelines": full_guidelines_text,
        "conversation": "\n\n".join([f"{s}: {m}" for s, m in conversation.items()]),
    })

def _create_llama_guard_generation(training_example, category_indices, formatter_configs):
    to_return = training_example.label
    if training_example.label == "unsafe" and formatter_configs.llama_guard_generation_configs.should_list_violated_codes:
        violated_indices = set(_convert_category_codes_to_indices(training_example.violated_category_codes, formatter_configs))
        mapping = {orig: f"{formatter_configs.guidelines.category_code_prefix}{rewrit + 1}" for rewrit, orig in enumerate(category_indices)}
        rewritten_codes = sorted([mapping[v_idx] for v_idx in violated_indices if v_idx in mapping])
        to_return += "\n" + ",".join(rewritten_codes)
    if formatter_configs.llama_guard_generation_configs.explanation_position == ExplanationPosition.BEFORE_DECISION:
        to_return = f"Explanation: {training_example.explanation}\n{to_return}"
    elif formatter_configs.llama_guard_generation_configs.explanation_position == ExplanationPosition.AFTER_DECISION:
        to_return = f"{to_return}\nExplanation: {training_example.explanation}"
    return to_return

def _convert_category_codes_to_indices(codes, formatter_configs):
    return [int(code.lstrip(formatter_configs.guidelines.category_code_prefix)) - 1 for code in codes]

def _maybe_add_data_augmentations_for_example(training_example, formatted_examples, all_indices, formatter_configs):
    violated_indices = _convert_category_codes_to_indices(training_example.violated_category_codes, formatter_configs)
    nonviolated_indices = list(set(all_indices) - set(violated_indices))
    if formatter_configs.augmentation_configs.should_add_examples_with_dropped_nonviolated_prompt_categories:
        num_drop = random.randint(0, len(nonviolated_indices))
        if num_drop == len(all_indices): num_drop -= 1
        retained = list(set(all_indices) - set(random.sample(nonviolated_indices, num_drop)))
        formatted_examples.append(_create_formatted_finetuning_example(training_example, formatter_configs, retained))
    if training_example.label == "unsafe" and formatter_configs.augmentation_configs.should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories:
        retained_safe = list(set(all_indices) - set(violated_indices) - set(random.sample(nonviolated_indices, random.randint(0, len(nonviolated_indices) - 1))))
        safe_copy = copy.deepcopy(training_example)
        safe_copy.label, safe_copy.violated_category_codes = "safe", []
        safe_copy.explanation = formatter_configs.augmentation_configs.explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories
        formatted_examples.append(_create_formatted_finetuning_example(safe_copy, formatter_configs, retained_safe))

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        buffer = {"input_ids": [], "attention_mask": [], "labels": []}
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
    def __getitem__(self, idx): return self.samples[idx]
    def __len__(self): return len(self.samples)

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size, self.drop_last, self.shuffle = batch_size, drop_last, shuffle
    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last: ids = ids[:len(ids) // self.batch_size * self.batch_size]
        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]
        if self.shuffle: random.shuffle(batches)
        for b in batches: yield b
    def __len__(self):
        return len(self.lengths) // self.batch_size + (0 if self.drop_last or len(self.lengths) % self.batch_size == 0 else 1)

class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle)
        self.num_replicas, self.rank = num_replicas, rank
    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)
    def __len__(self): return len(self.batch_sampler) // self.num_replicas

# ─── FuidAI Chat Template ─────────────────────────────────────────────────────
#
# Định dạng hỗ trợ:
#
# 1) Định dạng CŨ  — instruction/input/output (alpaca-style)
#    {"instruction": "...", "input": "...", "output": "..."}
#
# 2) Định dạng MỚI — messages (chat-style)
#    {"messages": [
#        {"role": "system",    "content": "..."},
#        {"role": "user",      "content": "..."},
#        {"role": "assistant", "content": "..."}
#    ]}
#
# Chat template tokens (thêm vào vocab nếu chưa có):
#   <|system|>      — bắt đầu turn system
#   <|user|>        — bắt đầu turn user
#   <|assistant|>   — bắt đầu turn assistant
#   <|end|>         — kết thúc mỗi turn
#
# Chỉ phần assistant được tính loss (label masking).

CHAT_ROLES = {
    "system":    "<|system|>",
    "user":      "<|user|>",
    "assistant": "<|assistant|>",
}
CHAT_END_TOKEN = "<|end|>"

PROMPT_TEMPLATE_LEGACY = "### Câu hỏi:\n{instruction}\n{input_section}### Trả lời:\n{output}"


def _build_chat_text(messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """
    Ghép danh sách messages thành một chuỗi theo chat template.

    Trả về:
        full_text      : toàn bộ chuỗi văn bản
        assistant_spans: danh sách (start, end) ký tự của phần assistant content
                         (không gồm token đặc biệt) — dùng để mask labels.
    """
    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    pos = 0

    for msg in messages:
        role    = msg.get("role", "user").lower()
        content = msg.get("content", "").strip()
        prefix  = CHAT_ROLES.get(role, f"<|{role}|>")
        chunk   = f"{prefix}\n{content}\n{CHAT_END_TOKEN}\n"

        if role == "assistant":
            # vị trí bắt đầu content trong chunk
            content_start = pos + len(prefix) + 1          # +1 cho '\n'
            content_end   = content_start + len(content)
            assistant_spans.append((content_start, content_end))

        parts.append(chunk)
        pos += len(chunk)

    return "".join(parts), assistant_spans


def _load_jsonl_or_json(path: Path) -> list[dict]:
    """Đọc file JSONL hoặc JSON, trả về list[dict]."""
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    # Plain text — wrap thành sample
    text = path.read_text(encoding="utf-8").strip()
    return [{"instruction": "", "output": text}] if text else []


# ─── FuidAI Dataset (dùng TokenizerTV từ mo_hinh.py) ─────────────────────────

class FuidAIDataset(Dataset):
    """
    Dataset đa định dạng cho mô hình fuidai.

    Hỗ trợ hai dạng mẫu:
    • Alpaca-style : {"instruction": "...", "input": "...", "output": "..."}
    • Chat-style   : {"messages": [{"role": "...", "content": "..."}, ...]}

    Với chat-style, chỉ phần assistant được tính loss (label masking).
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        dataset_config,
        tokenizer,
        partition: str = "train",
        context_length: int = 512,
        mask_prompt: bool = True,
    ):
        self.tokenizer      = tokenizer
        self.context_length = context_length
        self.mask_prompt    = mask_prompt   # mask phần prompt (non-assistant) khi True

        data_path = (
            dataset_config.data_path
            if partition == "train"
            else getattr(dataset_config, "val_path", dataset_config.data_path)
        )

        self.samples: list[dict] = _load_jsonl_or_json(Path(data_path))
        if not self.samples:
            print(f"[FuidAIDataset] Cảnh báo: {data_path} không tồn tại hoặc rỗng.")

    # ── Tokenize helper ───────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize văn bản bằng TokenizerTV hoặc HuggingFace tokenizer."""
        if hasattr(self.tokenizer, "ma_hoa"):
            return self.tokenizer.ma_hoa(text, them_bos_eos=False)
        encoded = self.tokenizer(text, add_special_tokens=False)
        return encoded["input_ids"]

    def _char_to_token_spans(
        self, text: str, char_spans: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """
        Chuyển đổi vị trí ký tự → vị trí token (xấp xỉ theo tỉ lệ ký tự).
        Với TokenizerTV (char-level), mỗi ký tự = 1 token nên spans khớp hoàn toàn.
        Với BPE tokenizer, dùng offset_mapping nếu có.
        """
        if hasattr(self.tokenizer, "ma_hoa"):
            # char-level → vị trí token = vị trí ký tự (sau BOS nếu có)
            return char_spans

        # HuggingFace: thử offset_mapping
        try:
            enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]   # list[(char_start, char_end)]
            token_spans = []
            for c_start, c_end in char_spans:
                t_start = next(
                    (i for i, (cs, ce) in enumerate(offsets) if cs >= c_start), None
                )
                t_end = next(
                    (i for i, (cs, ce) in enumerate(offsets) if ce >= c_end), len(offsets)
                )
                if t_start is not None:
                    token_spans.append((t_start, t_end))
            return token_spans
        except Exception:
            return []

    # ── Sample → (ids, label_mask) ────────────────────────────────────────────

    def _process_chat(self, sample: dict) -> tuple[list[int], list[bool]]:
        """
        Xử lý mẫu chat-style.
        Trả về (ids, is_assistant) với is_assistant[i] = True nếu token i thuộc assistant.
        """
        messages = sample["messages"]
        full_text, assistant_spans = _build_chat_text(messages)

        ids = self._tokenize(full_text)
        token_spans = self._char_to_token_spans(full_text, assistant_spans)

        is_assistant = [False] * len(ids)
        for t_start, t_end in token_spans:
            for i in range(t_start, min(t_end, len(ids))):
                is_assistant[i] = True

        return ids, is_assistant

    def _process_legacy(self, sample: dict) -> tuple[list[int], list[bool]]:
        """
        Xử lý mẫu alpaca-style.
        Mask toàn bộ phần prompt; chỉ phần output tính loss.
        """
        instruction  = sample.get("instruction", "").strip()
        input_text   = sample.get("input", "").strip()
        output       = sample.get("output", "").strip()
        input_section = f"### Ngữ cảnh:\n{input_text}\n" if input_text else ""

        prompt = PROMPT_TEMPLATE_LEGACY.format(
            instruction=instruction,
            input_section=input_section,
            output="",
        )
        full_text = prompt + output

        ids          = self._tokenize(full_text)
        prompt_ids   = self._tokenize(prompt)
        prompt_len   = len(prompt_ids)

        is_assistant = [i >= prompt_len for i in range(len(ids))]
        return ids, is_assistant

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        if "messages" in sample:
            ids, is_assistant = self._process_chat(sample)
        else:
            ids, is_assistant = self._process_legacy(sample)

        # Thêm BOS/EOS nếu dùng TokenizerTV
        if hasattr(self.tokenizer, "char2idx"):
            bos = self.tokenizer.char2idx.get("<BOS>", 0)
            eos = self.tokenizer.char2idx.get("<EOS>", 0)
            ids          = [bos] + ids + [eos]
            is_assistant = [False] + is_assistant + [True]   # EOS tính vào loss

        # Cắt về context_length + 1 (để shift)
        max_len = self.context_length + 1
        if len(ids) > max_len:
            ids          = ids[:max_len]
            is_assistant = is_assistant[:max_len]

        # Shift: input = ids[:-1], labels = ids[1:]
        input_ids    = ids[:-1]
        label_tokens = ids[1:]
        label_mask   = is_assistant[1:]   # mask tương ứng với labels

        # Áp dụng label masking
        if self.mask_prompt:
            labels = [
                tok if mask else self.IGNORE_INDEX
                for tok, mask in zip(label_tokens, label_mask)
            ]
        else:
            labels = list(label_tokens)

        # Padding về context_length
        pad_len   = self.context_length - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        labels    = labels    + [self.IGNORE_INDEX] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def get_fuidai_dataset(
    dataset_config,
    tokenizer,
    split: str = "train",
    context_length: int = 512,
    mask_prompt: bool = True,
) -> FuidAIDataset:
    """Hàm factory để finetuning.py gọi."""
    partition = "train" if split == "train" else "val"
    return FuidAIDataset(
        dataset_config,
        tokenizer,
        partition=partition,
        context_length=context_length,
        mask_prompt=mask_prompt,
    )
