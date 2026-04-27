"""
chuan_bi_data.py — Tiền xử lý dữ liệu JSON → train.bin / val.bin
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hỗ trợ TẤT CẢ định dạng JSON:

   {"text": "..."}
   {"instruction": "...", "input": "...", "output": "..."}
   {"question": "...", "answer": "..."}
   {"user": "...", "assistant": "..."}
   {"prompt": "...", "response": "..."}
   {"input": "...", "output": "..."}
   Bất kỳ tag nào khác — TỰ ĐỘNG ghép tất cả giá trị

Cách dùng:
  python chuan_bi_data.py                          # dùng thư mục mặc định
  python chuan_bi_data.py --data /path/to/jsons    # chỉ định thư mục JSON
  python chuan_bi_data.py --out  /path/to/output   # chỉ định thư mục đầu ra
  python chuan_bi_data.py --val_ratio 0.05         # tỉ lệ tập xác thực (mặc định 5%)
  python chuan_bi_data.py --format instruct        # ép buộc định dạng instruction
  python chuan_bi_data.py --sep "<|sep|>"          # dấu phân cách giữa các tag
  python chuan_bi_data.py --show                   # xem trước 3 mẫu đầu
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# ─── Màu sắc terminal ────────────────────────────────────────────────────────
DO   = "\033[91m"
XANH = "\033[92m"
VANG = "\033[93m"
CYAN = "\033[96m"
DAM  = "\033[1m"
TAT  = "\033[0m"
SEP  = "=" * 80

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent

# ─── Cấu hình mặc định ───────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir":   None,   # None = tự tìm trong PROJECT_ROOT
    "output_dir": None,   # None = PROJECT_ROOT/data_train
    "val_ratio":  0.05,   # 5% dữ liệu dành cho xác thực
    "format":     "auto", # auto | text | instruct | qa | chat
    "sep":        "\n",   # phân cách giữa các trường khi ghép
    "shuffle":    True,   # xáo trộn trước khi chia train/val
    "seed":       42,
    "show":       False,  # xem trước một số mẫu
    "show_n":     3,
}

def _fmt_text(item: dict, sep: str) -> str:
    """{"text": "..."}"""
    return item["text"].strip()


def _fmt_instruct(item: dict, sep: str) -> str:
    """{"instruction": "...", "input": "...", "output": "...", "type": "..."}
    Trường "type" (phong cách/thể loại) là tuỳ chọn — được thêm vào đầu nếu có.
    """
    parts = []
    loai  = item.get("type", "").strip()
    inst  = item.get("instruction", "").strip()
    inp   = item.get("input", "").strip()
    out   = item.get("output", "").strip()
    if loai:
        parts.append(f"### Phong cách:\n{loai}")
    if inst:
        parts.append(f"### Lệnh:\n{inst}")
    if inp:
        parts.append(f"### Đầu vào:\n{inp}")
    if out:
        parts.append(f"### Trả lời:\n{out}")
    return sep.join(parts)


def _fmt_qa(item: dict, sep: str) -> str:
    """{"question": "...", "answer": "..."}"""
    q = item.get("question", "").strip()
    a = item.get("answer",   "").strip()
    parts = []
    if q: parts.append(f"User: {q}")
    if a: parts.append(f"Fuid: {a}")
    return sep.join(parts)


def _fmt_chat(item: dict, sep: str) -> str:
    """{"user": "...", "assistant": "..."} hoặc {"prompt": "...", "response": "..."}"""
    u = (item.get("user")    or item.get("prompt")   or "").strip()
    a = (item.get("assistant") or item.get("response") or "").strip()
    parts = []
    if u: parts.append(f"User: {u}")
    if a: parts.append(f"Fuid: {a}")
    return sep.join(parts)


def _fmt_input_output(item: dict, sep: str) -> str:
    """{"input": "...", "output": "..."}"""
    i = item.get("input",  "").strip()
    o = item.get("output", "").strip()
    parts = []
    if i: parts.append(f"User: {i}")
    if o: parts.append(f"Fuid: {o}")
    return sep.join(parts)


def _fmt_auto(item: dict, sep: str) -> str:
    """Tự động ghép TẤT CẢ giá trị string trong dict"""
    # Thứ tự ưu tiên kiểm tra định dạng đã biết
    keys = set(item.keys())

    if "text" in keys:
        return _fmt_text(item, sep)

    # data17.json và các định dạng instruction (có hoặc không có "type")
    if "instruction" in keys and "output" in keys:
        return _fmt_instruct(item, sep)

    if "question" in keys and "answer" in keys:
        return _fmt_qa(item, sep)

    if ("user" in keys or "prompt" in keys) and \
       ("assistant" in keys or "response" in keys):
        return _fmt_chat(item, sep)

    if "input" in keys and "output" in keys and "instruction" not in keys:
        return _fmt_input_output(item, sep)

    # Fallback: ghép tất cả giá trị string theo thứ tự key
    parts = []
    for k, v in item.items():
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
        elif isinstance(v, list):
            # Xử lý conversations dạng list
            for msg in v:
                if isinstance(msg, dict):
                    role    = msg.get("role", msg.get("from", ""))
                    content = msg.get("content", msg.get("value", ""))
                    if role and content:
                        parts.append(f"{role}: {content.strip()}")
                    elif content:
                        parts.append(content.strip())
    return sep.join(parts)


FORMAT_MAP = {
    "text":           _fmt_text,
    "instruct":       _fmt_instruct,
    "instruction":    _fmt_instruct,
    "typed_instruct": _fmt_instruct,   # alias rõ ràng cho data17.json (instruction+type)
    "qa":             _fmt_qa,
    "chat":           _fmt_chat,
    "input_output":   _fmt_input_output,
    "auto":           _fmt_auto,
}


# ─── Đọc file JSON ────────────────────────────────────────────────────────────

def doc_json(duong_dan: Path) -> list[dict]:
    """Đọc file JSON, hỗ trợ: array [], object {}, jsonl (1 object/dòng)."""
    with open(duong_dan, "r", encoding="utf-8") as f:
        noi_dung = f.read().strip()

    if not noi_dung:
        return []

    # Thử parse toàn bộ
    try:
        du_lieu = json.loads(noi_dung)
        if isinstance(du_lieu, list):
            return du_lieu
        if isinstance(du_lieu, dict):
            return [du_lieu]
    except json.JSONDecodeError:
        pass

    # Thử JSONL (mỗi dòng 1 JSON)
    ket_qua = []
    for so_dong, dong in enumerate(noi_dung.splitlines(), 1):
        dong = dong.strip()
        if not dong:
            continue
        try:
            obj = json.loads(dong)
            if isinstance(obj, dict):
                ket_qua.append(obj)
            elif isinstance(obj, list):
                ket_qua.extend(obj)
        except json.JSONDecodeError as e:
            print(f"  {VANG}[Cảnh báo] Dòng {so_dong} trong {duong_dan.name}: {e}{TAT}")

    return ket_qua


def quet_thu_muc_json(thu_muc: Path) -> list[Path]:
    """Quét đệ quy tất cả file .json và .jsonl trong thư mục."""
    files = []
    for ext in ("*.json", "*.jsonl"):
        files.extend(sorted(thu_muc.rglob(ext)))
    return files


# ─── Phát hiện định dạng ─────────────────────────────────────────────────────

def phat_hien_dinh_dang(mau: dict) -> str:
    keys = set(mau.keys())
    if "text" in keys:
        return "text"
    # Hỗ trợ data17.json: instruction + output (+ input rỗng + type tuỳ chọn)
    if "instruction" in keys and "output" in keys:
        return "instruct"
    if "question" in keys and "answer" in keys:
        return "qa"
    if ("user" in keys or "prompt" in keys) and \
       ("assistant" in keys or "response" in keys):
        return "chat"
    if "input" in keys and "output" in keys:
        return "input_output"
    return "auto"


# ─── Tokenizer đơn giản (character-level) ────────────────────────────────────

class TokenizerKyTu:
    """Character-level tokenizer — giống TokenizerTV trong mo_hinh.py."""

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.vocab_size = 0

    def xay_dung(self, van_ban: str):
        """Xây dựng vocab từ văn bản."""
        ky_tu = sorted(set(van_ban))
        self.char2idx = {c: i for i, c in enumerate(ky_tu)}
        self.idx2char = {i: c for i, c in enumerate(ky_tu)}
        self.vocab_size = len(ky_tu)
        print(f"  Vocab size: {self.vocab_size} ký tự")

    def them_tu_vocab(self, vocab_dict: dict):
        """Thêm ký tự mới vào vocab đã có (khi train tiếp với data mới)."""
        them = 0
        for c in vocab_dict:
            if c not in self.char2idx:
                idx = self.vocab_size
                self.char2idx[c] = idx
                self.idx2char[idx] = c
                self.vocab_size += 1
                them += 1
        if them:
            print(f"  {CYAN}Thêm {them} ký tự mới vào vocab{TAT}")

    def ma_hoa(self, van_ban: str) -> list[int]:
        return [self.char2idx[c] for c in van_ban if c in self.char2idx]

    def luu(self, duong_dan: Path):
        with open(duong_dan, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)
        print(f"  {XANH}Đã lưu vocab → {duong_dan}{TAT}")

    def tai(self, duong_dan: Path):
        with open(duong_dan, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"  {XANH}Đã tải vocab: {self.vocab_size} ký tự ← {duong_dan}{TAT}")


# ─── Thống kê tag ─────────────────────────────────────────────────────────────

def thong_ke_tag(danh_sach: list[dict]) -> dict[str, int]:
    dem = {}
    for item in danh_sach:
        for k in item.keys():
            dem[k] = dem.get(k, 0) + 1
    return dict(sorted(dem.items(), key=lambda x: -x[1]))


# ─── Hàm chính ───────────────────────────────────────────────────────────────

def chuan_bi_data(cfg: dict):
    random.seed(cfg.get("seed", 42))

    # ── Đường dẫn ──────────────────────────────────────────────────────────
    data_dir = Path(cfg["data_dir"]) if cfg.get("data_dir") else None
    if data_dir is None:
        # Tìm thư mục chứa JSON tự động
        for thu_muc_ung_vien in [
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "data_json",
            PROJECT_ROOT / "dataset",
            PROJECT_ROOT,
        ]:
            if any(thu_muc_ung_vien.rglob("*.json")):
                data_dir = thu_muc_ung_vien
                break
        if data_dir is None:
            raise FileNotFoundError(
                "Không tìm thấy file JSON nào. "
                "Hãy dùng --data /path/to/json_folder"
            )

    output_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else \
                 PROJECT_ROOT / "data_train"
    output_dir.mkdir(parents=True, exist_ok=True)

    val_ratio = float(cfg.get("val_ratio", 0.05))
    fmt_name  = cfg.get("format", "auto")
    sep       = cfg.get("sep", "\n")
    do_shuffle = cfg.get("shuffle", True)
    show      = cfg.get("show", False)
    show_n    = int(cfg.get("show_n", 3))

    fmt_fn = FORMAT_MAP.get(fmt_name, _fmt_auto)

    print(f"\n{CYAN}{DAM}{SEP}{TAT}")
    print(f"{CYAN}{DAM}  CHUẨN BỊ DỮ LIỆU — FUID AI{TAT}")
    print(f"{CYAN}{DAM}{SEP}{TAT}")
    print(f"  Thư mục JSON  : {data_dir}")
    print(f"  Đầu ra        : {output_dir}")
    print(f"  Định dạng     : {fmt_name}")
    print(f"  Tỉ lệ val     : {val_ratio*100:.1f}%")
    print(f"  Dấu phân cách : {repr(sep)}\n")

    # ── Quét và đọc tất cả file JSON ───────────────────────────────────────
    files = quet_thu_muc_json(data_dir)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file .json/.jsonl trong: {data_dir}")

    print(f"  {DAM}Tìm thấy {len(files)} file JSON:{TAT}")
    tat_ca_items: list[dict] = []

    for tep in files:
        items = doc_json(tep)
        if not items:
            print(f"  {VANG}[Bỏ qua] {tep.name} — rỗng hoặc lỗi{TAT}")
            continue

        # Phát hiện định dạng nếu auto
        dinh_dang_tep = fmt_name
        if fmt_name == "auto" and items:
            dinh_dang_tep = phat_hien_dinh_dang(items[0])

        print(f"  ✓ {tep.name:40s} → {len(items):5d} mẫu  [{dinh_dang_tep}]")
        tat_ca_items.extend(items)

    print(f"\n  {DAM}Tổng cộng: {len(tat_ca_items):,} mẫu{TAT}")

    # ── Thống kê tag ───────────────────────────────────────────────────────
    tag_stats = thong_ke_tag(tat_ca_items)
    print(f"\n  {DAM}Thống kê tag:{TAT}")
    for tag, dem in tag_stats.items():
        thanh = "█" * min(40, int(40 * dem / len(tat_ca_items)))
        print(f"    {tag:20s}: {dem:6,}  {CYAN}{thanh}{TAT}")

    # ── Chuyển đổi thành văn bản ───────────────────────────────────────────
    print(f"\n  {DAM}Chuyển đổi sang văn bản...{TAT}")
    van_ban_list: list[str] = []
    loi = 0

    for i, item in enumerate(tat_ca_items):
        try:
            # Dùng fmt_auto để tự nhận dạng từng item riêng lẻ
            if fmt_name == "auto":
                van_ban = _fmt_auto(item, sep)
            else:
                van_ban = fmt_fn(item, sep)

            if van_ban.strip():
                van_ban_list.append(van_ban.strip())
            else:
                loi += 1
        except Exception as e:
            loi += 1
            if loi <= 5:
                print(f"  {VANG}[Mẫu {i}] Lỗi: {e}{TAT}")

    print(f"  Văn bản hợp lệ : {len(van_ban_list):,}")
    if loi:
        print(f"  {VANG}Mẫu lỗi/rỗng   : {loi:,}{TAT}")

    if not van_ban_list:
        raise ValueError("Không có văn bản nào sau khi chuyển đổi!")

    # ── Xem trước ──────────────────────────────────────────────────────────
    if show:
        print(f"\n{DAM}  XEM TRƯỚC {show_n} MẪU ĐẦU:{TAT}")
        for i, v in enumerate(van_ban_list[:show_n]):
            print(f"  {'─'*60}")
            print(f"  [Mẫu {i+1}]\n  {v[:400]}")
        print(f"  {'─'*60}\n")

    # ── Xáo trộn và chia train/val ─────────────────────────────────────────
    if do_shuffle:
        random.shuffle(van_ban_list)

    n_val   = max(1, int(len(van_ban_list) * val_ratio))
    n_train = len(van_ban_list) - n_val
    train_vb = van_ban_list[:n_train]
    val_vb   = van_ban_list[n_train:]

    print(f"\n  Train: {n_train:,} mẫu  |  Val: {n_val:,} mẫu")

    # ── Ghép văn bản với token phân cách ──────────────────────────────────
    PHAN_CACH = "\n\n"
    train_text = PHAN_CACH.join(train_vb)
    val_text   = PHAN_CACH.join(val_vb)

    print(f"  Train chars    : {len(train_text):,}")
    print(f"  Val chars      : {len(val_text):,}")

    # ── Xây dựng vocab ─────────────────────────────────────────────────────
    vocab_path = output_dir / "vocab.json"
    tokenizer  = TokenizerKyTu()

    if vocab_path.exists():
        print(f"\n  {CYAN}Tìm thấy vocab.json cũ — mở rộng thêm ký tự mới...{TAT}")
        tokenizer.tai(vocab_path)
        ky_tu_moi = set(train_text + val_text) - set(tokenizer.char2idx.keys())
        if ky_tu_moi:
            tokenizer.them_tu_vocab({c: 0 for c in sorted(ky_tu_moi)})
        tokenizer.luu(vocab_path)
    else:
        print(f"\n  Xây dựng vocab mới...")
        tokenizer.xay_dung(train_text + val_text)
        tokenizer.luu(vocab_path)

    # ── Tokenize ───────────────────────────────────────────────────────────
    print(f"\n  {DAM}Tokenize...{TAT}")

    def luu_bin(van_ban: str, duong_dan: Path, ten: str):
        ids = tokenizer.ma_hoa(van_ban)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(duong_dan)
        kb  = duong_dan.stat().st_size / 1024
        print(f"  {XANH}✓ {ten:12s}: {len(ids):,} token → {duong_dan.name} "
              f"({kb:.1f} KB){TAT}")
        return len(ids)

    n_train_tok = luu_bin(train_text, output_dir / "train.bin", "train")
    n_val_tok   = luu_bin(val_text,   output_dir / "val.bin",   "val")

    # ── Lưu metadata ───────────────────────────────────────────────────────
    metadata = {
        "tong_mau":          len(van_ban_list),
        "mau_train":         n_train,
        "mau_val":           n_val,
        "token_train":       n_train_tok,
        "token_val":         n_val_tok,
        "vocab_size":        tokenizer.vocab_size,
        "dinh_dang":         fmt_name,
        "tag_stats":         tag_stats,
        "files":             [str(f) for f in files],
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ── Tổng kết ───────────────────────────────────────────────────────────
    print(f"\n{XANH}{DAM}{SEP}{TAT}")
    print(f"{XANH}{DAM}  HOÀN TẤT CHUẨN BỊ DỮ LIỆU{TAT}")
    print(f"{XANH}{DAM}  ├─ vocab.json    : {tokenizer.vocab_size} ký tự{TAT}")
    print(f"{XANH}{DAM}  ├─ train.bin     : {n_train_tok:,} token  ({n_train:,} mẫu){TAT}")
    print(f"{XANH}{DAM}  ├─ val.bin       : {n_val_tok:,} token  ({n_val:,} mẫu){TAT}")
    print(f"{XANH}{DAM}  └─ metadata.json : thống kê đầy đủ{TAT}")
    print(f"{XANH}{DAM}{SEP}{TAT}\n")
    print(f"  {DAM}Bước tiếp theo:{TAT}")
    print(f"    python huan_luyen.py")
    print(f"    # hoặc train tiếp:")
    print(f"    python hlpb2.py\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(
        description="Tiền xử lý JSON → train.bin/val.bin cho FUID AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data",      type=str,   default=None,
                    help="Thư mục chứa file JSON/JSONL")
    ap.add_argument("--out",       type=str,   default=None,
                    help="Thư mục đầu ra (mặc định: ./data_train)")
    ap.add_argument("--val_ratio", type=float, default=0.05,
                    help="Tỉ lệ tập xác thực (mặc định: 0.05)")
    ap.add_argument("--format",    type=str,   default="auto",
                    choices=list(FORMAT_MAP.keys()),
                    help="Định dạng JSON (mặc định: auto)")
    ap.add_argument("--sep",       type=str,   default="\n",
                    help="Dấu phân cách giữa các trường (mặc định: newline)")
    ap.add_argument("--no_shuffle", action="store_true",
                    help="Không xáo trộn dữ liệu")
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--show",      action="store_true",
                    help="Xem trước một số mẫu sau khi chuyển đổi")
    ap.add_argument("--show_n",    type=int,   default=3)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = {
        "data_dir":   args.data,
        "output_dir": args.out,
        "val_ratio":  args.val_ratio,
        "format":     args.format,
        "sep":        args.sep.replace("\\n", "\n").replace("\\t", "\t"),
        "shuffle":    not args.no_shuffle,
        "seed":       args.seed,
        "show":       args.show,
        "show_n":     args.show_n,
    }
    chuan_bi_data(cfg)