"""
chuan_bi_data.py — Chuẩn bị dữ liệu huấn luyện cho Fuid AI

Hỗ trợ 2 format đầu vào:
  • Format JSONL chuẩn  (data_phan_1.jsonl):
        {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

  • Format JSON cũ  (2.json):
        [{"title": "...", "conversation": [{"User": "...", "Assistant": "..."}]}]

Tính năng chính:
  Tuỳ chỉnh system role (nhân cách Fuid) cho từng mẫu
  Chuyển đổi format cũ thành JSON chuẩn
  Gộp nhiều file / thư mục nguồn
  Tạo vocab.json, train.bin, val.bin
  Thống kê chi tiết

Cách dùng:
  python chuan_bi_data.py                         
  python chuan_bi_data.py --input data/ --output data_train/
  python chuan_bi_data.py --no-system            
  python chuan_bi_data.py --convert-only        
  python chuan_bi_data.py --val-ratio 0.05    
"""

import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROLE_CONFIG = {
    "them_system_role": True,

    "system_prompt": (
        "Mày là Fuid, trợ lý AI vui tính do Wbiu phát triển. "
        "Mày nói chuyện kiểu bạn bè thân thiết: thân mật, hài hước, "
        "đôi khi dùng tiếng lóng nhẹ nhưng không thô tục. "
        "Mày thông minh, giải thích rõ ràng và luôn nhiệt tình giúp đỡ. "
        "Khi gặp câu hỏi nguy hiểm hoặc phi đạo đức, mày từ chối nhẹ nhàng."
    ),

    "label_system":    "System",
    "label_user":      "User",
    "label_assistant": "Fuid",
    "sep_turn":        "\n",
    "sep_conv":        "\n\n",
}

DATA_CONFIG = {
    "input_paths": [
        "2.json",
        "data/data_hf_da_chia/data_phan_1.jsonl",
    ],
    "output_dir":   "data_train",
    "val_ratio":    0.10,
    "seed":         42,
    "convert_only": False,
}

DO   = "\033[91m"
XANH = "\033[92m"
VANG = "\033[93m"
CYAN = "\033[96m"
DAM  = "\033[1m"
TAT  = "\033[0m"
SEP  = "═" * 80

_KEY_USER = {
    "user", "User", "USER",
    "question", "Question", "QUESTION",
    "q", "Q",
    "input", "Input", "INPUT",
    "human", "Human", "HUMAN",
    "prompt", "Prompt", "PROMPT",
    "nguoi_dung", "NguoiDung",
    "hoi",
}
_KEY_ASST = {
    "assistant", "Assistant", "ASSISTANT",
    "answer", "Answer", "ANSWER",
    "a", "A",
    "output", "Output", "OUTPUT",
    "bot", "Bot", "BOT",
    "response", "Response", "RESPONSE",
    "fuid", "Fuid", "FUID",
    "ai", "AI",
    "tra_loi", "TraLoi",
    "dap",
}
_KEY_KIEN_THUC = {
    "text", "Text", "TEXT",
    "sentence", "Sentence", "SENTENCE",
    "sentences", "Sentences",
    "noi_dung", "van_ban", "data", "content", "Content",
}


def _nhan_dang_loai_object(obj: dict) -> str:
    if not isinstance(obj, dict):
        return "bo_qua"

    if "messages" in obj:
        msgs = obj["messages"]
        if isinstance(msgs, list):
            roles = {m.get("role", "") for m in msgs if isinstance(m, dict)}
            if "user" in roles or "assistant" in roles:
                return "conversation"

    if "conversation" in obj:
        return "conversation"

    keys = set(obj.keys())
    if keys & _KEY_USER or keys & _KEY_ASST:
        return "conversation"

    if keys & _KEY_KIEN_THUC:
        return "kien_thuc"

    return "bo_qua"


def _nhan_dang_loai_list(data: list) -> str:
    if not data:
        return "bo_qua"
    return _nhan_dang_loai_object(data[0])


def _phat_hien_role(d: dict):
    u, a = None, None
    for k, v in d.items():
        if isinstance(v, str):
            if k in _KEY_USER:
                u = v.strip()
            elif k in _KEY_ASST:
                a = v.strip()
    return u, a


def _normalise_messages(messages: list) -> list:
    ket_qua  = []
    vi_tri   = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role    = msg.get("role", "").strip().lower()
        content = msg.get("content", "")

        if role in ("user", "assistant", "system") and isinstance(content, str) and content.strip():
            ket_qua.append({"role": role, "content": content.strip()})
            if role != "system":
                vi_tri += 1
            continue

        if isinstance(content, str) and content.strip():
            role_gan = "user" if vi_tri % 2 == 0 else "assistant"
            ket_qua.append({"role": role_gan, "content": content.strip()})
            vi_tri += 1
            continue

        u, a = _phat_hien_role(msg)
        if u:
            ket_qua.append({"role": "user",      "content": u})
            vi_tri += 1
        if a:
            ket_qua.append({"role": "assistant", "content": a})
            vi_tri += 1

    return ket_qua


def _doc_jsonl_raw(duong_dan: Path) -> list:
    ket_qua = []
    with open(duong_dan, "r", encoding="utf-8") as f:
        for so_dong, dong in enumerate(f, 1):
            dong = dong.strip()
            if not dong:
                continue
            try:
                ket_qua.append(json.loads(dong))
            except json.JSONDecodeError as e:
                print(f"  {VANG}[Bỏ qua] {duong_dan.name}:{so_dong} — {e}{TAT}")
    return ket_qua


def _chuyen_conversation(obj: dict) -> dict | None:
    if "messages" in obj and isinstance(obj["messages"], list):
        msgs = _normalise_messages(obj["messages"])
        if msgs:
            return {"loai": "conversation", "messages": msgs}
        return None

    if "conversation" in obj and isinstance(obj["conversation"], list):
        msgs = []
        for luot in obj["conversation"]:
            u, a = _phat_hien_role(luot)
            if u:
                msgs.append({"role": "user",      "content": u})
            if a:
                msgs.append({"role": "assistant", "content": a})
        if msgs:
            return {"loai": "conversation", "messages": msgs,
                    "_title": obj.get("title", "")}
        return None

    u, a = _phat_hien_role(obj)
    msgs = []
    if u:
        msgs.append({"role": "user",      "content": u})
    if a:
        msgs.append({"role": "assistant", "content": a})
    if msgs:
        return {"loai": "conversation", "messages": msgs}
    return None


def _chuyen_kien_thuc(obj: dict) -> dict | None:
    for k in _KEY_KIEN_THUC:
        val = obj.get(k)
        if isinstance(val, str) and val.strip():
            return {"loai": "kien_thuc", "noi_dung": val.strip()}
        if isinstance(val, list):
            texts = [v.strip() for v in val if isinstance(v, str) and v.strip()]
            if texts:
                return {"loai": "kien_thuc", "noi_dung": " ".join(texts)}
    return None


def _chuyen_object(obj) -> dict | None:
    if isinstance(obj, str) and obj.strip():
        return {"loai": "kien_thuc", "noi_dung": obj.strip()}
    if not isinstance(obj, dict):
        return None
    loai = _nhan_dang_loai_object(obj)
    if loai == "conversation":
        return _chuyen_conversation(obj)
    if loai == "kien_thuc":
        return _chuyen_kien_thuc(obj)
    return None


def doc_file(duong_dan: Path) -> list:
    suffix = duong_dan.suffix.lower()

    if suffix == ".txt":
        print(f"  {VANG}[Bỏ qua — raw]{TAT} {duong_dan.name}")
        return []

    if suffix == ".jsonl":
        raw_list  = _doc_jsonl_raw(duong_dan)
        ket_qua   = [r for obj in raw_list if (r := _chuyen_object(obj))]
        n_conv    = sum(1 for r in ket_qua if r["loai"] == "conversation")
        n_kien    = sum(1 for r in ket_qua if r["loai"] == "kien_thuc")
        print(f"  {XANH}[JSONL]{TAT} {duong_dan.name} → "
              f"{n_conv} hội thoại / {n_kien} kiến thức")
        return ket_qua

    if suffix == ".json":
        with open(duong_dan, "r", encoding="utf-8") as f:
            data_raw = json.load(f)

        raw_list = data_raw if isinstance(data_raw, list) else [data_raw]
        loai_file = _nhan_dang_loai_list(raw_list)

        if loai_file == "kien_thuc":
            texts = []
            for obj in raw_list:
                if isinstance(obj, dict):
                    for k in _KEY_KIEN_THUC:
                        val = obj.get(k)
                        if isinstance(val, str) and val.strip():
                            texts.append(val.strip())
                            break
                elif isinstance(obj, str) and obj.strip():
                    texts.append(obj.strip())
            ket_qua = [{"loai": "kien_thuc", "noi_dung": t} for t in texts]
            print(f"  {CYAN}[JSON kiến thức]{TAT} {duong_dan.name} → {len(ket_qua)} mục")
            return ket_qua

        ket_qua = [r for obj in raw_list if (r := _chuyen_object(obj))]
        n_conv  = sum(1 for r in ket_qua if r["loai"] == "conversation")
        n_kien  = sum(1 for r in ket_qua if r["loai"] == "kien_thuc")
        print(f"  {XANH}[JSON]{TAT} {duong_dan.name} → "
              f"{n_conv} hội thoại / {n_kien} kiến thức")
        return ket_qua

    print(f"  {VANG}[Bỏ qua]{TAT} {duong_dan.name} — không hỗ trợ")
    return []


def doc_tat_ca(input_paths: list) -> list:
    tat_ca = []
    for duong_dan_str in input_paths:
        p = Path(duong_dan_str)
        if not p.exists():
            print(f"  {VANG}[Không tìm thấy]{TAT} {p}")
            continue
        if p.is_dir():
            files = (sorted(p.glob("**/*.json"))
                   + sorted(p.glob("**/*.jsonl"))
                   + sorted(p.glob("**/*.txt")))
            print(f"\n  Thư mục {p}: {len(files)} file")
            for f in files:
                tat_ca.extend(doc_file(f))
        else:
            tat_ca.extend(doc_file(p))
    return tat_ca


def _co_system_role(messages: list) -> bool:
    return any(m.get("role") == "system" for m in messages)


def them_system_role(mau_list: list, cfg_role: dict) -> list:
    ket_qua  = []
    them_moi = 0
    da_co    = 0

    for mau in mau_list:
        if mau["loai"] != "conversation":
            ket_qua.append(mau)
            continue

        messages = mau.get("messages", [])
        if _co_system_role(messages):
            da_co += 1
            ket_qua.append(mau)
        else:
            sys_msg = {"role": "system", "content": cfg_role["system_prompt"]}
            ket_qua.append({**mau, "messages": [sys_msg] + messages})
            them_moi += 1

    print(f"  System role: {them_moi} mẫu thêm mới, {da_co} mẫu đã có sẵn")
    return ket_qua


def kiem_tra_hop_le(mau_list: list) -> list:
    hop_le = []
    loi    = 0
    for mau in mau_list:
        if mau["loai"] == "kien_thuc":
            if mau.get("noi_dung", "").strip():
                hop_le.append(mau)
            else:
                loi += 1
            continue

        if mau["loai"] == "conversation":
            roles = [m.get("role") for m in mau.get("messages", [])]
            if "user" in roles and "assistant" in roles:
                hop_le.append(mau)
            else:
                loi += 1

    if loi:
        print(f"  {VANG}Bỏ qua {loi} mẫu không hợp lệ{TAT}")
    return hop_le


def _conversation_sang_van_ban(messages: list, cfg_role: dict) -> str:
    nhan = {
        "system":    cfg_role["label_system"],
        "user":      cfg_role["label_user"],
        "assistant": cfg_role["label_assistant"],
    }
    dong_list = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        prefix = nhan.get(role, role.capitalize())
        dong_list.append(f"{prefix}: {content}")
    return cfg_role["sep_turn"].join(dong_list)


def chuyen_sang_van_ban(mau_list: list, cfg_role: dict) -> list:
    ket_qua = []
    for mau in mau_list:
        if mau["loai"] == "conversation":
            vb = _conversation_sang_van_ban(mau.get("messages", []), cfg_role)
            if vb.strip():
                ket_qua.append(vb)
        elif mau["loai"] == "kien_thuc":
            noi_dung = mau.get("noi_dung", "").strip()
            if noi_dung:
                ket_qua.append(noi_dung)
    return ket_qua


def tao_vocab(van_ban_list: list) -> dict:
    ky_tu_set = set()
    for vb in van_ban_list:
        ky_tu_set.update(vb)
    return {ch: idx for idx, ch in enumerate(sorted(ky_tu_set))}


def ma_hoa(van_ban: str, vocab: dict) -> list:
    return [vocab[ch] for ch in van_ban if ch in vocab]


def luu_bin(ids: list, duong_dan: Path):
    np.array(ids, dtype=np.uint16).tofile(str(duong_dan))


def tao_bin_files(van_ban_list: list, vocab: dict,
                  output_dir: Path, val_ratio: float, seed: int, sep: str):
    rng     = random.Random(seed)
    indices = list(range(len(van_ban_list)))
    rng.shuffle(indices)

    n_val   = max(1, int(len(indices) * val_ratio))
    idx_val = set(indices[:n_val])

    train_vb = [van_ban_list[i] for i in indices if i not in idx_val]
    val_vb   = [van_ban_list[i] for i in indices if i     in idx_val]

    train_ids = ma_hoa(sep.join(train_vb), vocab)
    val_ids   = ma_hoa(sep.join(val_vb),   vocab)

    luu_bin(train_ids, output_dir / "train.bin")
    luu_bin(val_ids,   output_dir / "val.bin")

    return len(train_ids), len(val_ids), len(train_vb), len(val_vb)


def luu_jsonl_thong_nhat(mau_list: list, duong_dan: Path):
    with open(duong_dan, "w", encoding="utf-8") as f:
        for mau in mau_list:
            mau_sach = {k: v for k, v in mau.items() if not k.startswith("_")}
            f.write(json.dumps(mau_sach, ensure_ascii=False) + "\n")
    print(f"  {XANH}Đã lưu:{TAT} {duong_dan} ({len(mau_list)} mẫu)")


def luu_vi_du_van_ban(van_ban_list: list, duong_dan: Path, so_vi_du: int = 3):
    with open(duong_dan, "w", encoding="utf-8") as f:
        f.write(f"{'═'*80}\n")
        f.write(f"VÍ DỤ ({so_vi_du}/{len(van_ban_list)} mẫu)\n")
        f.write(f"{'═'*80}\n\n")
        for i, vb in enumerate(van_ban_list[:so_vi_du]):
            f.write(f"── Mẫu {i+1} {'─'*60}\n{vb}\n\n")
    print(f"  {XANH}Ví dụ:{TAT} {duong_dan}")


def chuan_bi(data_cfg: dict = None, role_cfg: dict = None):
    if data_cfg is None:
        data_cfg = DATA_CONFIG
    if role_cfg is None:
        role_cfg = ROLE_CONFIG

    output_dir = Path(data_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{CYAN}{DAM}{SEP}{TAT}")
    print(f"{CYAN}{DAM}  FUID AI — CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN{TAT}")
    print(f"{CYAN}{DAM}{SEP}{TAT}\n")
    print(f"  System role   : {'BẬT' if role_cfg['them_system_role'] else 'TẮT'}")
    print(f"  Đầu ra        : {output_dir}")
    print(f"  Tỷ lệ val     : {data_cfg['val_ratio']*100:.0f}%")
    print(f"  Chỉ chuyển đổi: {'CÓ' if data_cfg.get('convert_only') else 'KHÔNG'}")

    print(f"\n{DAM}[1/5] Đọc dữ liệu nguồn{TAT}")
    mau_list = doc_tat_ca(data_cfg["input_paths"])
    n_conv  = sum(1 for m in mau_list if m["loai"] == "conversation")
    n_kien  = sum(1 for m in mau_list if m["loai"] == "kien_thuc")
    print(f"  Tổng: {len(mau_list)} mẫu  ({n_conv} hội thoại, {n_kien} kiến thức)")

    if not mau_list:
        print(f"\n{DO}Không có dữ liệu. Kiểm tra lại input_paths.{TAT}")
        return

    print(f"\n{DAM}[2/5] Kiểm tra & lọc{TAT}")
    mau_list = kiem_tra_hop_le(mau_list)
    n_conv   = sum(1 for m in mau_list if m["loai"] == "conversation")
    n_kien   = sum(1 for m in mau_list if m["loai"] == "kien_thuc")
    print(f"  Còn lại: {len(mau_list)} mẫu  ({n_conv} hội thoại, {n_kien} kiến thức)")

    print(f"\n{DAM}[3/5] Xử lý system role{TAT}")
    if role_cfg["them_system_role"]:
        mau_list = them_system_role(mau_list, role_cfg)
    else:
        print("  System role bị tắt — giữ nguyên data")

    print(f"\n{DAM}[4/5] Lưu file thống nhất{TAT}")
    jsonl_path = output_dir / "merged.jsonl"
    luu_jsonl_thong_nhat(mau_list, jsonl_path)

    if data_cfg.get("convert_only"):
        print(f"\n{DAM}[5/5] Bỏ qua tokenise (convert-only){TAT}")
        print(f"\n{XANH}{DAM}  HOÀN TẤT — {jsonl_path}{TAT}\n")
        return

    print(f"\n{DAM}[5/5] Tokenise & tạo file nhị phân{TAT}")
    van_ban_list = chuyen_sang_van_ban(mau_list, role_cfg)
    print(f"  Tổng văn bản : {len(van_ban_list)}")

    vocab      = tao_vocab(van_ban_list)
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Vocab        : {len(vocab)} ký tự → {vocab_path}")

    n_train, n_val, n_cv_train, n_cv_val = tao_bin_files(
        van_ban_list, vocab, output_dir,
        data_cfg["val_ratio"], data_cfg["seed"],
        role_cfg["sep_conv"],
    )

    luu_vi_du_van_ban(van_ban_list, output_dir / "vi_du_format.txt")

    print(f"\n{XANH}{DAM}{SEP}{TAT}")
    print(f"{XANH}{DAM}  HOÀN TẤT{TAT}")
    print(f"  Văn bản train  : {n_cv_train:,}  |  val: {n_cv_val:,}")
    print(f"  Token train    : {n_train:,}  |  val: {n_val:,}")
    print(f"  Vocab size     : {len(vocab)}")
    print(f"  Đầu ra         : {output_dir}/")
    print(f"    ├── train.bin")
    print(f"    ├── val.bin")
    print(f"    ├── vocab.json")
    print(f"    ├── merged.jsonl")
    print(f"    └── vi_du_format.txt")
    print(f"{XANH}{DAM}{SEP}{TAT}\n")

    if role_cfg["them_system_role"]:
        print(f"{VANG}System prompt:{TAT}")
        print(f'  "{role_cfg["system_prompt"]}"\n')


def _parse_args():
    p = argparse.ArgumentParser(description="Chuẩn bị dữ liệu huấn luyện cho Fuid AI")
    p.add_argument("--input",         nargs="+")
    p.add_argument("--output",        default=None)
    p.add_argument("--no-system",     action="store_true")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--val-ratio",     type=float, default=None)
    p.add_argument("--convert-only",  action="store_true")
    p.add_argument("--seed",          type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args     = _parse_args()
    data_cfg = dict(DATA_CONFIG)
    role_cfg = dict(ROLE_CONFIG)

    if args.input:
        data_cfg["input_paths"] = args.input
    if args.output:
        data_cfg["output_dir"]  = args.output
    if args.val_ratio is not None:
        data_cfg["val_ratio"]   = args.val_ratio
    if args.seed is not None:
        data_cfg["seed"]        = args.seed
    if args.convert_only:
        data_cfg["convert_only"] = True
    if args.no_system:
        role_cfg["them_system_role"] = False
    if args.system_prompt:
        role_cfg["system_prompt"] = args.system_prompt

    chuan_bi(data_cfg, role_cfg)