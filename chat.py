import os
import sys
import time

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("Thiếu thư viện 'torch'. Hãy cài PyTorch trước khi chạy.") from exc

from mo_hinh import fuidai, TokenizerTV

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

THIET_BI       = "cpu"
THU_MUC_GOC    = os.path.dirname(os.path.abspath(__file__))
THU_MUC_DAU_RA = os.path.join(THU_MUC_GOC, "dau_ra_fuidai")

MAX_LUOT_GIU = 6

print("Đang đánh thức Fuid...")

duong_dan_ckpt  = os.path.join(THU_MUC_DAU_RA, "fuid_best.pt")
duong_dan_vocab = os.path.join(THU_MUC_DAU_RA, "vocab.json")

if not os.path.exists(duong_dan_ckpt):
    raise FileNotFoundError(f"Không tìm thấy điểm kiểm tra tại {duong_dan_ckpt}")
if not os.path.exists(duong_dan_vocab):
    raise FileNotFoundError(f"Không tìm thấy từ vựng tại {duong_dan_vocab}")

tokenizer = TokenizerTV.tai(duong_dan_vocab)

ban_sao = torch.load(duong_dan_ckpt, map_location=THIET_BI)
cfg_luu = ban_sao.get("cfg", {})

block_size = cfg_luu.get("block_size", 512)

mo_hinh = fuidai(
    kich_thuoc_tu_vung=tokenizer.vocab_size,
    d_model=cfg_luu.get("d_model", 780),
    so_lop=cfg_luu.get("n_layers", 14),
    so_dau=cfg_luu.get("n_heads", 12),
    kich_thuoc_khoi=block_size,
    ty_le_bo_qua=0.0,
).to(THIET_BI)

mo_hinh.load_state_dict(ban_sao["model_state"])
mo_hinh.eval()

print("Fuid đã sẵn sàng. Nhập 'quit' để thoát.\n")

lich_su: list[tuple[str, str]] = []


def xay_prompt(lich_su: list, cau_moi: str, block_size: int) -> str:
    ket_thuc = f"User: {cau_moi}\nFuid: "

    cac_luot = []
    for vai, noi_dung in lich_su[-(MAX_LUOT_GIU * 2):]:
        cac_luot.append(f"{vai}: {noi_dung}")

    while cac_luot:
        ung_vien = "\n".join(cac_luot) + "\n" + ket_thuc
        if len(ung_vien) <= block_size - 20:
            return ung_vien
        cac_luot.pop(0)

    return ket_thuc[-block_size + 20:]


def clean_output(text: str) -> str:
    for stop in ["\nUser:", "\n\n"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text.strip()


while True:
    try:
        cau_moi = input("Bạn: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nTạm biệt!")
        break

    if not cau_moi:
        continue
    if cau_moi.lower() in {"quit", "exit", "thoat"}:
        print("Fuid: Tạm biệt")
        break

    prompt = xay_prompt(lich_su, cau_moi, block_size)

    ids = tokenizer.ma_hoa(prompt, them_bos_eos=False)

    if len(ids) > block_size - 10:
        ids = ids[-(block_size - 10):]

    chi_muc = torch.tensor(ids, dtype=torch.long, device=THIET_BI).unsqueeze(0)

    print("Fuid: ", end="", flush=True)

    cac_token_sinh: list[int] = []
    van_ban_tich_luy = ""
    dung = False

    with torch.no_grad():
        for token_moi in mo_hinh.sinh_van_ban_streaming(
            chi_muc,
            so_token_moi=200,
            nhiet_do=0.8,
            top_k=30,
        ):
            eos_idx = tokenizer.char2idx.get(tokenizer.EOS, -1)
            if token_moi == eos_idx:
                break

            ky_tu = tokenizer.giai_ma([token_moi], bo_specials=True)
            van_ban_tich_luy += ky_tu

            if "\nUser:" in van_ban_tich_luy or van_ban_tich_luy.endswith("\nUser:"):
                dung = True
                break

            if ky_tu == "\n":
                so_dong = van_ban_tich_luy.count("\n")
                if so_dong >= 2:
                    break

            cac_token_sinh.append(token_moi)
            print(ky_tu, end="", flush=True)
            time.sleep(0.0005)

    print()

    phan_hoi = clean_output(van_ban_tich_luy)

    if not phan_hoi:
        phan_hoi = "..."

    lich_su.append(("User", cau_moi))
    lich_su.append(("Fuid", phan_hoi))

    print()