"""
hlpb2.py — Huấn luyện tiếp (Resume) từ file .pt
Dùng khi muốn train tiếp từ một checkpoint đã có, với data mới hoặc data cũ.

Cách dùng:
  python hlpb2.py                          # tự tìm fuid_best.pt
  python hlpb2.py --ckpt latest            # dùng fuid_latest.pt
  python hlpb2.py --ckpt best              # dùng fuid_best.pt (mặc định)
  python hlpb2.py --ckpt path/to/file.pt   # chỉ định file cụ thể
  python hlpb2.py --epochs 200             # train thêm 200 vòng
  python hlpb2.py --data /path/data_moi    # train tiếp với data mới
  python hlpb2.py --lr 1e-4                # override learning rate
"""

import argparse
import json
import math
import re
import sys
import time
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise SystemExit("Thiếu PyTorch. Vui lòng cài đặt: pip install torch")

from mo_hinh import fuidai, TokenizerTV

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent

RESUME_CONFIG = {
    "ckpt_path":                "best",     
    "data_dir":                 None,      
    "output_dir":               None,       
    "them_epochs":              500,        

    "learning_rate":            None,      
    "min_lr":                   None,
    "batch_size":               None,
    "dropout":                  None,
    "grad_clip":                None,
    "use_cosine_lr":            None,
    "warmup_steps":             0,      
    "warmup_epochs":            0,
    "val_every_n_steps":        None,
    "val_steps":                None,
    "save_every_n_epochs":      None,
    "delete_old_checkpoints":   True,
    "generate_every_n_epochs":  None,
    "generate_n_tokens":        None,
    "generate_temperature":     None,
    "generate_top_k":           None,
    "seed_text":                None,
    "auto_batch_scale":         None,
    "auto_batch_max":           None,
    "use_amp":                  None,
}


DO   = "\033[91m"
XANH = "\033[92m"
VANG = "\033[93m"
CYAN = "\033[96m"
DAM  = "\033[1m"
TAT  = "\033[0m"
SEP  = "=" * 100

CFG_FALLBACK = {
    "d_model": 780, "n_layers": 14, "n_heads": 12,
    "block_size": 512, "dropout": 0.1,
    "learning_rate": 3e-4, "min_lr": 1e-5,
    "beta1": 0.9, "beta2": 0.95,
    "weight_decay": 0.1, "grad_clip": 1.0,
    "use_cosine_lr": True, "warmup_steps": 0, "warmup_epochs": 0,
    "val_every_n_steps": 500, "val_steps": 16,
    "save_latest": True, "save_every_n_epochs": 1,
    "delete_old_checkpoints": True,
    "generate_every_n_epochs": 1,
    "generate_n_tokens": 150, "generate_temperature": 0.8,
    "generate_top_k": 30,
    "seed_text": "User: hôm nay bạn thế nào?\nFuid: ",
    "auto_batch_scale": False, "auto_batch_max": 128,
    "use_amp": True, "batch_size": 48,
}

def _tinh_perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def _tinh_lr_cosine(buoc_hien_tai, tong_buoc, lr_max, lr_min, warmup_buoc):
    if buoc_hien_tai < warmup_buoc:
        return lr_max * (buoc_hien_tai + 1) / max(warmup_buoc, 1)
    tien_trinh = (buoc_hien_tai - warmup_buoc) / max(tong_buoc - warmup_buoc, 1)
    he_so      = 0.5 * (1.0 + math.cos(math.pi * tien_trinh))
    return lr_min + he_so * (lr_max - lr_min)


def _dinh_dang_thoi_gian(giay: float) -> str:
    giay = max(0, int(giay))
    h, rem = divmod(giay, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _tu_dong_tang_batch(mo_hinh, block_size, batch_start, batch_max, device, dung_amp=False):
    if "cuda" not in str(device):
        print(f"  {VANG}[Auto-Batch] CPU — giữ batch_size={batch_start}{TAT}")
        return batch_start
    print(f"  {CYAN}[Auto-Batch] Thử tăng batch ({batch_start}→{batch_max})...{TAT}")
    batch, last_ok = batch_start, batch_start
    while batch <= batch_max:
        try:
            torch.cuda.empty_cache()
            x = torch.randint(0, 100, (batch, block_size), device=device)
            y = torch.randint(0, 100, (batch, block_size), device=device)
            mo_hinh.zero_grad(set_to_none=True)
            if dung_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = mo_hinh(x, y)
            else:
                _, loss = mo_hinh(x, y)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            mo_hinh.zero_grad(set_to_none=True)
            last_ok = batch
            batch   = int(batch * 1.5)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    print(f"  {XANH}[Auto-Batch] Dùng batch_size={last_ok}{TAT}")
    return last_ok

class TheoDõiETA:
    def __init__(self, cua_so=50):
        self.cua_so      = cua_so
        self._buoc_times = []

    def them_buoc(self, dt):
        self._buoc_times.append(dt)
        if len(self._buoc_times) > self.cua_so:
            self._buoc_times.pop(0)

    def trung_binh(self):
        return sum(self._buoc_times) / len(self._buoc_times) if self._buoc_times else 0.0

    def eta(self, buoc_con_lai):
        tb = self.trung_binh()
        return _dinh_dang_thoi_gian(tb * buoc_con_lai) if tb > 0 else "?"

class QuanLyCheckpoint:
    TEN_BEST   = "fuid_best.pt"
    TEN_LATEST = "fuid_latest.pt"
    PAT_CKPT   = re.compile(r"fuid_checkpoint_ep(\d+)_st(\d+)\.pt$")

    def __init__(self, thu_muc: Path):
        self.thu_muc = thu_muc
        self.thu_muc.mkdir(parents=True, exist_ok=True)

    def _doc_ep_st(self, ten_file):
        m = self.PAT_CKPT.match(ten_file)
        return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

    def quet_tat_ca(self):
        ket_qua = []
        for tep in self.thu_muc.glob("fuid_checkpoint_ep*_st*.pt"):
            ep, st = self._doc_ep_st(tep.name)
            if ep >= 0:
                ket_qua.append((ep, st, tep))
        ket_qua.sort(key=lambda x: (x[0], x[1]))
        return ket_qua

    def _tao_du_lieu(self, mo_hinh, bo_toi_uu, epoch, step,
                     train_loss, val_loss, tong_thoi_gian, cfg):
        _st = (mo_hinh.module.state_dict()
               if isinstance(mo_hinh, nn.DataParallel)
               else mo_hinh.state_dict())
        return {
            "epoch":                  epoch,
            "step":                   step,
            "cfg":                    cfg,
            "model_state":            _st,
            "optimizer_state":        bo_toi_uu.state_dict(),
            "train_loss":             train_loss,
            "val_loss":               val_loss,
            "perplexity":             _tinh_perplexity(val_loss),
            "total_time_accumulated": tong_thoi_gian,
        }

    def luu_checkpoint(self, mo_hinh, bo_toi_uu, epoch, step,
                       train_loss, val_loss, tong_thoi_gian, la_best, cfg, xoa_cu=True):
        ten_moi       = f"fuid_checkpoint_ep{epoch}_st{step}.pt"
        duong_dan_moi = self.thu_muc / ten_moi
        du_lieu       = self._tao_du_lieu(mo_hinh, bo_toi_uu, epoch, step,
                                          train_loss, val_loss, tong_thoi_gian, cfg)
        torch.save(du_lieu, duong_dan_moi)
        print(f"  {CYAN}[Lưu] {ten_moi} (val={val_loss:.4f}){TAT}")
        if la_best:
            torch.save(du_lieu, self.thu_muc / self.TEN_BEST)
            print(f"  {XANH}{DAM}[TỐT NHẤT] cập nhật fuid_best.pt{TAT}")
        torch.save(du_lieu, self.thu_muc / self.TEN_LATEST)
        if xoa_cu:
            for _, _, tep_cu in self.quet_tat_ca():
                if tep_cu.name != ten_moi:
                    try:
                        tep_cu.unlink()
                    except Exception:
                        pass
        return duong_dan_moi

    def luu_latest_nhe(self, mo_hinh, bo_toi_uu, epoch, step,
                       train_loss, val_loss, tong_thoi_gian, cfg):
        du_lieu = self._tao_du_lieu(mo_hinh, bo_toi_uu, epoch, step,
                                    train_loss, val_loss, tong_thoi_gian, cfg)
        torch.save(du_lieu, self.thu_muc / self.TEN_LATEST)

class BoDuLieuNhanh:
    def __init__(self, bin_path: Path, block_size, batch_size, device):
        if not bin_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {bin_path}")
        self.data         = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.T            = block_size
        self.B            = batch_size
        self.device       = device
        self.total_tokens = len(self.data)
        if self.total_tokens <= self.T:
            raise ValueError(f"Data quá ngắn ({self.total_tokens}) vs block_size={self.T}")

    def lay_lo_ngau_nhien(self):
        ix = torch.randint(0, self.total_tokens - self.T, (self.B,))
        x  = torch.stack([torch.from_numpy(self.data[i:i+self.T].astype(np.int64)) for i in ix])
        y  = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.T].astype(np.int64)) for i in ix])
        if "cuda" in str(self.device):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def so_buoc_moi_epoch(self):
        return max(1, self.total_tokens // (self.T * self.B))

def _danh_gia_val(mo_hinh, bo_val, so_buoc_val):
    tong_loss = 0.0
    dung_amp  = next(mo_hinh.parameters()).device.type == "cuda"
    with torch.inference_mode():
        for _ in range(so_buoc_val):
            x, y = bo_val.lay_lo_ngau_nhien()
            if dung_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = mo_hinh(x, y)
            else:
                _, loss = mo_hinh(x, y)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            tong_loss += loss.item()
    return tong_loss / max(so_buoc_val, 1)


def _sinh_van_ban(mo_hinh, tokenizer, cfg, device):
    van_ban = cfg.get("seed_text", "")
    ids     = tokenizer.ma_hoa(van_ban)
    if not ids:
        return ""
    x     = torch.tensor([ids], dtype=torch.long, device=device)
    n_gen = cfg.get("generate_n_tokens", 100)
    temp  = cfg.get("generate_temperature", 0.8)
    top_k = cfg.get("generate_top_k", 30)
    with torch.inference_mode():
        for _ in range(n_gen):
            x_crop    = x[:, -cfg["block_size"]:]
            logits, _ = mo_hinh(x_crop)
            logits    = logits[:, -1, :] / max(temp, 1e-9)
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            tok   = torch.multinomial(probs, num_samples=1)
            x     = torch.cat([x, tok], dim=1)
    return tokenizer.giai_ma(x[0].tolist())

def _in_bang_lich_su(lich_su):
    if not lich_su:
        return
    print(f"\n{DAM}  {'─'*78}")
    print(f"  {'VÒNG':^5} {'BƯỚC':^7} {'MM_HUẤN':^9} {'MM_XT':^9} "
          f"{'ĐPT':^8} {'ĐỘ LỆCH':^10} {'LƯU?':^6} {'LR':^10} {'TGIAN':^8}")
    print(f"  {'─'*78}{TAT}")
    for i, r in enumerate(lich_su):
        delta = ""
        if i > 0:
            d   = r["val_loss"] - lich_su[i - 1]["val_loss"]
            mau = XANH if d < 0 else DO
            delta = f"{mau}{d:+.4f}{TAT}"
        luu = f"{XANH}CÓ{TAT}" if r.get("la_best") else f"{VANG}KHÔNG{TAT}"
        print(
            f"  {r['epoch']:^5} {r['step']:^7} {r['train_loss']:^9.4f} "
            f"{r['val_loss']:^9.4f} {r['perplexity']:^8.4f} "
            f"{delta:^26} {luu:^14} {r['lr']:^10.2e} {r['thoi_gian']:^6.1f}s"
        )
    print(f"{DAM}  {'─'*78}{TAT}\n")

def _tim_ckpt(ckpt_arg: str, output_dir: Path) -> Path:
    if ckpt_arg == "best":
        p = output_dir / QuanLyCheckpoint.TEN_BEST
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy fuid_best.pt tại {output_dir}")
        return p
    if ckpt_arg == "latest":
        p = output_dir / QuanLyCheckpoint.TEN_LATEST
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy fuid_latest.pt tại {output_dir}")
        return p
    p = Path(ckpt_arg)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {p}")
    return p

def huan_luyen_tiep(rcfg: dict):
    parser = argparse.ArgumentParser(description="Resume training Fuid từ checkpoint")
    parser.add_argument("--ckpt",   type=str,   default=None)
    parser.add_argument("--epochs", type=int,   default=None, help="Số vòng train thêm")
    parser.add_argument("--data",   type=str,   default=None, help="Thư mục data mới")
    parser.add_argument("--lr",     type=float, default=None, help="Override learning rate")
    args = parser.parse_args()
    if args.ckpt:   rcfg["ckpt_path"]   = args.ckpt
    if args.epochs: rcfg["them_epochs"] = args.epochs
    if args.data:   rcfg["data_dir"]    = args.data
    if args.lr:     rcfg["learning_rate"] = args.lr
    so_gpu   = torch.cuda.device_count()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(rcfg["output_dir"]) if rcfg.get("output_dir") else PROJECT_ROOT / "dau_ra_fuidai"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _tim_ckpt(rcfg["ckpt_path"], output_dir)
    print(f"\n{CYAN}{DAM}{SEP}{TAT}")
    print(f"{CYAN}{DAM}  FUID AI - TIẾP TỤC HUẤN LUYỆN{TAT}")
    print(f"{CYAN}{DAM}{SEP}{TAT}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Thiết bị    : {device}" + (f" ({so_gpu} GPU)" if so_gpu > 1 else ""))

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = {**CFG_FALLBACK, **dict(ck.get("cfg", {}))}

    if not ck.get("cfg"):
        print(f"  {VANG} Checkpoint không có cfg, dùng giá trị mặc định.{TAT}")

    # Áp override từ RESUME_CONFIG (bỏ qua các key đặc biệt và None)
    skip_keys = {"ckpt_path", "data_dir", "output_dir", "them_epochs"}
    for k, v in rcfg.items():
        if k not in skip_keys and v is not None:
            cfg[k] = v

    start_epoch    = ck.get("epoch", 0) + 1
    tong_thoi_gian = ck.get("total_time_accumulated", 0.0)
    best_val_loss  = ck.get("val_loss", float("inf"))
    them_epochs    = rcfg.get("them_epochs", 500)
    end_epoch      = start_epoch + them_epochs
    cfg["epochs"]  = end_epoch

    dung_amp = cfg.get("use_amp", True) and "cuda" in str(device)
    scaler   = torch.amp.GradScaler("cuda") if dung_amp else None

    print(f"  AMP (FP16)  : {'BẬT' if dung_amp else 'TẮT'}")
    print(f"  Tiếp từ vòng: {start_epoch}")
    print(f"  Train thêm  : {them_epochs} vòng (đến vòng {end_epoch - 1})")
    print(f"  Val loss cũ : {best_val_loss:.4f} (ppl={_tinh_perplexity(best_val_loss):.4f})\n")

    if rcfg.get("data_dir"):
        data_dir_path = Path(rcfg["data_dir"])
    else:
        data_dir_path = None
        for candidate in [Path("/content/data_train_local"), PROJECT_ROOT / "data_train"]:
            if candidate.exists():
                data_dir_path = candidate
                break
    if data_dir_path is None or not data_dir_path.exists():
        raise FileNotFoundError("Không tìm thấy thư mục data. Hãy chỉ định --data <đường_dẫn>")

    print(f"  Thư mục data: {data_dir_path}")
    print(f"  Đầu ra      : {output_dir}\n")

    vocab_path = data_dir_path / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Không tìm thấy vocab.json tại: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    tokenizer            = TokenizerTV()
    tokenizer.char2idx   = vocab_raw
    tokenizer.idx2char   = {int(v): k for k, v in vocab_raw.items()}
    tokenizer.vocab_size = len(vocab_raw)
    print(f"  Từ vựng: {tokenizer.vocab_size} ký tự")

    ck_vocab = cfg.get("vocab_size", tokenizer.vocab_size)
    if tokenizer.vocab_size != ck_vocab:
        print(f"  {VANG}[CẢNH BÁO] vocab_size mới={tokenizer.vocab_size} "
              f"khác checkpoint={ck_vocab}. Dùng vocab của data mới.{TAT}")

    mo_hinh_goc = fuidai(
        kich_thuoc_tu_vung = tokenizer.vocab_size,
        d_model            = cfg["d_model"],
        so_lop             = cfg["n_layers"],
        so_dau             = cfg["n_heads"],
        kich_thuoc_khoi    = cfg["block_size"],
        ty_le_bo_qua       = cfg.get("dropout", 0.1),
    ).to(device)

    missing, unexpected = mo_hinh_goc.load_state_dict(ck["model_state"], strict=False)
    if missing:
        print(f"  {VANG}[Thiếu keys]: {missing[:5]}{'...' if len(missing) > 5 else ''}{TAT}")
    if unexpected:
        print(f"  {VANG}[Keys lạ]  : {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}{TAT}")
    print(f"  {XANH}Đã nạp trọng số model.{TAT}")

    if so_gpu > 1:
        mo_hinh = nn.DataParallel(mo_hinh_goc)
        print(f"  {XANH}[Multi-GPU] DataParallel trên {so_gpu} GPU{TAT}")
    else:
        mo_hinh = mo_hinh_goc

    so_tham_so = sum(p.numel() for p in mo_hinh_goc.parameters() if p.requires_grad)
    print(f"  Tham số mô hình: {so_tham_so:,}")

    batch_size = cfg.get("batch_size", 48)
    if cfg.get("auto_batch_scale", False):
        batch_size = _tu_dong_tang_batch(
            mo_hinh_goc, cfg["block_size"],
            batch_start=batch_size, batch_max=cfg.get("auto_batch_max", 128),
            device=device, dung_amp=dung_amp,
        )
        if so_gpu > 1:
            batch_size = max(so_gpu, int(batch_size * 0.6 // so_gpu) * so_gpu)
    else:
        if so_gpu > 1 and batch_size % so_gpu != 0:
            batch_size = (batch_size // so_gpu) * so_gpu
            print(f"  {CYAN}[Multi-GPU] Điều chỉnh batch_size={batch_size}{TAT}")

    bo_train = BoDuLieuNhanh(data_dir_path / "train.bin", cfg["block_size"], batch_size, device)
    bo_val   = BoDuLieuNhanh(data_dir_path / "val.bin",   cfg["block_size"], batch_size, device)

    so_buoc_epoch = bo_train.so_buoc_moi_epoch()
    tong_buoc     = so_buoc_epoch * them_epochs
    warmup_buoc   = cfg.get("warmup_steps") or so_buoc_epoch * cfg.get("warmup_epochs", 0)

    print(f"  batch_size thực tế : {batch_size}")
    print(f"  Token huấn luyện   : {bo_train.total_tokens:,}")
    print(f"  Token xác thực     : {bo_val.total_tokens:,}")
    print(f"  Bước/vòng          : {so_buoc_epoch:,}")
    print(f"  Tổng bước thêm     : {tong_buoc:,}")
    print(f"  Warmup bước        : {warmup_buoc:,}")

    bo_toi_uu = torch.optim.AdamW(
        mo_hinh.parameters(),
        lr           = cfg["learning_rate"],
        betas        = (cfg.get("beta1", 0.9), cfg.get("beta2", 0.95)),
        weight_decay = cfg.get("weight_decay", 0.1),
    )

    if "optimizer_state" in ck:
        try:
            bo_toi_uu.load_state_dict(ck["optimizer_state"])
            for nhom in bo_toi_uu.param_groups:
                nhom["lr"] = cfg["learning_rate"]
            print(f"  {XANH}Đã nạp optimizer state.{TAT}")
        except Exception as e:
            print(f"  {VANG}Không thể nạp optimizer state ({e}), khởi tạo mới.{TAT}")

    quan_ly     = QuanLyCheckpoint(output_dir)
    lich_su     = []
    t_phien     = time.time()
    eta_tracker = TheoDõiETA(cua_so=50)
    buoc_toan_cuc = 0

    val_every     = cfg.get("val_every_n_steps", 500)
    val_steps     = cfg.get("val_steps", 16)
    save_ep_every = cfg.get("save_every_n_epochs", 1)

    print(f"\n{SEP}")
    print(f"{DAM}Bắt đầu train tiếp từ vòng {start_epoch} → {end_epoch - 1}{TAT}")
    print(f"{SEP}\n")

    for epoch in range(start_epoch, end_epoch):
        mo_hinh.train()
        t0_epoch      = time.time()
        tong_mm_ep    = 0.0
        dem_buoc_ep   = 0
        last_val_loss = best_val_loss

        for step in range(so_buoc_epoch):

            lr_hien_tai = (
                _tinh_lr_cosine(buoc_toan_cuc, tong_buoc,
                                cfg["learning_rate"], cfg.get("min_lr", 1e-5), warmup_buoc)
                if cfg.get("use_cosine_lr", True)
                else cfg["learning_rate"]
            )
            for nhom in bo_toi_uu.param_groups:
                nhom["lr"] = lr_hien_tai

            t0_step = time.time()
            x, y    = bo_train.lay_lo_ngau_nhien()

            bo_toi_uu.zero_grad(set_to_none=True)
            if dung_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = mo_hinh(x, y)
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(bo_toi_uu)
                torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cfg.get("grad_clip", 1.0))
                scaler.step(bo_toi_uu)
                scaler.update()
            else:
                _, loss = mo_hinh(x, y)
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cfg.get("grad_clip", 1.0))
                bo_toi_uu.step()

            loss_val      = loss.item()
            tong_mm_ep   += loss_val
            dem_buoc_ep  += 1
            buoc_toan_cuc += 1
            dt_step       = time.time() - t0_step
            eta_tracker.them_buoc(dt_step)

            if step % 10 == 0:
                print(
                    f"  [TRAIN] ep={epoch:4d} | st={step:6d}/{so_buoc_epoch} "
                    f"| mm={loss_val:.4f} | lr={lr_hien_tai:.2e} "
                    f"| {dt_step*1000:.1f}ms/st "
                    f"| ETA={eta_tracker.eta(tong_buoc - buoc_toan_cuc)}"
                )

            if (step + 1) % val_every == 0 or step == so_buoc_epoch - 1:
                mo_hinh.eval()
                val_loss = _danh_gia_val(mo_hinh, bo_val, val_steps)
                mo_hinh.train()

                ppl       = _tinh_perplexity(val_loss)
                train_avg = tong_mm_ep / max(dem_buoc_ep, 1)
                la_best   = val_loss < best_val_loss
                last_val_loss = val_loss

                print(
                    f"\n  {DAM}[EVAL]{TAT} ep={epoch} st={step} "
                    f"| train={train_avg:.4f} | val={val_loss:.4f} | ppl={ppl:.4f} "
                    f"{'← TỐT NHẤT' if la_best else ''}"
                )

                if la_best:
                    best_val_loss = val_loss
                    t_hien = tong_thoi_gian + (time.time() - t_phien)
                    _st_b  = (mo_hinh.module.state_dict()
                              if isinstance(mo_hinh, nn.DataParallel)
                              else mo_hinh.state_dict())
                    torch.save({
                        "epoch": epoch, "step": step, "cfg": cfg,
                        "model_state":     _st_b,
                        "optimizer_state": bo_toi_uu.state_dict(),
                        "train_loss":      train_avg, "val_loss": val_loss,
                        "perplexity":      ppl,
                        "total_time_accumulated": t_hien,
                    }, output_dir / QuanLyCheckpoint.TEN_BEST)
                    print(f"  {XANH}{DAM}[TỐT NHẤT] cập nhật fuid_best.pt{TAT}")

                lich_su.append({
                    "epoch": epoch, "step": step,
                    "train_loss": train_avg, "val_loss": val_loss,
                    "perplexity": ppl, "lr": lr_hien_tai,
                    "la_best": la_best, "thoi_gian": time.time() - t0_epoch,
                })

        dt_epoch     = time.time() - t0_epoch
        t_tich_luy   = tong_thoi_gian + (time.time() - t_phien)
        train_avg_ep = tong_mm_ep / max(dem_buoc_ep, 1)

        buoc_tinh_save = epoch + 1 - start_epoch
        if buoc_tinh_save % save_ep_every == 0:
            la_best_ep = last_val_loss < best_val_loss
            quan_ly.luu_checkpoint(
                mo_hinh, bo_toi_uu,
                epoch=epoch, step=so_buoc_epoch - 1,
                train_loss=train_avg_ep, val_loss=last_val_loss,
                tong_thoi_gian=t_tich_luy, la_best=la_best_ep,
                cfg=cfg, xoa_cu=cfg.get("delete_old_checkpoints", True),
            )
            if la_best_ep:
                best_val_loss = last_val_loss
        else:
            quan_ly.luu_latest_nhe(
                mo_hinh, bo_toi_uu,
                epoch=epoch, step=so_buoc_epoch - 1,
                train_loss=train_avg_ep, val_loss=last_val_loss,
                tong_thoi_gian=t_tich_luy, cfg=cfg,
            )

        print(f"\n{CYAN}{DAM}{SEP}{TAT}")
        print(
            f"{CYAN}{DAM}  XONG VÒNG {epoch} | "
            f"train={train_avg_ep:.4f} | "
            f"tốt nhất={best_val_loss:.4f} (ppl={_tinh_perplexity(best_val_loss):.4f}) | "
            f"{dt_epoch:.1f}s | tích lũy={t_tich_luy/3600:.2f}h | "
            f"ETA={eta_tracker.eta(tong_buoc - buoc_toan_cuc)}{TAT}"
        )

        gen_every = cfg.get("generate_every_n_epochs", 1)
        if buoc_tinh_save % gen_every == 0:
            mo_hinh.eval()
            _mo_gen = mo_hinh.module if isinstance(mo_hinh, nn.DataParallel) else mo_hinh
            van_ban = _sinh_van_ban(_mo_gen, tokenizer, cfg, device)
            mo_hinh.train()
            if van_ban:
                print(f"\n  {DAM}[SINH VĂN BẢN]{TAT}\n  {'-'*60}")
                print(f"  {van_ban[:300]}\n  {'-'*60}")

        _in_bang_lich_su(lich_su[-10:])

    print(f"\n{XANH}{DAM}{SEP}{TAT}")
    print(f"{XANH}{DAM}  HOÀN TẤT TRAIN TIẾP{TAT}")
    if lich_su:
        br = min(lich_su, key=lambda r: r["val_loss"])
        print(f"{XANH}{DAM}  Mất mát tốt nhất: {br['val_loss']:.4f} "
              f"(vòng {br['epoch']} bước {br['step']}){TAT}")
        print(f"{XANH}{DAM}  Độ phức tạp tốt nhất: {br['perplexity']:.4f}{TAT}")
    print(f"{XANH}{DAM}  Model tốt nhất: {output_dir / QuanLyCheckpoint.TEN_BEST}{TAT}")
    print(f"{XANH}{DAM}{SEP}{TAT}\n")


if __name__ == "__main__":
    huan_luyen_tiep(RESUME_CONFIG)
