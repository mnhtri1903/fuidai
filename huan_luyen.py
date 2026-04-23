import json
import math
import re
import sys
import time
import os
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

DATA_DIR = Path("/content/data_train_local")
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "data_train"

OUTPUT_DIR = PROJECT_ROOT / "dau_ra_fuidai"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG = {
    "d_model":                  780,
    "n_layers":                 14,
    "n_heads":                  12,
    "block_size":               512,
    "dropout":                  0.1,
    "epochs":                   3,         
    "batch_size":               48,        
    "learning_rate":            3e-4,
    "min_lr":                   1e-5,
    "beta1":                    0.9,
    "beta2":                    0.95,
    "weight_decay":             0.1,
    "grad_clip":                1.0,
    "use_cosine_lr":            True,
    "warmup_epochs":            0,         
    "warmup_steps":             2000,      
    "val_every_n_steps":        500,       
    "val_steps":                16,
    "save_latest":              True,
    "save_every_n_epochs":      1,
    "delete_old_checkpoints":   True,
    "generate_every_n_epochs":  1,
    "generate_n_tokens":        150,
    "generate_temperature":     0.8,
    "generate_top_k":           30,
    "seed_text":                "User: hôm nay bạn thế nào?\nFuid: ",
    "auto_batch_scale":         False,     
    "auto_batch_max":           128,
    "use_amp":                  True,
}

DO   = "\033[91m"
XANH = "\033[92m"
VANG = "\033[93m"
CYAN = "\033[96m"
DAM  = "\033[1m"
TAT  = "\033[0m"
SEP  = "=" * 100


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


def _tu_dong_tang_batch(mo_hinh, block_size: int, batch_start: int,
                        batch_max: int, device, dung_amp: bool = False) -> int:
    if "cuda" not in str(device):
        print(f"  {VANG}[Auto-Batch] CPU mode — giữ batch_size={batch_start}{TAT}")
        return batch_start

    print(f"  {CYAN}[Auto-Batch] Đang thử tăng batch_size "
          f"({batch_start} → tối đa {batch_max}) bằng forward+backward thật...{TAT}")

    batch   = batch_start
    last_ok = batch_start

    while batch <= batch_max:
        try:
            torch.cuda.empty_cache()
            x_test = torch.randint(0, 100, (batch, block_size), device=device)
            y_test = torch.randint(0, 100, (batch, block_size), device=device)
            mo_hinh.zero_grad(set_to_none=True)
            if dung_amp:
                with torch.amp.autocast("cuda"):
                    _, loss = mo_hinh(x_test, y_test)
            else:
                _, loss = mo_hinh(x_test, y_test)
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


def _dinh_dang_thoi_gian(giay: float) -> str:
    giay = max(0, int(giay))
    h, rem = divmod(giay, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ─── ETA Tracker ──────────────────────────────────────────────────────────────

class TheoDõiETA:
    def __init__(self, cua_so: int = 50):
        self.cua_so       = cua_so
        self._buoc_times: list[float] = []

    def them_buoc(self, dt: float):
        self._buoc_times.append(dt)
        if len(self._buoc_times) > self.cua_so:
            self._buoc_times.pop(0)

    def trung_binh(self) -> float:
        if not self._buoc_times:
            return 0.0
        return sum(self._buoc_times) / len(self._buoc_times)

    def eta(self, buoc_con_lai: int) -> str:
        tb = self.trung_binh()
        if tb <= 0:
            return "?"
        return _dinh_dang_thoi_gian(tb * buoc_con_lai)


# ─── Checkpoint Manager ───────────────────────────────────────────────────────

class QuanLyCheckpoint:
    TEN_BEST   = "fuid_best.pt"
    TEN_LATEST = "fuid_latest.pt"
    PAT_CKPT   = re.compile(r"fuid_checkpoint_ep(\d+)_st(\d+)\.pt$")

    def __init__(self, thu_muc: Path):
        self.thu_muc = thu_muc
        self.thu_muc.mkdir(parents=True, exist_ok=True)

    def _doc_ep_st(self, ten_file: str):
        m = self.PAT_CKPT.match(ten_file)
        if m:
            return int(m.group(1)), int(m.group(2))
        return -1, -1

    def quet_tat_ca(self):
        ket_qua = []
        for tep in self.thu_muc.glob("fuid_checkpoint_ep*_st*.pt"):
            ep, st = self._doc_ep_st(tep.name)
            if ep >= 0:
                ket_qua.append((ep, st, tep))
        ket_qua.sort(key=lambda x: (x[0], x[1]))
        return ket_qua

    def tim_moi_nhat(self):
        ds = self.quet_tat_ca()
        if not ds:
            return None
        ep, st, tep = ds[-1]
        return tep, ep, st

    def in_danh_sach(self):
        ds = self.quet_tat_ca()
        if not ds:
            print(f"  {VANG}Không tìm thấy tệp điểm kiểm tra {self.thu_muc}{TAT}")
            return
        print(f"\n  {DAM}DANH SÁCH ĐIỂM KIỂM TRA ({len(ds)} tệp):{TAT}")
        print(f"  {'STT':^5} {'TÊN TỆP':^45} {'VÒNG':^7} {'BƯỚC':^7} {'KÍCH THƯỚC':^12}")
        print(f"  {'-'*76}")
        for i, (ep, st, tep) in enumerate(ds):
            kich_thuoc = tep.stat().st_size / 1024 / 1024
            dau = f"{XANH}*{TAT}" if i == len(ds) - 1 else " "
            print(f"  {dau}{i+1:^5} {tep.name:^45} {ep:^7} {st:^7} {kich_thuoc:^10.1f}MB")
        print(f"{SEP}")

    def co_best(self):
        return (self.thu_muc / self.TEN_BEST).exists()

    def doc_best_loss(self):
        tep = self.thu_muc / self.TEN_BEST
        if not tep.exists():
            return float("inf")
        try:
            ck = torch.load(tep, map_location="cpu", weights_only=False)
            return ck.get("val_loss", float("inf"))
        except Exception:
            return float("inf")

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
                       train_loss, val_loss, tong_thoi_gian,
                       la_best, cfg, xoa_cu=True):
        ten_moi       = f"fuid_checkpoint_ep{epoch}_st{step}.pt"
        duong_dan_moi = self.thu_muc / ten_moi
        du_lieu       = self._tao_du_lieu(mo_hinh, bo_toi_uu, epoch, step,
                                          train_loss, val_loss, tong_thoi_gian, cfg)
        torch.save(du_lieu, duong_dan_moi)
        print(f"  {CYAN}[Lưu] {ten_moi}  "
              f"(mm={val_loss:.4f} ppl={_tinh_perplexity(val_loss):.4f}){TAT}")

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

    def tai_checkpoint(self, mo_hinh, bo_toi_uu, device, lua_chon="moi_nhat"):
        if lua_chon == "best" and self.co_best():
            tep = self.thu_muc / self.TEN_BEST
            print(f"  Đang tải: {tep.name}")
        elif lua_chon == "latest" and (self.thu_muc / self.TEN_LATEST).exists():
            tep = self.thu_muc / self.TEN_LATEST
            print(f"  Đang tải: {tep.name}")
        else:
            ket_qua = self.tim_moi_nhat()
            if ket_qua is None:
                return None
            tep, ep, st = ket_qua
            print(f"  Đang tải điểm kiểm tra mới nhất: {tep.name}")

        ck = torch.load(tep, map_location=device, weights_only=False)
        mo_hinh.load_state_dict(ck["model_state"])
        bo_toi_uu.load_state_dict(ck["optimizer_state"])
        print(f"  {XANH}Đã tải: vòng={ck['epoch']} bước={ck['step']} "
              f"mm_xt={ck.get('val_loss', -1):.4f} "
              f"ppl={ck.get('perplexity', -1):.4f}{TAT}")
        return ck


# ─── Data Loader ──────────────────────────────────────────────────────────────

class BoDuLieuNhanh:
    def __init__(self, bin_path: Path, block_size: int, batch_size: int, device):
        if not bin_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {bin_path}")
        self.data         = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.T            = block_size
        self.B            = batch_size
        self.device       = device
        self.total_tokens = len(self.data)
        if self.total_tokens <= self.T:
            raise ValueError(
                f"Dữ liệu quá ngắn ({self.total_tokens} token) so với block_size={self.T}"
            )

    def lay_lo_ngau_nhien(self):
        ix = torch.randint(0, self.total_tokens - self.T, (self.B,))
        x  = torch.stack([
            torch.from_numpy(self.data[i: i + self.T].astype(np.int64)) for i in ix
        ])
        y  = torch.stack([
            torch.from_numpy(self.data[i + 1: i + 1 + self.T].astype(np.int64)) for i in ix
        ])
        if "cuda" in str(self.device):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def so_buoc_moi_epoch(self):
        return max(1, self.total_tokens // (self.T * self.B))


# ─── Val & Generate ───────────────────────────────────────────────────────────

def _danh_gia_val(mo_hinh, bo_val, so_buoc_val: int) -> float:
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


def _sinh_van_ban(mo_hinh, tokenizer, cfg: dict, device) -> str:
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


# ─── History Printer ──────────────────────────────────────────────────────────

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


def _xoa_tat_ca_checkpoint(quan_ly: QuanLyCheckpoint):
    for _, _, tep in quan_ly.quet_tat_ca():
        try:
            tep.unlink()
        except Exception:
            pass
    for ten in [quan_ly.TEN_BEST, quan_ly.TEN_LATEST]:
        p = quan_ly.thu_muc / ten
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def huan_luyen(cfg: dict):
    so_gpu   = torch.cuda.device_count()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dung_amp = cfg.get("use_amp", True) and "cuda" in str(device)
    scaler   = torch.amp.GradScaler("cuda") if dung_amp else None

    print(f"\n{CYAN}{DAM}{SEP}{TAT}")
    print(f"{CYAN}{DAM}  FUID AI - HUẤN LUYỆN LLM{TAT}")
    print(f"{CYAN}{DAM}{SEP}{TAT}")
    if so_gpu > 1:
        print(f"  Thiết bị  : {device}  ({so_gpu} GPU khả dụng)")
    else:
        print(f"  Thiết bị  : {device}")
    print(f"  AMP (FP16): {'BẬT' if dung_amp else 'TẮT'}")
    print(f"  Thư mục dl: {DATA_DIR}")
    print(f"  Đầu ra    : {OUTPUT_DIR}\n")

    vocab_path = DATA_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Không tìm thấy vocab.json tại: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)

    tokenizer            = TokenizerTV()
    tokenizer.char2idx   = vocab_raw
    tokenizer.idx2char   = {int(v): k for k, v in vocab_raw.items()}
    tokenizer.vocab_size = len(vocab_raw)
    print(f"  Từ vựng: {tokenizer.vocab_size} ký tự")

    mo_hinh_goc = fuidai(
        kich_thuoc_tu_vung = tokenizer.vocab_size,
        d_model            = cfg["d_model"],
        so_lop             = cfg["n_layers"],
        so_dau             = cfg["n_heads"],
        kich_thuoc_khoi    = cfg["block_size"],
        ty_le_bo_qua       = cfg["dropout"],
    ).to(device)

    if so_gpu > 1:
        mo_hinh = nn.DataParallel(mo_hinh_goc)
        print(f"  {XANH}[Multi-GPU] DataParallel trên {so_gpu} GPU{TAT}")
    else:
        mo_hinh = mo_hinh_goc

    so_tham_so = sum(p.numel() for p in mo_hinh_goc.parameters() if p.requires_grad)
    print(f"  Tham số mô hình: {so_tham_so:,}")

    batch_size = cfg["batch_size"]
    if cfg.get("auto_batch_scale", False):
        batch_size = _tu_dong_tang_batch(
            mo_hinh_goc, cfg["block_size"],
            batch_start = batch_size,
            batch_max   = cfg.get("auto_batch_max", 128),
            device      = device,
            dung_amp    = dung_amp,
        )
        if so_gpu > 1:
            batch_size = max(so_gpu, int(batch_size * 0.6 // so_gpu) * so_gpu)
            print(f"  {CYAN}[Multi-GPU] batch_size an toàn={batch_size} "
                  f"({batch_size // so_gpu} mẫu/GPU){TAT}")
    else:
        # Đảm bảo batch_size chia hết cho số GPU
        if so_gpu > 1 and batch_size % so_gpu != 0:
            batch_size = (batch_size // so_gpu) * so_gpu
            print(f"  {CYAN}[Multi-GPU] Điều chỉnh batch_size={batch_size} "
                  f"(chia hết {so_gpu} GPU){TAT}")

    train_bin = DATA_DIR / "train.bin"
    val_bin   = DATA_DIR / "val.bin"

    bo_train = BoDuLieuNhanh(train_bin, cfg["block_size"], batch_size, device)
    bo_val   = BoDuLieuNhanh(val_bin,   cfg["block_size"], batch_size, device)

    so_buoc_epoch = bo_train.so_buoc_moi_epoch()
    tong_buoc     = so_buoc_epoch * cfg["epochs"]

    # FIX: warmup_steps ưu tiên hơn warmup_epochs
    warmup_buoc = (
        cfg.get("warmup_steps")
        or so_buoc_epoch * cfg.get("warmup_epochs", 0)
    )

    print(f"  batch_size thực tế    : {batch_size}")
    print(f"  Token huấn luyện      : {bo_train.total_tokens:,}")
    print(f"  Token xác thực        : {bo_val.total_tokens:,}")
    print(f"  Bước/vòng             : {so_buoc_epoch:,}")
    print(f"  Tổng bước             : {tong_buoc:,}")
    print(f"  Warmup bước           : {warmup_buoc:,}")
    print(f"  Val mỗi               : {cfg.get('val_every_n_steps')} bước "
          f"({cfg.get('val_steps')} bước/lần)")

    bo_toi_uu = torch.optim.AdamW(
        mo_hinh.parameters(),
        lr           = cfg["learning_rate"],
        betas        = (cfg["beta1"], cfg["beta2"]),
        weight_decay = cfg["weight_decay"],
    )

    quan_ly = QuanLyCheckpoint(OUTPUT_DIR)

    start_epoch    = 0
    start_step     = 0
    tong_thoi_gian = 0.0
    best_val_loss  = float("inf")
    buoc_toan_cuc  = 0

    _xoa_tat_ca_checkpoint(quan_ly)
    print(f"  {XANH}Bắt đầu huấn luyện từ đầu.{TAT}")

    lich_su: list[dict] = []
    t_phien     = time.time()
    eta_tracker = TheoDõiETA(cua_so=50)

    val_every     = cfg.get("val_every_n_steps", 500)
    val_steps     = cfg.get("val_steps", 16)
    save_ep_every = cfg.get("save_every_n_epochs", 1)

    print(f"\n{DAM}Bắt đầu huấn luyện:{TAT}")
    print(f"  Từ vòng={start_epoch} bước={start_step}")
    print(f"  Mất mát tốt nhất hiện tại: {best_val_loss:.4f} "
          f"(ppl={_tinh_perplexity(best_val_loss):.4f})")
    print(f"\n{SEP}")

    for epoch in range(start_epoch, cfg["epochs"]):
        mo_hinh.train()
        t0_epoch      = time.time()
        buoc_bd       = start_step if epoch == start_epoch else 0
        tong_mm_ep    = 0.0
        dem_buoc_ep   = 0
        last_val_loss = best_val_loss

        for step in range(buoc_bd, so_buoc_epoch):

            lr_hien_tai = (
                _tinh_lr_cosine(buoc_toan_cuc, tong_buoc,
                                cfg["learning_rate"], cfg["min_lr"], warmup_buoc)
                if cfg.get("use_cosine_lr")
                else cfg["learning_rate"]
            )
            for nhom in bo_toi_uu.param_groups:
                nhom["lr"] = lr_hien_tai

            t0_step = time.time()
            x, y    = bo_train.lay_lo_ngau_nhien()

            bo_toi_uu.zero_grad(set_to_none=True)
            if dung_amp:
                with torch.amp.autocast("cuda"):
                    logits, loss = mo_hinh(x, y)
                    if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                        loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(bo_toi_uu)
                torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cfg["grad_clip"])
                scaler.step(bo_toi_uu)
                scaler.update()
            else:
                logits, loss = mo_hinh(x, y)
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cfg["grad_clip"])
                bo_toi_uu.step()

            loss_val      = loss.item()
            tong_mm_ep   += loss_val
            dem_buoc_ep  += 1
            buoc_toan_cuc += 1
            dt_step       = time.time() - t0_step
            eta_tracker.them_buoc(dt_step)

            if step % 10 == 0:
                buoc_con_lai = tong_buoc - buoc_toan_cuc
                print(
                    f"  [TRAIN] ep={epoch:4d} | st={step:6d}/{so_buoc_epoch} "
                    f"| mm={loss_val:.4f} "
                    f"| lr={lr_hien_tai:.2e} "
                    f"| {dt_step*1000:.1f}ms/st "
                    f"| ETA={eta_tracker.eta(buoc_con_lai)}"
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
                    f"| train={train_avg:.4f} "
                    f"| val={val_loss:.4f} "
                    f"| ppl={ppl:.4f} "
                    f"{'← TỐT NHẤT' if la_best else ''}"
                )

                if la_best:
                    best_val_loss = val_loss
                    t_hien   = tong_thoi_gian + (time.time() - t_phien)
                    _st_best = (mo_hinh.module.state_dict()
                                if isinstance(mo_hinh, nn.DataParallel)
                                else mo_hinh.state_dict())
                    torch.save({
                        "epoch":                  epoch,
                        "step":                   step,
                        "cfg":                    cfg,
                        "model_state":            _st_best,
                        "optimizer_state":        bo_toi_uu.state_dict(),
                        "train_loss":             train_avg,
                        "val_loss":               val_loss,
                        "perplexity":             ppl,
                        "total_time_accumulated": t_hien,
                    }, OUTPUT_DIR / QuanLyCheckpoint.TEN_BEST)
                    print(f"  {XANH}{DAM}[TỐT NHẤT] cập nhật fuid_best.pt{TAT}")

                lich_su.append({
                    "epoch":      epoch,
                    "step":       step,
                    "train_loss": train_avg,
                    "val_loss":   val_loss,
                    "perplexity": ppl,
                    "lr":         lr_hien_tai,
                    "la_best":    la_best,
                    "thoi_gian":  time.time() - t0_epoch,
                })

        # ── Cuối epoch ────────────────────────────────────────────────────
        dt_epoch     = time.time() - t0_epoch
        t_tich_luy   = tong_thoi_gian + (time.time() - t_phien)
        train_avg_ep = tong_mm_ep / max(dem_buoc_ep, 1)

        if (epoch + 1) % save_ep_every == 0:
            la_best_epoch = last_val_loss < best_val_loss
            quan_ly.luu_checkpoint(
                mo_hinh, bo_toi_uu,
                epoch          = epoch,
                step           = so_buoc_epoch - 1,
                train_loss     = train_avg_ep,
                val_loss       = last_val_loss,
                tong_thoi_gian = t_tich_luy,
                la_best        = la_best_epoch,
                cfg            = cfg,
                xoa_cu         = cfg.get("delete_old_checkpoints", True),
            )
            if la_best_epoch:
                best_val_loss = last_val_loss
        else:
            quan_ly.luu_latest_nhe(
                mo_hinh, bo_toi_uu,
                epoch          = epoch,
                step           = so_buoc_epoch - 1,
                train_loss     = train_avg_ep,
                val_loss       = last_val_loss,
                tong_thoi_gian = t_tich_luy,
                cfg            = cfg,
            )

        print(f"\n{CYAN}{DAM}{SEP}{TAT}")
        print(
            f"{CYAN}{DAM}"
            f"  XONG VÒNG {epoch} | "
            f"train={train_avg_ep:.4f} | "
            f"tốt nhất={best_val_loss:.4f} (ppl={_tinh_perplexity(best_val_loss):.4f}) | "
            f"{dt_epoch:.1f}s | "
            f"tích lũy={t_tich_luy/3600:.2f}h | "
            f"ETA={eta_tracker.eta(tong_buoc - buoc_toan_cuc)}"
            f"{TAT}"
        )

        gen_every = cfg.get("generate_every_n_epochs", 1)
        if (epoch + 1) % gen_every == 0:
            mo_hinh.eval()
            _mo_gen = mo_hinh.module if isinstance(mo_hinh, nn.DataParallel) else mo_hinh
            van_ban = _sinh_van_ban(_mo_gen, tokenizer, cfg, device)
            mo_hinh.train()
            if van_ban:
                print(f"\n  {DAM}[SINH VĂN BẢN]{TAT}")
                print(f"  {'-'*60}")
                print(f"  {van_ban[:300]}")
                print(f"  {'-'*60}")

        _in_bang_lich_su(lich_su[-10:])

    # ── Kết thúc ──────────────────────────────────────────────────────────
    print(f"\n{XANH}{DAM}{SEP}{TAT}")
    print(f"{XANH}{DAM}  HOÀN TẤT HUẤN LUYỆN{TAT}")
    if lich_su:
        br = min(lich_su, key=lambda r: r["val_loss"])
        print(f"{XANH}{DAM}  Mất mát tốt nhất : {br['val_loss']:.4f}  "
              f"(vòng {br['epoch']} bước {br['step']}){TAT}")
        print(f"{XANH}{DAM}  Độ phức tạp tốt nhất : {br['perplexity']:.4f}{TAT}")
    print(f"{XANH}{DAM}  Mô hình tốt nhất tại : "
          f"{OUTPUT_DIR / QuanLyCheckpoint.TEN_BEST}{TAT}")
    print(f"{XANH}{DAM}{SEP}{TAT}\n")


if __name__ == "__main__":
    huan_luyen(TRAIN_CONFIG)
