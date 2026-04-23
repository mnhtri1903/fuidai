"""
Fuid Project: An experimental conversational agent.

This project, developed by Wbiu (Nguyễn Minh Trí), explores the creation of a more natural, human-like conversational experience in Vietnamese. The core idea is to move beyond typical "assistant" paradigms.

Key characteristics include a distinct personality, dynamic register-switching based on user address (e.g. `cậu-tớ` vs. `tao-mày`), and use of informal text-based emoticons. All conversation history and context are stored locally to ensure user privacy.

Technical specifications:
Internal Model: fuidai-0.01
Language: Python
Developer: Wbiu · Nguyễn Minh Trí
"""
import math
import unicodedata
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenizerTV:
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SPECIALS = [PAD, UNK, BOS, EOS]

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
        self.vocab_size: int = 0

    @staticmethod
    def chuan_hoa(van_ban: str) -> str:
        van_ban = unicodedata.normalize("NFC", van_ban)
        van_ban = re.sub(r"[ \t]+", " ", van_ban)
        van_ban = re.sub(r"\n{3,}", "\n\n", van_ban)
        return van_ban.strip()

    def xay_tu_vung(self, van_ban: str):
        unique_chars = sorted(list(set(van_ban)))
        normalized_chars = sorted(list({unicodedata.normalize("NFC", char) for char in unique_chars}))

        self.char2idx = {}
        self.idx2char = {}
        for i, tok in enumerate(self.SPECIALS):
            self.char2idx[tok] = i
            self.idx2char[i] = tok
        
        offset = len(self.SPECIALS)
        for i, ky_tu in enumerate(normalized_chars):
            idx = i + offset
            self.char2idx[ky_tu] = idx
            self.idx2char[idx] = ky_tu
            
        self.vocab_size = len(self.char2idx)
        print(f"Kích thước từ vựng của bộ mã hóa: {self.vocab_size} ký tự")

    def ma_hoa(self, van_ban: str, them_bos_eos: bool = False) -> list[int]:
        # Dùng .get() tránh KeyError khi vocab không có special tokens
        unk_idx = self.char2idx.get(self.UNK, 0)
        ids = [self.char2idx.get(c, unk_idx) for c in van_ban]
        if them_bos_eos:
            bos_idx = self.char2idx.get(self.BOS, 0)
            eos_idx = self.char2idx.get(self.EOS, 0)
            ids = [bos_idx] + ids + [eos_idx]
        return ids

    def giai_ma(self, ids: list[int], bo_specials: bool = True) -> str:
        specials_set = set(self.SPECIALS)
        ky_tu = []
        for idx in ids:
            tok = self.idx2char.get(idx, self.UNK)
            if bo_specials and tok in specials_set:
                continue
            ky_tu.append(tok)
        return "".join(ky_tu)

    def luu(self, duong_dan: str):
        import json, pathlib
        pathlib.Path(duong_dan).parent.mkdir(parents=True, exist_ok=True)
        with open(duong_dan, "w", encoding="utf-8") as f:
            json.dump({"char2idx": self.char2idx, "idx2char": {str(k): v for k, v in self.idx2char.items()}}, f, ensure_ascii=False)
        print(f"Đã lưu bộ mã hóa từ vựng vào {duong_dan}")

    @classmethod
    def tai(cls, duong_dan: str):
        import json
        tok = cls()
        with open(duong_dan, "r", encoding="utf-8") as f:
            d = json.load(f)
        
        # Hỗ trợ cả 2 định dạng:
        # Mới: {"char2idx": {...}, "idx2char": {...}}
        # Cũ (chuan_bi_du_lieu.py): flat {char: idx}
        try:
            tok.char2idx = d["char2idx"]
            tok.idx2char = {int(k): v for k, v in d["idx2char"].items()}
        except KeyError:
            print("Cảnh báo: vocab.json định dạng cũ, tự động chuyển đổi...")
            tok.char2idx = dict(d)
            tok.idx2char = {int(v): k for k, v in d.items()}

        # KHÔNG thêm special tokens — vocab_size phải khớp model đã train
        tok.vocab_size = len(tok.char2idx)
        return tok

class MaHoaViTri(nn.Module):
    def __init__(self, d_model: int, kich_thuoc_khoi: int, ty_le_bo_qua: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(ty_le_bo_qua)
        pe = torch.zeros(kich_thuoc_khoi, d_model)
        vi_tri = torch.arange(0, kich_thuoc_khoi).unsqueeze(1).float()
        mau_so = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(vi_tri * mau_so)
        pe[:, 1::2] = torch.cos(vi_tri * mau_so)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TuChuYNhanQua(nn.Module):
    def __init__(self, d_model: int, so_dau: int, kich_thuoc_khoi: int, ty_le_bo_qua: float):
        super().__init__()
        assert d_model % so_dau == 0
        self.so_dau = so_dau
        self.d_model = d_model
        self.d_dau = d_model // so_dau
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(ty_le_bo_qua)
        self.resid_drop = nn.Dropout(ty_le_bo_qua)
        mask = torch.tril(torch.ones(kich_thuoc_khoi, kich_thuoc_khoi))
        self.register_buffer("mask", mask.view(1, 1, kich_thuoc_khoi, kich_thuoc_khoi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        def tach_dau(t):
            return t.view(B, T, self.so_dau, self.d_dau).transpose(1, 2)
        q, k, v = tach_dau(q), tach_dau(k), tach_dau(v)
        diem = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_dau))
        diem = diem.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        trong_so = F.softmax(diem, dim=-1)
        trong_so = self.attn_drop(trong_so)
        dau_ra = trong_so @ v
        dau_ra = dau_ra.transpose(1, 2).contiguous().view(B, T, C)
        dau_ra = self.resid_drop(self.c_proj(dau_ra))
        return dau_ra

class MangDaTang(nn.Module):
    def __init__(self, d_model: int, ty_le_bo_qua: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(ty_le_bo_qua),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class KhoiTransformer(nn.Module):
    def __init__(self, d_model: int, so_dau: int, kich_thuoc_khoi: int, ty_le_bo_qua: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TuChuYNhanQua(d_model, so_dau, kich_thuoc_khoi, ty_le_bo_qua)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = MangDaTang(d_model, ty_le_bo_qua)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class fuidai(nn.Module):
    def __init__(self, kich_thuoc_tu_vung: int, d_model: int, so_lop: int, so_dau: int, kich_thuoc_khoi: int, ty_le_bo_qua: float = 0.1):
        super().__init__()
        self.kich_thuoc_khoi = kich_thuoc_khoi
        self.wte = nn.Embedding(kich_thuoc_tu_vung, d_model)
        self.pe = MaHoaViTri(d_model, kich_thuoc_khoi, ty_le_bo_qua)
        self.cac_khoi = nn.ModuleList([KhoiTransformer(d_model, so_dau, kich_thuoc_khoi, ty_le_bo_qua) for _ in range(so_lop)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, kich_thuoc_tu_vung, bias=False)
        self.lm_head.weight = self.wte.weight
        self._khoi_tao_trong_so()
        so_tham_so = sum(p.numel() for p in self.parameters())
        print(f"[fuidai] Tổng số tham số: {so_tham_so:,} ({so_tham_so / 1e6:.2f}M)")

    def _khoi_tao_trong_so(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor, nhan: torch.Tensor = None):
        B, T = ids.shape
        assert T <= self.kich_thuoc_khoi, f"Sequence length {T} exceeds block size {self.kich_thuoc_khoi}"
        x = self.wte(ids)
        x = self.pe(x)
        for khoi in self.cac_khoi:
            x = khoi(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loi = None
        if nhan is not None:
            loi = F.cross_entropy(logits.view(-1, logits.size(-1)), nhan.view(-1), ignore_index=0)
        return logits, loi

    @torch.no_grad()
    def sinh_van_ban(self, ids_bat_dau: torch.Tensor, so_token_moi: int, nhiet_do: float = 1.0, top_k: int = None) -> torch.Tensor:
        self.eval()
        ids = ids_bat_dau
        for _ in range(so_token_moi):
            ids_ngu_canh = ids[:, -self.kich_thuoc_khoi:]
            logits, _ = self(ids_ngu_canh)
            logits = logits[:, -1, :] / nhiet_do
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                nguong, _ = torch.topk(logits, k)
                nguong_min = nguong[:, [-1]]
                logits[logits < nguong_min] = float("-inf")
            xac_suat = F.softmax(logits, dim=-1)
            token_moi = torch.multinomial(xac_suat, 1)
            ids = torch.cat([ids, token_moi], dim=1)
        return ids

    @torch.no_grad()
    def sinh_van_ban_streaming(self, ids_bat_dau: torch.Tensor, so_token_moi: int, nhiet_do: float = 1.0, top_k: int = None):
        self.eval()
        ids = ids_bat_dau
        for _ in range(so_token_moi):
            ids_ngu_canh = ids[:, -self.kich_thuoc_khoi:]
            logits, _ = self(ids_ngu_canh)
            logits = logits[:, -1, :] / nhiet_do
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                nguong, _ = torch.topk(logits, k)
                nguong_min = nguong[:, [-1]]
                logits[logits < nguong_min] = float("-inf")
            xac_suat = F.softmax(logits, dim=-1)
            token_moi = torch.multinomial(xac_suat, 1)
            yield token_moi.item()
            ids = torch.cat([ids, token_moi], dim=1)