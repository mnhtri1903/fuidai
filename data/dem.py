"""
dem.py - file dùng để đếm token trong thư mục/file chỉ định

cách dùng:
 python dem.py 

"""

import os
import json
import re
import time
from pathlib import Path

def bo_dem_token_tuy_chinh(van_ban):
    if not isinstance(van_ban, str):
        van_ban = str(van_ban)
    tokens = re.findall(r'\w+|[^\w\s]', van_ban, re.UNICODE)
    return int(len(tokens) * 1.2)

def doc_va_dem_file(duong_dan_file):
    tong_token = 0
    try:
        with open(duong_dan_file, 'r', encoding='utf-8') as f:
            if duong_dan_file.endswith('.jsonl'):
                for dong in f:
                    dong = dong.strip()
                    if dong:
                        data = json.loads(dong)
                        van_ban = data.get("van_ban", data.get("text", str(data)))
                        tong_token += bo_dem_token_tuy_chinh(van_ban)
            elif duong_dan_file.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    for muc in data:
                        if isinstance(muc, dict):
                            van_ban = muc.get("van_ban", muc.get("text", str(muc)))
                            tong_token += bo_dem_token_tuy_chinh(van_ban)
                        elif isinstance(muc, str):
                            tong_token += bo_dem_token_tuy_chinh(muc)
                elif isinstance(data, dict):
                    van_ban = data.get("van_ban", data.get("text", str(data)))
                    tong_token += bo_dem_token_tuy_chinh(van_ban)            
    except Exception as e:
        print(f"Không thể đọc {os.path.basename(duong_dan_file)}: {e}")
    return tong_token

def dinh_dang_so(n):
    if n >= 1_000_000_000:
        return f"{n/1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1_000:
        return f"{n/1e3:.2f}K"
    return str(n)

def main():
    print("="*50)
    print("   TOKENS TỪ NHIỀU FILE JSON")
    print("="*50)
    duong_dan_nhap = input("nhập path: ")
    if duong_dan_nhap.startswith('"') and duong_dan_nhap.endswith('"'):
        duong_dan_nhap = duong_dan_nhap[1:-1]
    thu_muc = Path(duong_dan_nhap)
    if not thu_muc.exists():
        print(f"Đường dẫn không tồn tại: {thu_muc}")
        return
    danh_sach_file = []
    if thu_muc.is_file() and thu_muc.suffix in ['.json', '.jsonl']:
        danh_sach_file.append(thu_muc)
    elif thu_muc.is_dir():
        danh_sach_file.extend(thu_muc.rglob('*.json'))
        danh_sach_file.extend(thu_muc.rglob('*.jsonl'))
    else:
        print("Vui lòng trỏ đến một file .json/.jsonl hoặc thư mục chứa chúng.")
        return
    if not danh_sach_file:
        print("Không tìm thấy file JSON nào trong đường dẫn này.")
        return
    print(f"\nĐang quét {len(danh_sach_file)} file...")
    t_bat_dau = time.time()
    tong_cong_token = 0
    for file_path in danh_sach_file:
        token_file = doc_va_dem_file(str(file_path))
        tong_cong_token += token_file
        print(f" - {file_path.name}: {dinh_dang_so(token_file)} tokens")
    t_ket_thuc = time.time()
    print("\n" + "="*50)
    print(" TỔNG KẾT")
    print(f" Tổng số file đã quét: {len(danh_sach_file)}")
    print(f" Tổng số Tokens (Ước tính): {tong_cong_token:,} ({dinh_dang_so(tong_cong_token)})")
    print(f" Thời gian quét: {t_ket_thuc - t_bat_dau:.2f} giây")
    print("="*50)

if __name__ == "__main__":
    main()