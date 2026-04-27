import json
import os
from datasets import load_dataset

def tai_hf_va_chia_file(ten_dataset, thu_muc_luu="data_hf_da_chia", so_mau_moi_file=50000):
    if not os.path.exists(thu_muc_luu):
        os.makedirs(thu_muc_luu)
        
    print(f"[*] Đang kết nối tải dataset: {ten_dataset}")
    
    # Bạn có thể thêm token=HF_TOKEN vào trong load_dataset nếu dataset bị khóa (private)
    try:
        ds = load_dataset(ten_dataset, split="train", streaming=True)
        
        file_hien_tai = 1
        mau_da_ghi = 0
        
        ten_file = os.path.join(thu_muc_luu, f"data_phan_{file_hien_tai}.jsonl")
        f = open(ten_file, 'w', encoding='utf-8')
        print(f"> Đang tạo và ghi: {ten_file}")
        
        for mau in ds:
            # Lưu định dạng JSONL (mỗi mẫu 1 dòng)
            f.write(json.dumps(mau, ensure_ascii=False) + "\n")
            mau_da_ghi += 1
            
            # Cắt file khi đủ dung lượng dòng
            if mau_da_ghi >= so_mau_moi_file:
                f.close()
                file_hien_tai += 1
                mau_da_ghi = 0
                
                ten_file = os.path.join(thu_muc_luu, f"data_phan_{file_hien_tai}.jsonl")
                f = open(ten_file, 'w', encoding='utf-8')
                print(f"> Đang tạo và ghi: {ten_file}")
                
        f.close()
        print("\n[+] Hoàn tất việc chia dataset Hugging Face!")
        
    except Exception as e:
        print(f"[-] Lỗi khi tải Hugging Face dataset: {e}")

# Chạy chương trình
if __name__ == "__main__":
    link_hf = "hoanghai2110/vi-pretrain-clean"  # Thay bằng tên dataset Hugging Face bạn muốn tải
    # Đang thiết lập cứ 50,000 bài viết là cắt ra 1 file jsonl riêng
    tai_hf_va_chia_file(link_hf, so_mau_moi_file=50000)