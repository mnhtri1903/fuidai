---

# Fuid Project: Vietnamese Conversational AI

**Fuid** (phiên bản mô hình `fuidai-0.01`) là một dự án thử nghiệm về tác nhân hội thoại (conversational agent) dành riêng cho tiếng Việt. Dự án tập trung vào việc tạo ra trải nghiệm giao tiếp tự nhiên, mang đậm tính cá nhân hóa và xóa bỏ rào cản của các mô hình "trợ lý ảo" cứng nhắc thông thường.

Dự án được phát triển bởi **Wbiu (Nguyễn Minh Trí)**.

---

## Đặc điểm nổi bật

* **Personality-driven**: Mô hình có tính cách riêng, sử dụng đại từ linh hoạt (cậu-tớ, tao-mày) dựa trên ngữ cảnh.
* **Natural Vietnamese**: Tối ưu hóa cho tiếng Việt tự nhiên, bao gồm cả việc sử dụng các ký tự biểu cảm (emoticons) và văn phong không chính thức.
* **Local & Private**: Toàn bộ lịch sử trò chuyện và quá trình xử lý được thực hiện cục bộ (Local), đảm bảo quyền riêng tư tuyệt đối.
* **Kiến trúc Transformer**: Xây dựng trên nền tảng PyTorch với cơ chế Causal Self-Attention, Layer Normalization và Positional Encoding.

---

## Thông số kỹ thuật (Model Architecture)

Mô hình mặc định được cấu hình với các tham số:
* **Model Dimension ($d_{model}$)**: 780
* **Layers**: 14
* **Attention Heads**: 12
* **Block Size**: 512 tokens
* **Vocabulary**: Character-level Tokenizer tối ưu cho tiếng Việt.

---

## Cấu trúc dự án

* `mo_hinh.py`: Định nghĩa kiến trúc lõi của Transformer (fuidai) và bộ mã hóa TokenizerTV.
* `chat.py`: Script giao diện dòng lệnh để tương tác trực tiếp với mô hình sau khi train.
* `huan_luyen.py`: Quy trình huấn luyện mô hình từ đầu (from scratch) với tính năng Auto-batch scaling.
* `hlpb2.py`: Script hỗ trợ huấn luyện tiếp tục (Resume training) từ checkpoint sẵn có.
* `chuan_bi_du_lieu.py`: Xử lý văn bản thô, xây dựng từ vựng và chuyển đổi sang định dạng `.bin` (numpy memmap) để tối ưu tốc độ đọc.
* `tao_data.py`: Tự động tải và trích xuất dữ liệu từ các tập dữ liệu lớn như C4 (Common Crawl).
* `dem.py`: Công cụ thống kê số lượng token trong tập dữ liệu.

---

## ⚙️ Quy trình xử lý dữ liệu (Data Pipeline)

Để đảm bảo chất lượng mô hình, dữ liệu đầu vào phải trải qua hệ thống lọc chuyên sâu:

1.  **Làm sạch kỹ thuật**: Loại bỏ HTML, mã rác, lỗi encoding và chuẩn hóa khoảng trắng.
2.  **Lọc nội dung**: Loại bỏ Spam, quảng cáo, từ khóa SEO và các tiêu đề vô nghĩa.
3.  **Kiểm soát ngôn ngữ**: Chỉ giữ lại các câu hoàn chỉnh, tự nhiên. Loại bỏ nội dung gãy gọn hoặc vô nghĩa.
4.  **Tiêu chuẩn an toàn**: Loại bỏ nội dung 18+, cực đoan hoặc bạo lực.

---

## Hướng dẫn sử dụng

### 1. Chuẩn bị môi trường
```bash
pip install torch numpy datasets
```

### 2. Chuẩn bị dữ liệu
Đặt dữ liệu thô (`.json` hoặc `.jsonl`) vào thư mục dữ liệu và chạy:
```bash
python chuan_bi_du_lieu.py
```

### 3. Huấn luyện mô hình
Để bắt đầu huấn luyện mới:
```bash
python huan_luyen.py
```
Để huấn luyện tiếp từ một checkpoint:
```bash
python hlpb2.py --ckpt dau_ra_fuidai/fuid_best.pt --epochs 100
```

### 4. Trò chuyện
Sau khi có checkpoint trong thư mục `dau_ra_fuidai`, chạy:
```bash
python chat.py
```

---

## Quy định về dữ liệu đầu ra (Lọc chuyên sâu)
Hệ thống lọc dữ liệu của dự án tuân thủ các quy tắc nghiêm ngặt:
* Định dạng đầu ra luôn là **JSON**.
* Mỗi câu là một dòng riêng biệt, không có ký tự định dạng Markdown.
* Giữ nguyên trạng câu gốc, không paraphrase, không thêm nội dung mới.

---
*Phát triển bởi Wbiu · Nguyễn Minh Trí*

---

Hy vọng bản README này phản ánh đúng tâm huyết và kỹ thuật bạn đã đặt vào dự án. Bạn có muốn tôi bổ sung chi tiết hơn về các công thức toán học trong cơ chế Attention hay hướng dẫn cấu hình GPU không?
