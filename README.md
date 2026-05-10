# Fuid Project: Vietnamese Conversational AI (updating...)

**Fuid** (phiên bản mô hình `fuidai`) là một dự án thử nghiệm về Artificial Intelligence (AI) dành riêng cho tiếng Việt.

Dự án được phát triển bởi **Wbiu (Nguyễn Minh Trí)**.

---

## Đặc điểm nổi bật

* **Natural Vietnamese**: Tối ưu hóa cho tiếng Việt tự nhiên, bao gồm cả việc sử dụng các ký tự biểu cảm (emoticons) và văn phong không chính thức.
* **Local & Private**: Toàn bộ lịch sử trò chuyện và quá trình xử lý được thực hiện cục bộ (Local), đảm bảo quyền riêng tư tuyệt đối.
* **Kiến trúc Transformer**: Xây dựng trên nền tảng PyTorch với cơ chế Causal Self-Attention, Layer Normalization và Positional Encoding.

---

## Thông số kỹ thuật (Model Architecture)

Mô hình mặc định được cấu hình với các tham số:
* **Model Dimension (d_model)**: 0
* **Layers**: 0
* **Attention Heads**: 0
* **Block Size**: 0
* **Vocabulary**: Character-level Tokenizer tối ưu cho tiếng Việt.

---

## Cấu trúc dự án
* `đang update`
---

## Quy trình xử lý dữ liệu (Data Pipeline)

Để đảm bảo chất lượng mô hình, dữ liệu đầu vào phải trải qua hệ thống lọc chuyên sâu:

1. **Làm sạch kỹ thuật**: Loại bỏ HTML, mã rác, lỗi encoding và chuẩn hóa khoảng trắng.
2. **Lọc nội dung**: Loại bỏ Spam, quảng cáo, từ khóa SEO và các tiêu đề vô nghĩa.
3. **Kiểm soát ngôn ngữ**: Chỉ giữ lại các câu hoàn chỉnh, tự nhiên. Loại bỏ nội dung gãy gọn hoặc vô nghĩa.
4. **Tiêu chuẩn an toàn**: Loại bỏ nội dung 18+, cực đoan hoặc bạo lực.

---

## Hướng dẫn sử dụng

### 1. Chuẩn bị môi trường
```bash

```

### 2. Chuẩn bị dữ liệu
Đặt dữ liệu thô (`.json` hoặc `.jsonl`) vào thư mục dữ liệu và chạy:
```bash

```

### 3. Huấn luyện mô hình
Để bắt đầu huấn luyện mới:
```bash

```
Để huấn luyện tiếp từ một checkpoint:
```bash

```

### 4. Trò chuyện
Sau khi có checkpoint trong thư mục ``, chạy:
```bash

```

---

## Quy định về dữ liệu đầu ra (Lọc chuyên sâu)
Hệ thống lọc dữ liệu của dự án tuân thủ các quy tắc nghiêm ngặt:
* Định dạng đầu ra luôn là **JSON**.
* Mỗi câu là một dòng riêng biệt, không có ký tự định dạng Markdown.
* Giữ nguyên trạng câu gốc, không paraphrase, không thêm nội dung mới.

---
*Phát triển bởi Wbiu · Nguyễn Minh Trí*
