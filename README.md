# 🛡️ SensitiveAI

## Hệ thống phát hiện và kiểm duyệt nội dung nhạy cảm bằng AI

---

# 📖 Giới thiệu

SensitiveAI là hệ thống **AI phát hiện từ ngữ nhạy cảm** trong tiếng Anh và tiếng Việt được xây dựng nhằm hỗ trợ các diễn đàn, mạng xã hội, hệ thống bình luận và website tự động kiểm duyệt nội dung trước khi cho phép đăng tải.

Hệ thống sử dụng **Hybrid AI**, kết hợp:

- ✅ Rule-based Detection – Hệ thống luật (Keyword, Regex) phát hiện nhanh các từ ngữ và mẫu nội dung nhạy cảm.
  - Keyword: Các từ ngữ liên quan đến các vấn đề nhạy cảm.
  - Regex: Nhằm phát hiện các bài đăng có chứa các thông tin riêng tử của người dùng.
- ✅ Machine Learning (Transformer) – Mô hình AI phân tích ngữ cảnh và phát hiện các nội dung nhạy cảm mà Rule-based có thể bỏ sót.
- ✅ Text Normalization – Chuẩn hóa văn bản (Unicode, ký tự đặc biệt, viết tắt, ký tự lặp...) trước khi phân tích.
- ✅ Obfuscation Detection – Phát hiện các kỹ thuật lách luật bằng cách cố tình biến đổi từ ngữ nhạy cảm.

## để đạt được khả năng phát hiện chính xác hơn so với chỉ sử dụng từ điển từ cấm.

# 🎯 Mục tiêu

Ngăn chặn các nội dung:

- Chửi tục
- Công kích cá nhân
- Nội dung khiêu dâm
- Hate Speech
- Toxic Content
- Nội dung bị che giấu (obfuscation)

Ví dụ:
đụ má
dmm
đm
đ!t m3
đ*j m*
f\*ck you
go kill yourself
...

đều có thể được phát hiện chỉ cần bài đăng chứa 1 từ ngữ mang chủ đề nhạy cảm.

---

# 🧠 Kiến trúc Hybrid AI

                User Input
                     │
                     ▼
          Text Normalization
                     │
                     ▼
      ┌─────────────────────────┐
      │                         │
      ▼                         ▼

Rule Engine AI Transformer

      │                         │
      └──────────────┬──────────┘
                     ▼

             Hybrid Decision

                     ▼

         Allow / Review / Block

    + Allow: SAFE - là các nội dung an toàn không chứ chủ đề hoặc nội dung nhạy cảm.
    + Review: MAYBE SENSITIVE - là các nội dung có thể mang chủ đề, nội dung nhạy cảm hoặc không. Cần được review lại.
    + Block: SENSITIVE - là các nội dung mang chủ đề và nội dung nhạy cảm không được đăng lên.

---

# 🚀 Các thành phần

## Rule Engine

Rule Engine kiểm tra:

### Rule Engine

- **Keyword Matching** – Kiểm tra nội dung có chứa các từ khóa hoặc cụm từ thuộc danh sách từ ngữ nhạy cảm được xây dựng sẵn hay không.

- **Regex Pattern** – Phát hiện các mẫu nội dung theo biểu thức chính quy (Regular Expression), chẳng hạn như thông tin cá nhân (số điện thoại, email, CCCD...), đường dẫn, hoặc các mẫu từ ngữ nhạy cảm.

- **Unicode Normalize** – Chuẩn hóa văn bản về định dạng Unicode thống nhất nhằm tránh các trường hợp cùng một ký tự nhưng được biểu diễn dưới nhiều dạng khác nhau.

- **Zero Width Character Detection** – Phát hiện và loại bỏ các ký tự Zero Width - các ký tự trống không nhìn thấy nhưng vẫn tồn tại trong chuỗi (Zero Width Space, Zero Width Joiner,...) được chèn vào để che giấu từ ngữ nhạy cảm.

- **Homoglyph Detection** – Phát hiện các ký tự có hình dạng tương tự nhau nhưng thuộc các bảng mã khác nhau (ví dụ: chữ "a" Latin và "а" Cyrillic) nhằm ngăn chặn việc lách luật.

- **Leetspeak Detection** – Nhận diện các từ được thay thế bằng số hoặc ký tự đặc biệt (ví dụ: `h3ll`, `5ex`, `d!t`, `f*ck`) để che giấu nội dung nhạy cảm.

- **Repeated Characters Detection** – Chuẩn hóa và phát hiện các từ có ký tự bị lặp nhiều lần nhằm né tránh kiểm duyệt (ví dụ: `đmmmm`, `nguuuuuu`, `vlllllll`).

- **Obfuscation Detection** – Phát hiện các kỹ thuật che giấu hoặc biến đổi từ ngữ nhạy cảm bằng cách chèn khoảng trắng, dấu câu, ký tự đặc biệt hoặc viết tắt nhằm lách luật (ví dụ: `đ*m`, `đ.ụ`, `d m`, `dmm`, `v*l`).

---

## AI Model

Sử dụng Transformer được huấn luyện để phân loại nhiều loại nội dung.

Ví dụ output:

sexual
hate
violence
toxic
insult

Model được load thông qua:

Predictor()

và suy luận bằng:

ml_model.predict(text)

---

## Hybrid Decision

Rule Engine và AI Model cùng tham gia quyết định.

Ví dụ:

Rule Score = 0.20
AI Score = 0.96
Final Score = 0.96
Decision = BLOCK

Hoặc

Rule Score = 0.85
AI Score = 0.10
Final Score = 0.85
Decision = BLOCK

---

# ⚙️ Cài đặt

## Clone project

```bash
git clone https://github.com/yourname/SensitiveAI.git

cd SensitiveAI
```

---

## Tạo môi trường

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux

```bash
source .venv/bin/activate
```

---

## Cài thư viện

```bash
pip install -r requirements.txt
```

---

# ▶️ Chạy API

```bash
uvicorn src.api.app:app
```

API sẽ chạy tại:

```
http://127.0.0.1:8000
```

Swagger:

```
http://127.0.0.1:8000/docs
```

---

# 🌐 Giao diện Demo

Trang chủ:

```
http://127.0.0.1:8000
```

Mô phỏng chức năng:

✅ Người dùng tạo bài viết

↓

AI tự động quét

↓

Nếu có nội dung nhạy cảm

↓

❌ Không cho phép đăng bài

Ngược lại

↓

✅ Đăng thành công

---

# API Documentation

Base URL

```
http://localhost:8000
```

---

# POST /analyze

Phân tích toàn bộ nội dung bằng **Hybrid AI** (Rule-based + Machine Learning).

Đây là API chính được sử dụng khi kiểm duyệt bài viết.

## Request

```json
{
  "text": "đụ má"
}
```

## Response

```json
{
  "decision": "block",
  "risk": "critical",
  "score": 0.95,
  "has_sensitive": true,
  "categories": ["sexual"],
  "matches": [
    {
      "keyword": "đụ",
      "category": "sexual",
      "confidence": 0.85
    }
  ],
  "report": {
    "recommendation": "Reject content"
  }
}
```

## Decision

| Giá trị  | Ý nghĩa                            |
| -------- | ---------------------------------- |
| `allow`  | Nội dung an toàn, cho phép đăng    |
| `review` | Nội dung cần quản trị viên xem xét |
| `block`  | Nội dung bị chặn                   |

---

# POST /predict

Trả về quyết định cuối cùng của hệ thống AI.

## Request

```json
{
  "text": "đụ má"
}
```

## Response

```json
{
  "decision": "block"
}
```

---

# POST /score

Trả về điểm rủi ro (Risk Score) của nội dung.

Điểm càng cao thì nội dung càng có khả năng chứa từ ngữ nhạy cảm.

## Request

```json
{
  "text": "đụ má"
}
```

## Response

```json
{
  "score": 0.92
}
```

| Score       | Mức độ      |
| ----------- | ----------- |
| 0.00 - 0.30 | Safe        |
| 0.30 - 0.70 | Medium Risk |
| 0.70 - 1.00 | High Risk   |

---

# POST /categories

Trả về các nhóm nội dung nhạy cảm được phát hiện.

## Request

```json
{
  "text": "đụ má"
}
```

## Response

```json
{
  "categories": ["sexual", "toxic"]
}
```

Ví dụ các nhóm:

- sexual
- toxic
- violence
- hate
- harassment
- spam

---

# POST /explain

Giải thích lý do hệ thống AI đưa ra quyết định.

API này hữu ích khi quản trị viên muốn biết vì sao bài viết bị chặn hoặc cần xem xét.

## Request

```json
{
  "text": "đụ má"
}
```

## Response

```json
{
  "decision": "block",
  "risk": "critical",
  "score": 0.95,
  "matched_keywords": ["đụ"],
  "matched_categories": ["sexual"],
  "recommendation": "Reject content",
  "summary": {
    "category_count": 1,
    "match_count": 1
  },
  "ml_result": [
    {
      "label": "sexual",
      "confidence": 0.991,
      "predicted": true
    },
    {
      "label": "toxic",
      "confidence": 0.034,
      "predicted": false
    }
  ]
}
```

Thông tin trả về bao gồm:

- **decision**: Quyết định cuối cùng của hệ thống.
- **risk**: Mức độ rủi ro.
- **score**: Điểm đánh giá.
- **matched_keywords**: Từ khóa được Rule Engine phát hiện.
- **matched_categories**: Các nhóm nội dung nhạy cảm.
- **recommendation**: Khuyến nghị xử lý.
- **summary**: Thống kê số lượng từ khóa và danh mục được phát hiện.
- **ml_result**: Kết quả dự đoán từ mô hình Transformer AI.

---

# POST /batch

Phân tích nhiều đoạn văn cùng lúc.

## Request

```json
{
  "texts": ["Xin chào mọi người", "đụ má", "Bạn thật ngu"]
}
```

## Response

```json
{
  "count": 3,
  "results": [
    {
      "decision": "allow"
    },
    {
      "decision": "block"
    },
    {
      "decision": "review"
    }
  ]
}
```

---

# GET /health

Kiểm tra trạng thái hoạt động của API.

## Response

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

# 🤖 AI Model

Model được huấn luyện bằng:

- PyTorch
- HuggingFace Transformers

Quá trình suy luận:

```
Input Text

↓

Tokenizer

↓

Transformer

↓

Logits

↓

Confidence

↓

Predict Labels
```

Ví dụ:

```
sexual

0.98

True
```

---

# 📊 Hybrid Scoring

Điểm cuối cùng được tính bằng:

```
Final Score

=

max(

Rule Score,

AI Score

)
```

Sau đó đưa ra quyết định:

| Score     | Decision |
| --------- | -------- |
| <0.30     | Allow    |
| 0.30~0.79 | Review   |
| ≥0.80     | Block    |

---

# 🖥️ Công nghệ sử dụng

## Backend

- Python
- FastAPI

## AI

- PyTorch
- HuggingFace Transformers

## Frontend

- HTML
- CSS
- JavaScript

## NLP

- Regex
- Unicode Normalize
- Rule Engine

---

# 📌 Ứng dụng

SensitiveAI có thể tích hợp vào:

- Diễn đàn
- Website
- Blog
- Social Network
- Chat System
- Comment System
- Livestream Chat
- Game Chat
- Marketplace
- LMS
- E-learning

---

# 📈 Hướng phát triển

- [ ] Fine-tune PhoBERT
- [ ] PhoBERT Large
- [ ] Gemma
- [ ] Llama
- [ ] RAG
- [ ] Explainable AI
- [ ] Active Learning
- [ ] Dashboard
- [ ] Admin Moderation
- [ ] Database Logging
- [ ] Docker
- [ ] CI/CD
- [ ] Kubernetes Deployment

---

# 👨‍💻 Tác giả

**Khả Vy Phạm**

Capstone Project

Trường Đại học FPT

---
