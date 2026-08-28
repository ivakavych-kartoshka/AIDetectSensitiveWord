# SensitiveAI Proxy

Proxy trung gian đứng giữa **Next.js proxy của frevia-clone** và **Backend NestJS** để chèn AI quét nội dung trước khi bài viết được lưu xuống database.

## Luồng hoạt động (tích hợp frevia-clone)

```
Browser (bấm đăng bài)
        │
        ▼
Next.js FE (port 3001)  route /api/backend/api/forums/posts
        │  route.ts chọn: POST /api/forums/posts → NESTJS_PROXY_URL (3002)
        │                 còn lại → NESTJS_API_URL (3000)
        ▼
PROXY (port 3002)
        │  (chỉ nhận POST tạo bài; các request khác không đi qua proxy)
        ▼
Gọi SensitiveAI /analyze (port 8000)
        ▼
Tính điểm (score 0..1)
 ├─ score ≥ 0.8      (BLOCK)  ──► Trả lỗi 400, KHÔNG ghi DB
 ├─ 0.3 ≤ score < 0.8 (REVIEW) ──► Forward xuống BE kèm moderation {status:"PENDING"}
 │                                  → BE lưu bài, status = PENDING (chưa hiển thị công khai)
 │                                  → Admin duyệt: Approve → APPROVED (hiển thị) / Reject → REJECTED (trash)
 └─ score < 0.3      (ALLOW)  ──► Forward xuống BE kèm moderation {status:"APPROVED"}
                                    → BE lưu bài, status = APPROVED (hiển thị ngay)
        ▼
BACKEND NestJS (port 3000)
```

## Cài đặt

```bash
cd proxy
copy .env.example .env   # Windows
# hoặc: cp .env.example .env

npm install
```

## Chạy

```bash
npm start        # chạy production
npm run dev      # chạy dev (auto reload)
```

Proxy mặc định chạy ở: `http://127.0.0.1:3002`

## Cấu hình (.env)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `PROXY_PORT` | `3002` | Cổng proxy |
| `AI_URL` | `http://127.0.0.1:8000` | URL SensitiveAI |
| `AI_ANALYZE_PATH` | `/analyze` | Endpoint quét |
| `BACKEND_URL` | `http://127.0.0.1:3000` | URL Backend đích |
| `POST_CREATE_METHOD` | `POST` | Method tạo bài |
| `POST_CREATE_PATH` | `/api/forums/posts` | Route tạo bài **trên proxy** (sau khi Next.js proxy bỏ prefix `/api/backend`). Khớp chính xác, không phải startsWith |
| `TEXT_FIELDS` | `title,content` | Các field trong JSON body dùng để quét |

> **Lưu ý:** `POST_CREATE_PATH` phải là path mà Next.js gửi xuống **proxy** (tức `/api/forums/posts`), không phải path browser `/api/backend/api/forums/posts`.

## Cách FE (frevia-clone) trỏ qua proxy

Không cần sửa code FE — **chỉ chạy bài đăng forum qua proxy** bằng biến riêng `NESTJS_PROXY_URL`. Backend `NESTJS_API_URL` vẫn trỏ thẳng NestJS để mọi request khác (auth, đọc bài, comment, socket.io, upload) không bị ảnh hưởng (proxy có thể tạo single point of failure/không xử lý websocket).

### Bước 1 — Set biến proxy trên Infisical (`--path=/web`):

```bash
cd C:\AI\frevia-clone

# Web (`--path=/web`):
infisical secrets set NESTJS_API_URL=http://127.0.0.1:3000 --env=dev --path=/web   # backend thật
infisical secrets set NESTJS_PROXY_URL=http://127.0.0.1:3002 --env=dev --path=/web # proxy SensitiveAI
```

(với `dev:local` thì thêm vào file `.env.local` của `apps/web`.)

### Bước 2 — `NESTJS_PROXY_URL` chỉ ảnh hưởng POST tạo bài forum

Route Next.js `app/api/backend/[...path]/route.ts` (đã chỉnh) dùng:
- `POST /api/forums/posts` (path chính xác) → forward sang `NESTJS_PROXY_URL` (proxy → SensitiveAI)
- Các POST khác + PUT/PATCH/DELETE comment/like/report → forward thẳng `NESTJS_API_URL` (NestJS)
- refresh-token, socket.io → thẳng NestJS (proxy không xử lý)

Header `Authorization: Bearer <token>` vẫn được chuyển tiếp nên auth hoạt động bình thường.

## Response khi bị chặn

**BLOCK (HTTP 400):**
```json
{
  "success": false,
  "message": "Nội dung chứa từ ngữ nhạy cảm. Không cho phép đăng bài.",
  "moderation": { "decision": "block", "score": 0.95, "categories": ["sexual"], "matches": [...] }
}
```

**Forward thành công (theo score):**
- `score < 0.3` → proxy gắn `moderation.status = "APPROVED"` → bài hiển thị ngay, không cần admin.
- `0.3 ≤ score < 0.8` → proxy gắn `moderation.status = "PENDING"` → BE lưu bài (chưa hiển thị), admin duyệt sau.

```json
// body gửi xuống BE
{
  "title": "...",
  "content": "...",
  "moderation": { "status": "PENDING", "score": 0.55, "categories": ["insult"] }
}
```
BE chỉ lưu các status `PENDING` / `APPROVED` / `REJECTED` (không có `ALLOW`).

## Lưu ý

- Nếu AI không khả dụng (lỗi mạng/không chạy), proxy **fail-open** — forward bài đi tiếp để không chặn oan người dùng. Muốn fail-closed, hãy sửa logic bắt lỗi trong middleware.
- Body mọi request được buffer và forward nguyên vẹn (kể cả multipart upload), proxy chỉ đọc-nội-dung để quét chứ không sửa body.
- Mọi path không phải route tạo bài (VD: đọc bài, auth, upload...) đều forward thẳng tới BE, không qua AI.
