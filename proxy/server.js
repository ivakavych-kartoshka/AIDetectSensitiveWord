import 'dotenv/config';
import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';

const PROXY_PORT = Number(process.env.PROXY_PORT || 3002);
const AI_URL = process.env.AI_URL || 'http://127.0.0.1:8000';
const AI_ANALYZE_PATH = process.env.AI_ANALYZE_PATH || '/analyze';
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:3000';

// ==== MODERATION THRESHOLDS (score 0..1) ====
// score >= BLOCK_SCORE             -> BLOCK  (chặn, không forward xuống BE)
// score >= REVIEW_SCORE (< BLOCK)  -> REVIEW (forward xuống BE, status = PENDING, admin duyệt)
// score <  REVIEW_SCORE            -> ALLOW  (forward xuống BE, status = APPROVED -> hiển thị ngay)
const BLOCK_SCORE = Number(process.env.MODERATION_BLOCK_SCORE || 0.8);
const REVIEW_SCORE = Number(process.env.MODERATION_REVIEW_SCORE || 0.3);

const POST_CREATE_METHOD = (process.env.POST_CREATE_METHOD || 'POST').toUpperCase();
const POST_CREATE_PATH = process.env.POST_CREATE_PATH || '/api/backend/api/forums/posts';

const TEXT_FIELDS = (process.env.TEXT_FIELDS || 'title,content')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

const app = express();

// JSON: body-parser đọc stream và buffer raw body qua `verify`,
// giữ nguyên chunk để forward xuống BE. (Không thêm 'data' listener thủ công
// vì sẽ làm body-parser lỗi "stream is not readable".)
app.use(
  express.json({
    limit: '2mb',
    verify: (req, _res, buf) => {
      req.rawBody = buf;
      req.forwardBody = buf;
    },
  }),
);

// Content-type khác JSON (multipart/form-data upload...): body-parser bỏ qua,
// tự capture raw stream để vẫn forward nguyên vẹn body xuống BE.
app.use((req, res, next) => {
  if (req.headers['content-type']?.includes('application/json')) {
    return next();
  }
  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', () => {
    req.rawBody = Buffer.concat(chunks);
    req.forwardBody = req.rawBody;
    next();
  });
});

async function analyze(text) {
  const resp = await fetch(`${AI_URL}${AI_ANALYZE_PATH}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!resp.ok) {
    throw new Error(`AI analyze failed: ${resp.status}`);
  }
  return resp.json();
}

function toScore(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function extractScanTexts(body) {
  const parts = [];
  for (const field of TEXT_FIELDS) {
    const value = body?.[field];
    if (value !== undefined && value !== null) {
      parts.push(String(value));
    }
  }
  return parts.filter((s) => s.trim().length > 0).join('\n').trim();
}

// Khớp chính xác route tạo bài (không dùng startsWith để tránh quét nhầm
// các POST khác dưới /api/forums/posts như comments, like, reports).
const isCreatePost = (req) => {
  const normalizedPath = (POST_CREATE_PATH || '').replace(/\/+$/, '');
  if (!normalizedPath) return false;
  return (
    req.method.toUpperCase() === POST_CREATE_METHOD &&
    (req.path === normalizedPath || req.path === `${normalizedPath}/`)
  );
};

// Middleware moderation: quét mọi POST tạo bài ở route cấu hình
app.use(async (req, res, next) => {
  if (!isCreatePost(req)) {
    return next();
  }

  const userAgent = (req.headers['user-agent'] || '').toLowerCase();
  // Bỏ qua request nội bộ từ chính proxy để tránh loop (an toàn)
  if (userAgent.includes('sensitiveai-proxy')) {
    return next();
  }

  let body;
  try {
    if (req.headers['content-type']?.includes('application/json')) {
      body = req.body || {};
    } else {
      body = {};
    }
  } catch {
    body = {};
  }

  try {
    const text = extractScanTexts(body);
    const moderation = { decision: 'allow', status: 'APPROVED', score: 0, categories: [] };

    if (text) {
      const result = await analyze(text);
      moderation.decision = result.decision;
      moderation.score = toScore(result.score);
      moderation.categories = result.categories || [];
      moderation.matches = result.matches || [];
      moderation.risk = result.risk;
      moderation.recommendation = result.report?.recommendation;
    }

    const score = moderation.score;

    // BLOCK: score >= BLOCK_SCORE -> chặn trực tiếp, KHÔNG gửi request xuống BE
    if (score >= BLOCK_SCORE) {
      return res.status(400).json({
        success: false,
        message: 'Nội dung chứa từ ngữ nhạy cảm. Không cho phép đăng bài.',
        moderation: { ...moderation, status: 'BLOCKED' },
      });
    }

    // REVIEW: 0.3 <= score < 0.8 -> forward xuống BE, status = PENDING (admin duyệt)
    // ALLOW:  score < 0.3        -> forward xuống BE, status = APPROVED (hiển thị ngay)
    moderation.status = score >= REVIEW_SCORE ? 'PENDING' : 'APPROVED';

    // Gắn moderation vào JSON body (BE sẽ lưu status + score + categories)
    if (req.headers['content-type']?.includes('application/json')) {
      const parsed = JSON.parse(req.rawBody.toString('utf8') || '{}');
      parsed.moderation = moderation;
      req.forwardBody = Buffer.from(JSON.stringify(parsed), 'utf8');
    } else {
      req.forwardBody = req.rawBody;
    }

    return next();
  } catch (err) {
    // AI lỗi -> fail-open để không chặn oan người dùng
    console.error('[moderation] AI failed, pass-through:', err.message);
    req.forwardBody = req.rawBody;
    return next();
  }
});

const backendProxy = createProxyMiddleware({
  target: BACKEND_URL,
  changeOrigin: true,
  ws: true,
  logLevel: 'warn',
  on: {
    proxyReq: (proxyReq, req) => {
      // Body đã được body-parser đọc hết nên stream gốc đã cạn.
      // Phải ghi lại raw body đã buffer và kết thúc request, nếu không
      // upstream sẽ treo chờ body không bao giờ tới.
      if (req.forwardBody) {
        proxyReq.setHeader('Content-Type', req.headers['content-type'] || 'application/json');
        proxyReq.setHeader('Content-Length', Buffer.byteLength(req.forwardBody));
        proxyReq.write(req.forwardBody);
      }
      proxyReq.end();
    },
  },
});

app.use(backendProxy);

app.listen(PROXY_PORT, () => {
  console.log(`[proxy] SensitiveAI moderation proxy running at http://127.0.0.1:${PROXY_PORT}`);
  console.log(`[proxy] Scanning POST ${POST_CREATE_PATH} -> AI ${AI_URL}${AI_ANALYZE_PATH}`);
  console.log(`[proxy] Forwarding to backend ${BACKEND_URL}`);
  console.log(`[proxy] Scanned fields: ${TEXT_FIELDS.join(', ')}`);
});
