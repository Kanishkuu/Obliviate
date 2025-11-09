// routes/unlearn.js
import express from "express";
import fetch from "node-fetch";
import FormData from "form-data";
import https from "https";
import Job from "../models/job.js";

const router = express.Router();

const UNLEARN_BASE = "https://unsnouted-camilla-painstakingly.ngrok-free.dev";

// A no-keepalive TLS agent helps avoid "bad record mac" with some ngrok tunnels
const noKeepAliveHttpsAgent = new https.Agent({ keepAlive: false });

async function postMultipart(url, form) {
  // Ensure Content-Length is set (some proxies/ngrok dislike chunked form bodies)
  const headers = form.getHeaders();
  const contentLength = await new Promise((resolve, reject) => {
    form.getLength((err, len) => (err ? reject(err) : resolve(len)));
  });
  headers["Content-Length"] = contentLength;

  return fetch(url, {
    method: "POST",
    body: form,
    headers,
    // Disable TLS keep-alive only for https
    agent: (parsedURL) =>
      parsedURL.protocol === "https:" ? noKeepAliveHttpsAgent : undefined,
  });
}

// POST /api/unlearn/start  { jobId }
router.post("/start", async (req, res) => {
  try {
    const { jobId } = req.body || {};
    if (!jobId) return res.status(400).json({ error: "jobId required" });

    const job = await Job.findById(jobId).lean();
    if (!job) return res.status(404).json({ error: "Job not found" });

    // Build multipart form expected by upstream
    const form = new FormData();
    form.append("model_name", job.modelName);
    form.append(
      "max_retain_samples",
      String(Math.max(100, Math.min(20000, Number(job.maxRetainSamples || 100))))
    );
    form.append("forget_set_uri", job.forgetDataset);
    // Upstream field maps to our trainDataset
    form.append("complete_dataset_uri", job.trainDataset);

    const base = UNLEARN_BASE.replace(/\/$/, "");
    const httpsUrl = `${base}/unlearn/`;

    let upstream;
    let firstErr;

    // Try HTTPS first
    try {
      upstream = await postMultipart(httpsUrl, form);
    } catch (e) {
      firstErr = e;
    }

    // If HTTPS call failed to even connect (TLS etc.), try HTTP fallback
    if (!upstream || (upstream.status === 502 && firstErr)) {
      const httpUrl = httpsUrl.replace(/^https:/, "http:");
      try {
        upstream = await postMultipart(httpUrl, form);
      } catch (e2) {
        // If fallback also fails, bubble up the original error + fallback info
        const msg = `HTTPS failed: ${String(firstErr || "")} | HTTP failed: ${String(e2 || "")}`;
        return res.status(502).json({
          error: "Upstream connection failed",
          details: msg,
          hint:
            "Likely an ngrok/TLS issue. Try refreshing your ngrok URL, ensure it’s reachable, or set UNLEARN_URL to a stable host.",
        });
      }
    }

    // Read upstream body
    const raw = await upstream.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      return res.status(upstream.status).json({
        error: "Invalid JSON from upstream",
        status: upstream.status,
        raw: raw?.slice(0, 1000),
      });
    }

    return res.status(upstream.status).json(data);
  } catch (e) {
    const msg = String(e?.message || e);
    const sslHint = /ssl|tls|record|mac|decryption|certificate/i.test(msg)
      ? "TLS error talking to upstream. Disable keep-alive (done), refresh ngrok URL, or use UNLEARN_URL=http://… as a test."
      : undefined;

    return res.status(500).json({
      error: "Failed to start unlearning",
      details: msg,
      hint: sslHint,
    });
  }
});

export default router;
