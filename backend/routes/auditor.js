// server/routes/auditor.js
import express from "express";
// If Node <18, uncomment next line and use fetch from node-fetch
// import fetch from "node-fetch";

const router = express.Router();

const UPSTREAM = process.env.AUDIT_ENDPOINT

// POST /api/auditor/audit/answer
router.post("/audit/answer", async (req, res) => {
  try {
    const { question } = req.body || {};
    if (!question || !String(question).trim()) {
      return res.status(400).json({ error: "Missing 'question'." });
    }

    const upstreamRes = await fetch(UPSTREAM, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const text = await upstreamRes.text();
    // pass through status/content-type as-is
    res
      .status(upstreamRes.status)
      .type(upstreamRes.headers.get("content-type") || "application/json")
      .send(text);
  } catch (err) {
    console.error("Proxy error:", err);
    res.status(502).json({ error: "Upstream call failed." });
  }
});

export default router;
