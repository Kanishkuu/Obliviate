import express from "express";
import fetch from "node-fetch";
import { Readable } from "stream";
import FinetunedModel from "../models/FinetunedModel.js";

const router = express.Router();
const FASTAPI_URL =
  process.env.FASTAPI_URL ||
  "https://diploblastic-clearheadedly-kelly.ngrok-free.dev";

/**
 * POST /api/finetune/start
 * Forwards payload to FastAPI.
 * If FastAPI returns image/* (FileResponse), we stream it back directly.
 * Otherwise we forward JSON.
 */
router.post("/start", async (req, res) => {
  try {
    const {
      base_model,
      dataset_url, // required
      output_dir = "outputs",
      learning_rate,
      num_train_epochs,
      gradient_accumulation_steps,
      warmup_steps,
      max_steps,
      lora_r,
      lora_alpha,
      lora_dropout,
      weight_decay,
      optimizer,
      lr_scheduler_type,
      random_state,
      logging_steps,
    } = req.body || {};

    if (!base_model) {
      return res.status(400).json({ error: "base_model is required" });
    }
    if (!dataset_url) {
      return res.status(400).json({ error: "dataset_url is required (gs://...)" });
    }
    if (!dataset_url.startsWith("gs://") && !/^https?:\/\//i.test(dataset_url)) {
      return res
        .status(400)
        .json({ error: "dataset_url must start with gs:// or http(s)://", received: dataset_url });
    }

    const payload = {
      base_model,
      dataset_url,
      output_dir,
      learning_rate,
      num_train_epochs,
      gradient_accumulation_steps,
      warmup_steps,
      max_steps,
      lora_r,
      lora_alpha,
      lora_dropout,
      weight_decay,
      optimizer,
      lr_scheduler_type,
      random_state,
      logging_steps,
    };

    // Call FastAPI (prefer /finetune; fallback /finetune/start)
    let upstream = await fetch(`${FASTAPI_URL}/finetune`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (upstream.status === 404) {
      upstream = await fetch(`${FASTAPI_URL}/finetune/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    const ctype = (upstream.headers.get("content-type") || "").toLowerCase();

    // --- 🔥 Robust image streaming (FileResponse) ---
    if (ctype.startsWith("image/")) {
      res.status(upstream.status);
      res.setHeader("Content-Type", ctype);
      res.setHeader("Cache-Control", "no-store");

      // node-fetch v3 returns a WHATWG ReadableStream; convert to Node stream and pipe
      if (upstream.body) {
        try {
          const nodeStream = Readable.fromWeb(upstream.body);
          nodeStream.on("error", (e) => {
            console.error("Proxy stream error:", e);
            if (!res.headersSent) res.status(500).end();
          });
          nodeStream.pipe(res);
        } catch (e) {
          // Fallback: buffer the whole body (small images ok)
          const ab = await upstream.arrayBuffer();
          res.end(Buffer.from(ab));
        }
      } else {
        // No body? End gracefully
        res.end();
      }
      return; // important
    }
    // --- end image streaming ---

    // Otherwise, forward JSON (or text fallback)
    const raw = await upstream.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      data = { raw };
    }

    if (!upstream.ok) {
      return res.status(502).json({ error: "FastAPI failed", details: data });
    }

    const tunedModelName =
      data.tunedModelName ||
      data.tuned_model_name ||
      `${base_model}-finetuned-${Date.now()}`;

    try {
      await FinetunedModel.create({ tunedModelName, baseModel: base_model });
    } catch (e) {
      console.warn("Failed to save FinetunedModel:", e.message);
    }

    return res.json({
      success: true,
      message: "Fine-tuning started",
      tunedModelName,
      fastapi: data,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Server error", details: err.message });
  }
});

export default router;
