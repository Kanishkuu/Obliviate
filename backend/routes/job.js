// routes/jobs.js
import express from "express";
import Job from "../models/job.js";

const router = express.Router();

router.post("/jobs/save-inputs", async (req, res) => {
  try {
    let { userId, modelName, trainDataset, forgetDataset, maxRetainSamples } = req.body;
    if (!modelName || !trainDataset || !forgetDataset) {
      return res.status(400).json({ error: "Missing fields" });
    }
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Number(v)));
    maxRetainSamples = clamp(maxRetainSamples ?? 100, 100, 20000);

    const job = await Job.create({
      userId: userId || "demo-user",
      modelName,
      trainDataset,
      forgetDataset,
      maxRetainSamples,
      status: "READY",
      createdAt: new Date(),
    });

    res.json({ jobId: job._id, message: "✅ Inputs saved" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed" });
  }
});

router.get("/jobs/:id", async (req, res) => {
  try {
    const job = await Job.findById(req.params.id).lean();
    if (!job) return res.status(404).json({ error: "Not found" });
    res.json(job);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed" });
  }
});

export default router;
