import express from "express";
import FinetunedModel from "../models/FinetunedModel.js";

const router = express.Router();

// ✅ GET all fine-tuned models
router.get("/finetuned", async (req, res) => {
  try {
    const models = await FinetunedModel.find({})
      .sort({ createdAt: -1 })   // newest first
      .lean();

    return res.json(models);
  } catch (err) {
    console.error("🚨 Error fetching finetuned models:", err);
    return res.status(500).json({ error: "Failed to fetch models" });
  }
});

export default router;
