import mongoose from "mongoose";

const JobSchema = new mongoose.Schema({
  userId: String,
  modelName: String,
  trainDataset: String,
  forgetDataset: String,
  maxRetainSamples: Number,
  createdAt: Date,
  status: String
});

export default mongoose.model("Job", JobSchema);
