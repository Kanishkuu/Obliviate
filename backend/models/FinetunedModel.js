import mongoose from "mongoose";

const FinetunedModelSchema = new mongoose.Schema(
  {
    tunedModelName: { type: String, unique: true, index: true },
    baseModel: { type: String, index: true },
    notes: { type: String },
  },
  { timestamps: true }
);

const FinetunedModel = mongoose.model("FinetunedModel", FinetunedModelSchema);
export default FinetunedModel;
