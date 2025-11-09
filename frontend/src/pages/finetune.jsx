import React, { useState } from "react";

const Finetune = () => {
  const [baseModel, setBaseModel] = useState("");
  const [datasetUrl, setDatasetUrl] = useState(""); // expects gs://...

  // Hyperparameters
  const [numTrainEpochs, setNumTrainEpochs] = useState(10);
  const [maxSteps, setMaxSteps] = useState(60);
  const [gradientAccum, setGradientAccum] = useState(4);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [warmupSteps, setWarmupSteps] = useState(5);
  const [loraR, setLoraR] = useState(16);
  const [loraAlpha, setLoraAlpha] = useState(16);
  const [loraDropout, setLoraDropout] = useState(0);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [optimizer, setOptimizer] = useState("adamw_8bit");
  const [lrSchedulerType, setLrSchedulerType] = useState("linear");
  const [randomState, setRandomState] = useState(3407);
  const [loggingSteps, setLoggingSteps] = useState(1);

  const OUTPUT_DIR = "outputs";
  const [submitted, setSubmitted] = useState(false);

  // For image returned directly by /start
  const [previewUrl, setPreviewUrl] = useState(""); // blob URL if we get an image
  const [errorMsg, setErrorMsg] = useState("");
  const [tunedModelName, setTunedModelName] = useState(""); // if JSON response includes it

  const datasetLooksLikeGcs = datasetUrl.trim().toLowerCase().startsWith("gs://");
  const canStart = baseModel.trim().length > 0 && datasetLooksLikeGcs;

  const startTraining = async () => {
    setErrorMsg("");
    setPreviewUrl("");
    setTunedModelName("");

    const safeDatasetUrl = encodeURI(datasetUrl.trim()); // encodes spaces/()

    const payload = {
      base_model: baseModel,
      dataset_url: safeDatasetUrl,
      output_dir: OUTPUT_DIR,
      learning_rate: learningRate,
      num_train_epochs: numTrainEpochs,
      gradient_accumulation_steps: gradientAccum,
      warmup_steps: warmupSteps,
      max_steps: maxSteps,
      lora_r: loraR,
      lora_alpha: loraAlpha,
      lora_dropout: loraDropout,
      weight_decay: weightDecay,
      optimizer,
      lr_scheduler_type: lrSchedulerType,
      random_state: randomState,
      logging_steps: loggingSteps,
    };

    try {
      const res = await fetch("http://localhost:5000/api/finetune/start", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

const ct = (res.headers.get("content-type") || "").toLowerCase();
console.log("Response content-type:", ct);

if (ct.startsWith("image/")) {
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  setPreviewUrl(url);
  setSubmitted(true);
  return;
}

const body = ct.includes("application/json") ? await res.json() : await res.text();
console.log("Response body:", body);

if (!res.ok) {
  const msg = typeof body === "string" ? body : JSON.stringify(body);
  throw new Error(msg || "Failed to start fine-tuning");
}

setSubmitted(true);
if (typeof body === "object" && body?.tunedModelName) {
  setTunedModelName(body.tunedModelName);
}

    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Failed to start fine-tuning");
      alert(err.message || "Failed to start fine-tuning");
    }
  };

  return (
    <div className="min-h-screen w-full bg-black text-white relative overflow-hidden">
      {/* Floating bubbles */}
      <div className="bubbles-container">
        {[...Array(30)].map((_, i) => (
          <div
            key={i}
            className="bubble"
            style={{
              left: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 20}s`,
              animationDuration: `${10 + Math.random() * 15}s`,
              width: `${2 + Math.random() * 5}px`,
              height: `${2 + Math.random() * 5}px`,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 mx-auto w-full max-w-4xl px-6 py-16">
        {/* Back button */}
        <button
          onClick={() => window.history.back()}
          className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-12 text-sm"
        >
          <span>←</span>
          <span>Back to Dashboard</span>
        </button>

        {/* Header */}
        <div className="mb-16">
          <div className="flex items-center gap-3 mb-4">
            <span className="processing-dot"></span>
            <span className="text-xs tracking-[0.3em] text-gray-500">MODEL TRAINING</span>
          </div>
          <h1 className="text-6xl font-light mb-4 tracking-tight">ModelForge</h1>
          <p className="text-gray-400 text-lg">Fine-tune and register models into PrivacyPatch</p>
        </div>

        {/* Main card */}
        <div className="card space-y-8">
          {/* Base model */}
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-white opacity-50" />
              <span className="text-xs tracking-wider text-gray-500">BASE MODEL</span>
            </div>
            <label className="block text-gray-400 text-sm mb-2">Hugging Face Model Name</label>
            <input
              type="text"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              placeholder="e.g., facebook/opt-125m"
              className="w-full rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors"
            />
          </div>

          {/* GCS dataset input */}
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-white opacity-50" />
              <span className="text-xs tracking-wider text-gray-500">TRAINING DATASET (GCS FILE)</span>
            </div>

            <label className="block text-gray-400 text-sm mb-2">GCS Path (gs://...)</label>
            <input
              type="text"
              value={datasetUrl}
              onChange={(e) => setDatasetUrl(e.target.value)}
              placeholder="gs://hack36/users/demo-user-001/datasets/04183acc2296719a63de5658/sample (1).json"
              className="w-full rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors"
            />

            {!datasetLooksLikeGcs && datasetUrl.trim().length > 0 && (
              <p className="text-xs text-red-400 mt-2">
                The dataset must start with <code>gs://</code>.
              </p>
            )}
            <p className="text-xs text-gray-500 mt-2">
              Spaces and parentheses will be encoded automatically.
            </p>
          </div>

          {/* Hyperparameters */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-2 h-2 rounded-full bg-white opacity-50" />
              <span className="text-xs tracking-wider text-gray-500">HYPERPARAMETERS</span>
            </div>

            <div className="space-y-6">
              {/* num_train_epochs */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">Epochs</span>
                  <span className="font-mono text-white">{numTrainEpochs}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={numTrainEpochs}
                  onChange={(e) => setNumTrainEpochs(parseInt(e.target.value))}
                  className="slider"
                />
              </div>

              {/* max_steps */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">Max Steps</span>
                  <span className="font-mono text-white">{maxSteps}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="2000"
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value))}
                  className="slider"
                />
              </div>

              {/* gradient_accumulation_steps */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">Gradient Accumulation</span>
                  <span className="font-mono text-white">{gradientAccum}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="32"
                  value={gradientAccum}
                  onChange={(e) => setGradientAccum(parseInt(e.target.value))}
                  className="slider"
                />
              </div>

              {/* learning_rate */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">Learning Rate</span>
                  <span className="font-mono text-white">{learningRate}</span>
                </div>
                <input
                  type="range"
                  min="0.00001"
                  max="0.001"
                  step="0.00001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="slider"
                />
              </div>

              {/* warmup_steps */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">Warmup Steps</span>
                  <span className="font-mono text-white">{warmupSteps}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="500"
                  value={warmupSteps}
                  onChange={(e) => setWarmupSteps(parseInt(e.target.value))}
                  className="slider"
                />
              </div>

              {/* lora_r */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">LoRA r</span>
                  <span className="font-mono text-white">{loraR}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="128"
                  value={loraR}
                  onChange={(e) => setLoraR(parseInt(e.target.value))}
                  className="slider"
                />
              </div>

              {/* lora_alpha */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">LoRA α</span>
                  <span className="font-mono text-white">{loraAlpha}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="256"
                  value={loraAlpha}
                  onChange={(e) => setLoraAlpha(parseInt(e.target.value))}
                  className="slider"
                />
              </div>

              {/* lora_dropout */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">LoRA Dropout</span>
                  <span className="font-mono text-white">{loraDropout}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.01"
                  value={loraDropout}
                  onChange={(e) => setLoraDropout(parseFloat(e.target.value))}
                  className="slider"
                />
              </div>

              {/* weight_decay */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">Weight Decay</span>
                  <span className="font-mono text-white">{weightDecay}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.2"
                  step="0.001"
                  value={weightDecay}
                  onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
                  className="slider"
                />
              </div>

              {/* optimizer */}
              <div>
                <label className="block text-gray-400 text-sm mb-2">Optimizer</label>
                <select
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value)}
                  className="w-full rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors"
                >
                  <option value="adamw_8bit" className="bg-black">adamw_8bit</option>
                  <option value="adamw_torch" className="bg-black">adamw_torch</option>
                  <option value="adamw" className="bg-black">adamw</option>
                </select>
              </div>

              {/* lr_scheduler_type */}
              <div>
                <label className="block text-gray-400 text-sm mb-2">LR Scheduler</label>
                <select
                  value={lrSchedulerType}
                  onChange={(e) => setLrSchedulerType(e.target.value)}
                  className="w-full rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors"
                >
                  <option value="linear" className="bg-black">linear</option>
                  <option value="cosine" className="bg-black">cosine</option>
                  <option value="constant" className="bg-black">constant</option>
                  <option value="cosine_with_restarts" className="bg-black">cosine_with_restarts</option>
                  <option value="polynomial" className="bg-black">polynomial</option>
                </select>
              </div>

              {/* random_state */}
              <div>
                <label className="block text-gray-400 text-sm mb-2">Random State</label>
                <input
                  type="number"
                  value={randomState}
                  onChange={(e) => setRandomState(parseInt(e.target.value || "0", 10))}
                  className="w-full rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors"
                />
              </div>

              {/* logging_steps */}
              <div>
                <div className="flex justify-between text-sm mb-3">
                  <span className="text-gray-400">Logging Steps</span>
                  <span className="font-mono text-white">{loggingSteps}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={loggingSteps}
                  onChange={(e) => setLoggingSteps(parseInt(e.target.value))}
                  className="slider"
                />
              </div>
            </div>
          </div>

          {/* CTA */}
          <div className="flex justify-end pt-4">
            <button
              disabled={!canStart}
              onClick={startTraining}
              className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                canStart
                  ? "bg-white text-black hover:bg-gray-200"
                  : "bg-gray-800 text-gray-500 cursor-not-allowed"
              }`}
            >
              Start Fine-Tuning
            </button>
          </div>
        </div>

        {/* Result card */}
        {submitted && (
          <div className="card mt-6">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-sm text-emerald-400">
                Fine-tuning started! {tunedModelName ? `Job: ${tunedModelName}` : ""}
              </span>
            </div>

            {/* If server returned an image directly, show it */}
            {previewUrl && (
              <div className="mt-4">
                <div className="text-xs text-gray-500 mb-2">Loss curve (from FileResponse):</div>
                <img
                  src={previewUrl}
                  alt="Loss graph"
                  className="rounded-xl border border-gray-800 max-w-full"
                />
              </div>
            )}

            {errorMsg && (
              <div className="mt-3 text-xs text-red-400">Error: {errorMsg}</div>
            )}
          </div>
        )}
      </div>

      <style>{`
        .bubbles-container { position: absolute; width: 100%; height: 100%; overflow: hidden; pointer-events: none; }
        .bubble { position: absolute; bottom: -100px; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.15), rgba(255,255,255,0.05)); border-radius: 50%; animation: bubbleRise linear infinite; box-shadow: 0 0 10px rgba(255,255,255,0.1); }
        @keyframes bubbleRise { 0% { bottom: -50px; opacity: 0; transform: translateX(0); } 10% { opacity: 0.6; } 50% { transform: translateX(20px); } 90% { opacity: 0.6; } 100% { bottom: 110vh; opacity: 0; transform: translateX(-15px); } }
        .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(12px); border-radius: 16px; padding: 24px; }
        .processing-dot { width: 8px; height: 8px; background: #fff; border-radius: 50%; animation: processingPulse 1.5s ease-in-out infinite; box-shadow: 0 0 15px rgba(255,255,255,0.5); }
        @keyframes processingPulse { 0%, 100% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.4); opacity: 0.3; } }
        .slider { -webkit-appearance: none; width: 100%; height: 4px; border-radius: 10px; background: rgba(255,255,255,0.1); cursor: pointer; transition: background 0.3s ease; }
        .slider:hover { background: rgba(255,255,255,0.15); }
        .slider::-webkit-slider-thumb { -webkit-appearance: none; height: 16px; width: 16px; background: white; border-radius: 50%; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 0 10px rgba(255,255,255,0.5); }
        .slider::-webkit-slider-thumb:hover { transform: scale(1.2); box-shadow: 0 0 20px rgba(255,255,255,0.8); }
        .slider::-moz-range-thumb { height: 16px; width: 16px; background: white; border-radius: 50%; border: none; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 0 10px rgba(255,255,255,0.5); }
        .slider::-moz-range-thumb:hover { transform: scale(1.2); box-shadow: 0 0 20px rgba(255,255,255,0.8); }
      `}</style>
    </div>
  );
};

export default Finetune;
