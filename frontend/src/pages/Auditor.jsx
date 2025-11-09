import React, { useState } from "react";

const Auditor = () => {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showFacts, setShowFacts] = useState(false);

  // --- Top controls state (all hardcoded/fake) ---
  const [hfModel, setHfModel] = useState("meta-llama/Llama-3-8b-instruct");
  const [embeddingModel, setEmbeddingModel] = useState("text-embedding-3-small");
  const [chunkSize, setChunkSize] = useState(800);
  const [datasetFile, setDatasetFile] = useState(null);
  const [indexingMsg, setIndexingMsg] = useState("");

  // Fake "upload & index" (no backend call)
  const onUploadAndIndex = async () => {
    if (!datasetFile) {
      setIndexingMsg("Please choose a dataset file first.");
      return;
    }
    setIndexingMsg("Indexing…");
    await new Promise((r) => setTimeout(r, 800));
    setIndexingMsg(
      `Indexed "${datasetFile.name}" with chunkSize=${chunkSize}, embed=${embeddingModel}.`
    );
  };

  // ✅ Real API call for question → same behavior as earlier
  const onRun = async () => {
    const q = question.trim();
    if (!q) return;
    setLoading(true);
    setError("");
    setResult(null);
    setShowFacts(false);

    try {
      const res = await fetch("http://localhost:5000/api/auditor/audit/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Only send the question; the rest of the top controls are UI-only (fake)
        body: JSON.stringify({ question: q }),
      });

      const text = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);

      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error("Could not parse JSON from server.");
      }

      setResult(data);
    } catch (e) {
      setError(e.message || "Something went wrong while contacting the auditor.");
    } finally {
      setLoading(false);
    }
  };

  const isAnswer = result?.decision === "ANSWER";
  const isRefuse = result?.decision === "REFUSE";
  const gaugeWidth =
    result?.isr_score != null
      ? Math.min(100, Math.max(0, (Number(result.isr_score) / 3.5) * 100))
      : 0;

  return (
    <div className="min-h-screen w-full bg-black text-white relative overflow-hidden">
      {/* Floating bubbles - behind everything */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {[...Array(35)].map((_, i) => (
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

      <div className="relative z-10 mx-auto w-full max-w-6xl px-6 py-16">
        {/* Back button */}
        <button className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-12 text-sm cursor-pointer">
          <span>←</span>
          <span>Back to Dashboard</span>
        </button>

        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="processing-dot"></span>
            <span className="text-xs tracking-[0.3em] text-gray-500">ISR GATE</span>
          </div>
          <h1 className="text-6xl font-light mb-2 tracking-tight">Hallucination Auditor</h1>
          <p className="text-gray-400 text-lg">
            Confidence-gated answers using Information Sufficiency Ratio
          </p>
        </div>

        {/* Top control bar (all hardcoded/fake) */}
        <div className="card mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* HF model */}
            <div className="lg:col-span-2">
              <label className="block text-gray-400 text-xs mb-2">Hugging Face Model</label>
              <select
                value={hfModel}
                onChange={(e) => setHfModel(e.target.value)}
                className="w-full rounded-xl bg-white/5 border border-gray-800 px-3 py-2 outline-none focus:border-gray-600 transition-colors text-sm"
              >
                <option className="bg-black" value="meta-llama/Llama-3-8b-instruct">
                  meta-llama/Llama-3-8b-instruct
                </option>
                <option className="bg-black" value="mistralai/Mixtral-8x7B-Instruct-v0.1">
                  mistralai/Mixtral-8x7B-Instruct-v0.1
                </option>
                <option className="bg-black" value="google/gemma-7b-it">
                  google/gemma-7b-it
                </option>
                <option className="bg-black" value="HuggingFaceH4/zephyr-7b-beta">
                  HuggingFaceH4/zephyr-7b-beta
                </option>
              </select>
            </div>

            {/* Embedding model */}
            <div className="lg:col-span-2">
              <label className="block text-gray-400 text-xs mb-2">Embedding Model</label>
              <select
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
                className="w-full rounded-xl bg-white/5 border border-gray-800 px-3 py-2 outline-none focus:border-gray-600 transition-colors text-sm"
              >
                <option className="bg-black" value="text-embedding-3-small">
                  text-embedding-3-small
                </option>
                <option className="bg-black" value="text-embedding-3-large">
                  text-embedding-3-large
                </option>
                <option className="bg-black" value="bge-m3">
                  BAAI/bge-m3
                </option>
                <option className="bg-black" value="gte-large">
                  Alibaba-NLP/gte-large
                </option>
              </select>
            </div>

            {/* Chunk size */}
            <div>
              <label className="block text-gray-400 text-xs mb-2">Chunk Size</label>
              <input
                type="number"
                min={100}
                max={2000}
                step={50}
                value={chunkSize}
                onChange={(e) => setChunkSize(parseInt(e.target.value || "0", 10))}
                className="w-full rounded-xl bg-white/5 border border-gray-800 px-3 py-2 outline-none focus:border-gray-600 transition-colors text-sm"
              />
            </div>

            {/* Dataset upload */}
            <div className="md:col-span-2 lg:col-span-5">
              <label className="block text-gray-400 text-xs mb-2">
                Dataset (upload to vector DB)
              </label>
              <div className="flex items-center gap-2">
                <label className="flex-1 cursor-pointer">
                  <input
                    type="file"
                    onChange={(e) => setDatasetFile(e.target.files?.[0] || null)}
                    className="hidden"
                    id="file-upload"
                  />
                  <div className="w-full rounded-xl bg-white/5 border border-gray-800 px-3 py-2 text-sm hover:border-gray-600 transition-colors flex items-center justify-between min-h-[42px]">
                    <span className="text-gray-400 truncate">
                      {datasetFile ? datasetFile.name : "No file chosen"}
                    </span>
                    <span className="px-3 py-1 rounded-lg bg-white/10 text-white text-xs ml-2 shrink-0">
                      Choose File
                    </span>
                  </div>
                </label>
                <button
                  onClick={onUploadAndIndex}
                  className="px-4 py-2.5 rounded-xl text-xs font-medium bg-white text-black hover:bg-gray-200 transition-all whitespace-nowrap shrink-0"
                >
                  Upload & Index
                </button>
              </div>
              {indexingMsg && (
                <div className="mt-2 text-[11px] text-gray-400">{indexingMsg}</div>
              )}
            </div>
          </div>
        </div>

        {/* Question input */}
        <div className="card mb-6">
          <label className="block text-gray-400 text-sm mb-3">Question</label>
          <textarea
            className="w-full h-40 rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors resize-none"
            placeholder="Ask anything you want the Auditor to check…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />

          <div className="flex items-center justify-between mt-5">
            {error && <div className="text-sm text-red-400">{error}</div>}
            <div className="flex-1" />
            <button
              disabled={!question.trim() || loading}
              className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                question.trim() && !loading
                  ? "bg-white text-black hover:bg-gray-200 cursor-pointer"
                  : "bg-gray-800 text-gray-500 cursor-not-allowed"
              }`}
              onClick={onRun}
            >
              {loading ? "Running…" : "Run Auditor"}
            </button>
          </div>
        </div>

        {/* Results */}
        {result && (
          <div className="card space-y-6">
            {/* Status badge and toggle */}
            <div className="flex items-center justify-between">
              <span
                className={`px-3 py-1 text-xs rounded-full border font-medium ${
                  isAnswer
                    ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/10"
                    : isRefuse
                    ? "border-red-500/30 text-red-400 bg-red-500/10"
                    : "border-gray-700 text-gray-500"
                }`}
              >
                {isAnswer ? "ANSWER" : isRefuse ? "REFUSE" : "UNKNOWN"}
              </span>

              <button
                onClick={() => setShowFacts((s) => !s)}
                className="text-xs px-3 py-1.5 rounded-lg border border-gray-800 hover:border-gray-600 text-gray-400 hover:text-white transition-colors cursor-pointer"
              >
                {showFacts ? "Hide Facts" : "Show Facts"}
              </button>
            </div>

            {/* ISR Score */}
            <div>
              <div className="flex items-center justify-between mb-2 text-sm">
                <span className="text-gray-400">ISR Score</span>
                <span className="text-white font-mono">
                  {result.isr_score != null ? Number(result.isr_score).toFixed(3) : "—"}
                </span>
              </div>
              <div className="w-full h-1 rounded-full bg-gray-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    isAnswer
                      ? "bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]"
                      : isRefuse
                      ? "bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.5)]"
                      : "bg-gray-700"
                  }`}
                  style={{ width: `${gaugeWidth}%` }}
                />
              </div>
            </div>

            {/* Answer */}
            <div
              className={`p-5 rounded-xl border ${
                isAnswer
                  ? "bg-emerald-950/30 border-emerald-500/20 text-emerald-100"
                  : isRefuse
                  ? "bg-red-950/30 border-red-500/20 text-red-100"
                  : "bg-gray-900 border-gray-800 text-gray-500"
              }`}
            >
              {result.answer || "—"}
            </div>

            {/* Retrieved Facts (collapsible) */}
            <div
              className={`transition-all duration-300 overflow-hidden ${
                showFacts ? "max-h-[600px] opacity-100" : "max-h-0 opacity-0"
              }`}
            >
              <div className="pt-4 border-t border-gray-800">
                <div className="text-sm text-gray-400 mb-3">Retrieved Facts</div>
                <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-4 text-sm text-gray-400 whitespace-pre-wrap font-mono leading-relaxed">
                  {String(result.retrieved_context || "").trim() || "—"}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .bubble {
          position: absolute;
          bottom: -100px;
          background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.05));
          border-radius: 50%;
          animation: bubbleRise linear infinite;
          box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        }
        @keyframes bubbleRise {
          0% { bottom: -50px; opacity: 0; transform: translateX(0); }
          10% { opacity: 0.6; }
          50% { transform: translateX(20px); }
          90% { opacity: 0.6; }
          100% { bottom: 110vh; opacity: 0; transform: translateX(-15px); }
        }
        .card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          backdrop-filter: blur(12px);
          border-radius: 16px;
          padding: 24px;
          position: relative;
          z-index: 1;
        }
        .processing-dot {
          width: 8px; height: 8px; background: #fff; border-radius: 50%;
          animation: processingPulse 1.5s ease-in-out infinite;
          box-shadow: 0 0 15px rgba(255, 255, 255, 0.5);
        }
        @keyframes processingPulse {
          0%, 100% { transform: scale(1); opacity: 0.8; }
          50% { transform: scale(1.4); opacity: 0.3; }
        }
      `}</style>
    </div>
  );
};

export default Auditor;
