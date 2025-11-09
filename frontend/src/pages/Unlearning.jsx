import React, { useState, useRef, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";

const Unlearning = () => {
  const navigate = useNavigate();

  const [fineModels, setFineModels] = useState([]);
  const [modelName, setModelName] = useState("");
  const [step1Locked, setStep1Locked] = useState(false);
  const canNext = modelName.trim().length > 0;

  const [trainFile, setTrainFile] = useState("");
  const [trainGcs, setTrainGcs] = useState("");
  const [uploadingTrain, setUploadingTrain] = useState(false);
  const trainRef = useRef(null);

  const [forgetFile, setForgetFile] = useState("");
  const [forgetGcs, setForgetGcs] = useState("");
  const [uploadingForget, setUploadingForget] = useState(false);
  const forgetRef = useRef(null);

  const [maxRetainSample, setMaxRetainSample] = useState(100);

  // NEW: explicit save states
  const [saving, setSaving] = useState(false);
  const [savedJob, setSavedJob] = useState(null);

  useEffect(() => {
    fetch("http://localhost:5000/api/models/finetuned")
      .then((r) => r.json())
      .then((data) => setFineModels(Array.isArray(data) ? data : []))
      .catch(() => setFineModels([]));
  }, []);

  const uploadToGCS = async (file, type) => {
    const res = await fetch("http://localhost:5000/api/assets/upload-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type, filename: file.name }),
    });

    let data;
    try {
      data = await res.json();
    } catch {
      const raw = await res.text();
      throw new Error(`Upload-URL responded with invalid JSON: ${raw}`);
    }

    if (!res.ok) throw new Error(data?.error || `Upload-URL failed`);

    const putRes = await fetch(data.uploadUrl, { method: "PUT", body: file });
    if (!putRes.ok) throw new Error(`Failed uploading to GCS`);

    const commitRes = await fetch("http://localhost:5000/api/assets/commit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ assetId: data.assetId, type, filename: file.name }),
    });

    let commit;
    try {
      commit = await commitRes.json();
    } catch {
      const raw = await commitRes.text();
      throw new Error(`Commit returned invalid JSON: ${raw}`);
    }
    if (!commitRes.ok) throw new Error(commit?.error || "Commit failed");

    return data.gcsPath; // "gs://bucket/path/file"
  };

  const onTrain = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setTrainFile(f.name);
    setUploadingTrain(true);
    try {
      const link = await uploadToGCS(f, "train");
      setTrainGcs(link);
      setSavedJob(null); // editing inputs => clear previous save banner
    } catch (err) {
      console.error(err);
      alert(err.message);
      setTrainFile("");
    } finally {
      setUploadingTrain(false);
    }
  };

  const onForget = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setForgetFile(f.name);
    setUploadingForget(true);
    try {
      const link = await uploadToGCS(f, "forget");
      setForgetGcs(link);
      setSavedJob(null);
    } catch (err) {
      console.error(err);
      alert(err.message);
      setForgetFile("");
    } finally {
      setUploadingForget(false);
    }
  };

  // NEW: manual save only when user clicks the button
  const onSave = async () => {
    if (!modelName.trim() || !trainGcs || !forgetGcs || maxRetainSample < 100) return;
    setSaving(true);
    setSavedJob(null);
    try {
      const res = await fetch("http://localhost:5000/api/jobs/save-inputs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          userId: "demo-user",
          modelName,
          trainDataset: trainGcs,
          forgetDataset: forgetGcs,
          maxRetainSamples: maxRetainSample, // plural to match schema
        }),
      });
      const data = await res.json();
      localStorage.setItem("currentJobId", data.jobId); 
      if (!res.ok) throw new Error(data?.error || "Failed");
      setSavedJob(data); // { jobId, message }
    } catch (e) {
      alert(e.message || "Failed saving inputs");
    } finally {
      setSaving(false);
    }
  };

  const currentStep = !step1Locked ? 1 : !trainGcs ? 2 : !forgetGcs ? 3 : 4;

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
        {/* Back */}
        <Link to="/" className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-12 text-sm">
          <span>←</span>
          <span>Back to Dashboard</span>
        </Link>

        {/* Header */}
        <div className="mb-12">
          <div className="flex items-center gap-3 mb-4">
            <span className="processing-dot"></span>
            <span className="text-xs tracking-[0.3em] text-gray-500">MACHINE UNLEARNING</span>
          </div>
          <h1 className="text-6xl font-light mb-4 tracking-tight">PrivacyPatch</h1>
          <p className="text-gray-400 text-lg">Selective unlearning control panel for T5 sentiment models</p>
        </div>

        {/* Tabs */}
        <div className="card mb-8 p-2">
          <div className="flex items-center gap-2">
            {[
              { key: "upload", label: "Upload", active: true },
              { key: "unlearn", label: "Unlearn" },
              { key: "jobs", label: "Jobs" },
            ].map((t) => (
              <button
                key={t.key}
                className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  t.active ? "bg-white text-black" : "text-gray-400 hover:text-white hover:bg-white/5"
                }`}
                onClick={() => !t.active && navigate(`/${t.key}`)}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Progress (4 steps) */}
        <div className="card mb-8">
          <div className="flex items-center justify-between">
            {[
              { num: 1, label: "Model",        done: step1Locked },
              { num: 2, label: "Train Dataset",done: !!trainGcs },
              { num: 3, label: "Forget Set",   done: !!forgetGcs },
              { num: 4, label: "Max Retain",   done: !!forgetGcs && maxRetainSample >= 100 },
            ].map((step, idx, arr) => (
              <React.Fragment key={step.num}>
                <div className="flex items-center gap-3">
                  <div
                    className={`w-8 h-8 rounded-full grid place-items-center text-sm transition-all ${
                      step.done
                        ? "bg-white text-black"
                        : currentStep === step.num
                        ? "border-2 border-white text-white"
                        : "border border-gray-700 text-gray-500"
                    }`}
                  >
                    {step.done ? "✓" : step.num}
                  </div>
                  <span className={`${step.done || currentStep === step.num ? "text-white" : "text-gray-500"} text-sm`}>
                    {step.label}
                  </span>
                </div>
                {idx < arr.length - 1 && (
                  <div className="flex-1 h-px mx-4 bg-gray-800 relative overflow-hidden">
                    {(step.done && currentStep > step.num) && <div className="absolute inset-0 bg-white progress-bar" />}
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Step 1: Model */}
        <div className={`card mb-6 transition-all duration-500 ${!step1Locked ? "opacity-100" : "opacity-50"}`}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-2 h-2 rounded-full bg-white opacity-50" />
            <span className="text-xs tracking-wider text-gray-500">STEP 1</span>
          </div>
          <h2 className="text-3xl font-light mb-2">Select Model</h2>
          <p className="text-gray-400 mb-6">Choose from fine-tuned models or enter HuggingFace Hub name</p>

          <div className="space-y-4">
            <div>
              <label className="block text-gray-400 text-sm mb-2">Fine-tuned Models</label>
              <select
                className="w-full rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors"
                value={modelName}
                onChange={(e) => { setModelName(e.target.value); setSavedJob(null); }}
                disabled={step1Locked}
              >
                <option value="">Select a model...</option>
                {fineModels.map((m) => (
                  <option key={m._id || m.tunedModelName} value={m.tunedModelName}>
                    {m.tunedModelName}
                  </option>
                ))}
              </select>
            </div>

            <div className="relative">
              <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-xs text-gray-600">OR</div>
              <div className="h-px bg-gray-800" />
            </div>

            <div>
              <label className="block text-gray-400 text-sm mb-2">Custom Model Name</label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => { setModelName(e.target.value); setSavedJob(null); }}
                placeholder="e.g., your-org/t5-imdb-finetuned"
                className="w-full rounded-xl bg-white/5 border border-gray-800 px-4 py-3 outline-none focus:border-gray-600 transition-colors"
                disabled={step1Locked}
              />
            </div>
          </div>

          <div className="flex justify-end mt-6">
            <button
              disabled={!canNext || step1Locked}
              className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                canNext && !step1Locked ? "bg-white text-black hover:bg-gray-200" : "bg-gray-800 text-gray-500 cursor-not-allowed"
              }`}
              onClick={() => setStep1Locked(true)}
            >
              Continue to Train Dataset
            </button>
          </div>
        </div>

        {/* Step 2: Train Dataset */}
        {step1Locked && (
          <div className={`card mb-6 transition-all duration-500 ${!trainGcs ? "opacity-100" : "opacity-50"}`}>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-white opacity-50" />
              <span className="text-xs tracking-wider text-gray-500">STEP 2</span>
            </div>
            <h2 className="text-3xl font-light mb-2">Train Dataset</h2>
            <p className="text-gray-400 mb-6">Upload training dataset (CSV / JSON / JSONL)</p>

            <input ref={trainRef} className="hidden" type="file" accept=".csv,.json,.jsonl" onChange={onTrain} disabled={!!trainGcs} />

            <div className="flex items-center gap-4">
              <button
                onClick={() => trainRef.current?.click()}
                disabled={!!trainGcs}
                className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  !trainGcs ? "bg-white/10 border border-gray-700 hover:border-gray-600 text-white" : "bg-gray-800 text-gray-500 cursor-not-allowed"
                }`}
              >
                Choose File
              </button>

              <div className="text-sm text-gray-400">
                {uploadingTrain ? (
                  <span className="flex items-center gap-2"><span className="uploading-spinner" /> Uploading...</span>
                ) : trainGcs ? (
                  <span className="text-emerald-400">✓ {trainFile}</span>
                ) : trainFile ? (
                  trainFile
                ) : (
                  "No file selected"
                )}
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Forget Set */}
        {trainGcs && (
          <div className={`card mb-6 transition-all duration-500 ${!forgetGcs ? "opacity-100" : "opacity-50"}`}>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-white opacity-50" />
              <span className="text-xs tracking-wider text-gray-500">STEP 3</span>
            </div>
            <h2 className="text-3xl font-light mb-2">Forget Set</h2>
            <p className="text-gray-400 mb-6">Upload forget dataset (CSV / JSON / JSONL)</p>

            <input ref={forgetRef} className="hidden" type="file" accept=".csv,.json,.jsonl" onChange={onForget} disabled={!!forgetGcs} />

            <div className="flex items-center gap-4">
              <button
                onClick={() => forgetRef.current?.click()}
                disabled={!!forgetGcs}
                className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  !forgetGcs ? "bg-white/10 border border-gray-700 hover:border-gray-600 text-white" : "bg-gray-800 text-gray-500 cursor-not-allowed"
                }`}
              >
                Choose File
              </button>

              <div className="text-sm text-gray-400">
                {uploadingForget ? (
                  <span className="flex items-center gap-2"><span className="uploading-spinner" /> Uploading...</span>
                ) : forgetGcs ? (
                  <span className="text-emerald-400">✓ {forgetFile}</span>
                ) : forgetFile ? (
                  forgetFile
                ) : (
                  "No file selected"
                )}
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Max Retain Samples */}
        {forgetGcs && (
          <div className="card mb-6 transition-all duration-500 opacity-100">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-2 h-2 rounded-full bg-white opacity-50" />
              <span className="text-xs tracking-wider text-gray-500">STEP 4</span>
            </div>
            <h2 className="text-3xl font-light mb-2">Max Retain Samples</h2>
            <p className="text-gray-400 mb-6">Set maximum number of samples to retain during unlearning</p>

            <div>
              <div className="flex justify-between text-sm mb-3">
                <span className="text-gray-400">Sample Count</span>
                <span className="font-mono text-white">{maxRetainSample.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min="100"
                max="20000"
                step="100"
                value={maxRetainSample}
                onChange={(e) => { setMaxRetainSample(parseInt(e.target.value)); setSavedJob(null); }}
                className="slider"
              />
              <div className="flex justify-between text-xs text-gray-600 mt-2">
                <span>100</span><span>20,000</span>
              </div>
            </div>
          </div>
        )}

        {/* Save button (only when all inputs ready) */}
        {modelName.trim() && trainGcs && forgetGcs && (
          <div className="flex justify-end mb-6">
            <button
              onClick={onSave}
              disabled={saving}
              className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                !saving ? "bg-white text-black hover:bg-gray-200" : "bg-gray-800 text-gray-500 cursor-not-allowed"
              }`}
            >
              {saving ? "Saving…" : "Save Inputs"}
            </button>
          </div>
        )}

        {/* Success message (after manual save) */}
        {savedJob && (
          <div className="card border-emerald-500/20 bg-emerald-950/30">
            <div className="flex items-center gap-3 text-emerald-400">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-sm">
                {savedJob.message || "Saved"} — Job ID: <span className="text-white/90">{savedJob.jobId}</span> — Max Retain: {maxRetainSample.toLocaleString()}
              </span>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .bubbles-container { position:absolute; width:100%; height:100%; overflow:hidden; pointer-events:none; }
        .bubble { position:absolute; bottom:-100px; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.15), rgba(255,255,255,0.05)); border-radius:50%; animation:bubbleRise linear infinite; box-shadow:0 0 10px rgba(255,255,255,0.1); }
        @keyframes bubbleRise { 0% { bottom:-50px; opacity:0; transform:translateX(0); } 10% { opacity:0.6; } 50% { transform:translateX(20px); } 90% { opacity:0.6; } 100% { bottom:110vh; opacity:0; transform:translateX(-15px); } }
        .card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); backdrop-filter: blur(12px); border-radius: 16px; padding: 24px; }
        .processing-dot { width:8px; height:8px; background:#fff; border-radius:50%; animation: processingPulse 1.5s ease-in-out infinite; box-shadow:0 0 15px rgba(255,255,255,0.5); }
        @keyframes processingPulse { 0%, 100% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.4); opacity: 0.3; } }
        .progress-bar { animation: progressSlide 0.8s ease-out; }
        @keyframes progressSlide { from { transform: translateX(-100%); } to { transform: translateX(0); } }
        .uploading-spinner { display:inline-block; width:12px; height:12px; border:2px solid rgba(255,255,255,0.2); border-top-color:white; border-radius:50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .slider { -webkit-appearance: none; width: 100%; height: 4px; border-radius: 10px; background: rgba(255,255,255,0.2); cursor: pointer; transition: background 0.3s ease; outline: none; }
        .slider:hover { background: rgba(255,255,255,0.3); }
        .slider::-webkit-slider-thumb { -webkit-appearance: none; height: 18px; width: 18px; background: white; border-radius: 50%; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 0 15px rgba(255, 255, 255, 0.7); }
        .slider::-webkit-slider-thumb:hover { transform: scale(1.3); box-shadow: 0 0 25px rgba(255, 255, 255, 0.9); }
        .slider::-moz-range-thumb { height: 18px; width: 18px; background: white; border: none; border-radius: 50%; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 0 15px rgba(255, 255, 255, 0.7); }
        .slider::-moz-range-thumb:hover { transform: scale(1.3); box-shadow: 0 0 25px rgba(255, 255, 255, 0.9); }
        .slider::-moz-range-track { background: rgba(255,255,255,0.2); border-radius: 10px; height: 4px; }
      `}</style>
    </div>
  );
};

export default Unlearning;
