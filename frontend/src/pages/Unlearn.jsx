import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const API = "http://localhost:5000/api";

const Unlearn = () => {
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState(null);

  const [starting, setStarting] = useState(false);
  const [err, setErr] = useState("");

  // Upstream response + artifacts we fetch from GCS
  const [resp, setResp] = useState(null);
  const [plotUrl, setPlotUrl] = useState("");
  const [csvRows, setCsvRows] = useState([]);
  const [modelFiles, setModelFiles] = useState([]);

  useEffect(() => {
    const id = localStorage.getItem("currentJobId") || "";
    setJobId(id);
    if (id) {
      fetch(`${API}/jobs/${encodeURIComponent(id)}`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then(setJob)
        .catch((e) => {
          console.error("Failed to load job:", e);
        });
    }
  }, []);

  const startUnlearning = async () => {
    setErr("");
    setResp(null);
    setPlotUrl("");
    setCsvRows([]);
    setModelFiles([]);

    if (!jobId) {
      setErr("No jobId found. Save inputs first on the Upload screen.");
      return;
    }

    setStarting(true);
    try {
      // Kick off unlearning
      const r = await fetch(`${API}/unlearn/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId }),
      });
      const text = await r.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(`Invalid JSON from server: ${text?.slice(0, 400) || "(empty)"}`);
      }
      if (!r.ok) throw new Error(data?.error || "Failed to start unlearning");

      setResp(data);

      // Resolve artifacts
      const plot = data?.artifacts?.comparison_plot_uri;       // gs://...png
      const csv  = data?.artifacts?.summary_table_uri;         // gs://...csv
      const modelPrefix = data?.artifacts?.unlearned_model_uri; // gs://.../unlearned_model/

      // Sign the plot for <img>
      if (plot) {
        const s = await fetch(`${API}/assets/sign?uri=${encodeURIComponent(plot)}&expires=300`);
        const sj = await s.json();
        if (sj?.url) setPlotUrl(sj.url);
      }

      // CSV preview
      if (csv) {
        const cj = await fetch(`${API}/assets/csv?uri=${encodeURIComponent(csv)}&limit=300`);
        const cjJson = await cj.json();
        setCsvRows(Array.isArray(cjJson?.rows) ? cjJson.rows : []);
      }

      // List model folder
      if (modelPrefix) {
        const lj = await fetch(`${API}/assets/list?prefix=${encodeURIComponent(modelPrefix)}`);
        const ljJson = await lj.json();
        const items = Array.isArray(ljJson?.items) ? ljJson.items : [];
        setModelFiles(items.map((it) => ({ ...it, __bucket: ljJson.bucket })));
      }
    } catch (e) {
      console.error(e);
      setErr(e.message || "Request failed");
    } finally {
      setStarting(false);
    }
  };

  const fmt = (v) => (v == null ? "—" : typeof v === "number" ? v.toFixed(3) : String(v));
  const mO = resp?.metrics?.Original;
  const mU = resp?.metrics?.Unlearned;

  const copy = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {}
  };

  const MetricRow = ({ name, o, u }) => {
    const lowerIsBetter = /loss|entropy/i.test(name);
    const d = o == null || u == null ? null : u - o;
    const good = d == null ? null : lowerIsBetter ? d < 0 : d > 0;
    return (
      <div className="grid grid-cols-12 gap-3 items-center py-2 text-sm">
        <div className="col-span-4 text-white/80">{name}</div>
        <div className="col-span-3 font-mono">{fmt(o)}</div>
        <div className="col-span-3 font-mono">{fmt(u)}</div>
        <div className="col-span-2">
          {d == null ? (
            <span className="text-white/50">—</span>
          ) : (
            <span
              className={`px-2 py-0.5 rounded-md text-xs ${
                good
                  ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-400/30"
                  : "bg-red-500/15 text-red-300 ring-1 ring-red-400/30"
              }`}
              title={lowerIsBetter ? "Lower is better" : "Higher is better"}
            >
              Δ {fmt(d)} {lowerIsBetter ? "(↓ good)" : "(↑ good)"}
            </span>
          )}
        </div>
      </div>
    );
  };

  const interp = resp?.interpretation || {};
  const improvements = resp?.improvements || {};
  const artifacts = resp?.artifacts || {};

  return (
    <div className="min-h-screen w-full bg-black text-white relative overflow-hidden">
      {/* Floating bubbles */}
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
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-12 text-sm"
        >
          <span>←</span>
          <span>Back to Dashboard</span>
        </Link>

        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="processing-dot"></span>
            <span className="text-xs tracking-[0.3em] text-gray-500">PRIVACYPATCH</span>
          </div>
          <h1 className="text-6xl font-light mb-2 tracking-tight">Unlearn</h1>
          <p className="text-gray-400 text-lg">Run unlearning and see metrics & artifacts here</p>
        </div>

        {/* Summary + Start */}
        <div className="card space-y-4 mb-8">
          <h2 className="text-xl font-bold text-white mb-4">Job Configuration</h2>
          <div className="text-sm text-gray-300">
            <div className="mb-3">
              <span className="text-gray-500 font-semibold">Job ID:</span>
              <span className="font-mono ml-2 font-semibold">{jobId || "—"}</span>
            </div>
            {job && (
              <div className="space-y-1.5 text-xs text-gray-400 pl-4 border-l-2 border-gray-800">
                <div>
                  <span className="text-gray-500 font-semibold">Model:</span>{" "}
                  <span className="font-mono text-gray-300 font-medium">{job.modelName}</span>
                </div>
                <div>
                  <span className="text-gray-500 font-semibold">Forget:</span>{" "}
                  <span className="break-all font-medium">{job.forgetDataset}</span>
                </div>
                <div>
                  <span className="text-gray-500 font-semibold">Train:</span>{" "}
                  <span className="break-all font-medium">{job.trainDataset}</span>
                </div>
                <div>
                  <span className="text-gray-500 font-semibold">Max Retain:</span>{" "}
                  <span className="font-mono font-medium">{job.maxRetainSamples}</span>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-gray-800">
            {err ? <div className="text-sm text-red-400">{err}</div> : <div />}
            <button
              onClick={startUnlearning}
              disabled={starting || !jobId}
              className={`px-6 py-2.5 rounded-xl text-sm font-medium transition-all ${
                starting || !jobId
                  ? "bg-gray-800 text-gray-500 cursor-not-allowed"
                  : "bg-white text-black hover:bg-gray-200 cursor-pointer"
              }`}
            >
              {starting ? "Starting…" : "Start Unlearning"}
            </button>
          </div>
        </div>

        {/* Response header */}
        {resp && (
          <div className="card mb-8 flex items-center justify-between">
            <div className="text-sm">
              <div className="text-gray-500 text-xs mb-1">Request ID</div>
              <div className="font-mono text-gray-200">{resp.request_id || "—"}</div>
            </div>
            <div>
              <span
                className={`px-3 py-1 text-xs rounded-full border font-medium ${
                  (resp.status || "").toLowerCase() === "success"
                    ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/10"
                    : "border-gray-700 text-gray-400 bg-gray-800/50"
                }`}
              >
                {(resp.status || "unknown").toUpperCase()}
              </span>
            </div>
          </div>
        )}

        {/* Interpretation */}
        {resp && (
          <div className="card space-y-4 mb-8">
            <h2 className="text-xl font-semibold text-white">Interpretation</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              {Object.entries(resp?.interpretation || {}).map(([key, val]) => (
                <div key={key} className="p-3 rounded-xl bg-white/5 border border-gray-800">
                  <div className="text-gray-500 text-xs mb-1">{key}</div>
                  <div className="font-mono text-gray-200">{val || "—"}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Improvements */}
        {resp && (
          <div className="card space-y-4 mb-8">
            <h2 className="text-xl font-semibold text-white">Improvements (Unlearned − Original)</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              {[
                ["retain_accuracy_change", resp?.improvements?.retain_accuracy_change, false],
                ["forget_accuracy_change", resp?.improvements?.forget_accuracy_change, true],
                ["forget_mlm_loss_change", resp?.improvements?.forget_mlm_loss_change, true],
                ["forget_entropy_change", resp?.improvements?.forget_entropy_change, true],
              ].map(([k, v, lowerBetter]) => {
                const good = v == null ? null : lowerBetter ? v < 0 : v > 0;
                return (
                  <div
                    key={k}
                    className="p-3 rounded-xl bg-white/5 border border-gray-800 flex items-center justify-between"
                  >
                    <div className="text-gray-300">{k}</div>
                    <span
                      className={`ml-3 px-2 py-0.5 rounded-md text-xs font-mono ${
                        v == null
                          ? "bg-gray-800 text-gray-500"
                          : good
                          ? "bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-400/30"
                          : "bg-red-500/15 text-red-300 ring-1 ring-red-400/30"
                      }`}
                    >
                      {fmt(v)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Metrics */}
        {resp && (mO || mU) && (
          <div className="card space-y-6 mb-8">
            <h2 className="text-xl font-semibold text-white">Metrics Comparison</h2>
            <div className="grid grid-cols-12 gap-3 text-xs text-gray-500">
              <div className="col-span-4">Metric</div>
              <div className="col-span-3">Original</div>
              <div className="col-span-3">Unlearned</div>
              <div className="col-span-2">Δ (gain)</div>
            </div>
            <div className="h-px bg-gray-800" />
            <MetricRow name="retain_accuracy" o={mO?.retain_accuracy} u={mU?.retain_accuracy} />
            <MetricRow name="forget_accuracy" o={mO?.forget_accuracy} u={mU?.forget_accuracy} />
            <MetricRow name="forget_mlm_loss" o={mO?.forget_mlm_loss} u={mU?.forget_mlm_loss} />
            <MetricRow name="forget_entropy" o={mO?.forget_entropy} u={mU?.forget_entropy} />
          </div>
        )}

        {/* Comparison Plot */}
        {plotUrl && (
          <div className="card space-y-4 mb-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Comparison Plot</h2>
              {artifacts?.comparison_plot_uri && (
                <a
                  href={plotUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-gray-400 hover:text-white transition-colors"
                >
                  Open original →
                </a>
              )}
            </div>
            <div className="rounded-xl overflow-hidden border border-gray-800 bg-white/5">
              <img src={plotUrl} alt="Unlearning comparison" className="w-full h-auto" />
            </div>
          </div>
        )}

        {/* Summary CSV preview */}
        {csvRows.length > 0 && (
          <div className="card space-y-4 mb-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Summary Table</h2>
              {artifacts?.summary_table_uri && (
                <a
                  href={`${API}/assets/stream?uri=${encodeURIComponent(artifacts.summary_table_uri)}`}
                  className="text-xs text-gray-400 hover:text-white transition-colors"
                >
                  Download CSV →
                </a>
              )}
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="text-gray-500">
                  <tr>
                    {Object.keys(csvRows[0]).map((h) => (
                      <th key={h} className="text-left px-3 py-2 border-b border-gray-800">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {csvRows.map((row, i) => (
                    <tr key={i} className="hover:bg-white/5 transition-colors">
                      {Object.keys(csvRows[0]).map((h) => (
                        <td key={h} className="px-3 py-2 border-b border-gray-800/50 text-gray-300">
                          {String(row[h])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="text-xs text-gray-500 mt-2">Showing {csvRows.length} rows.</div>
            </div>
          </div>
        )}

        {/* Artifacts quick links */}
        {resp && (
          <div className="card space-y-4 mb-8">
            <h2 className="text-xl font-semibold text-white">Artifacts</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              {[
                ["Unlearned model (prefix)", artifacts.unlearned_model_uri],
                ["Comparison plot (gs://)", artifacts.comparison_plot_uri],
                ["Summary table (gs://)", artifacts.summary_table_uri],
                ["Metrics JSON (gs://)", artifacts.metrics_uri],
              ].map(([label, uri]) => (
                <div
                  key={label}
                  className="p-3 rounded-xl bg-white/5 border border-gray-800 flex items-center justify-between gap-3"
                >
                  <div className="truncate">
                    <div className="text-gray-500 text-xs mb-1">{label}</div>
                    <div className="font-mono text-gray-300 text-xs truncate">{uri || "—"}</div>
                  </div>
                  {uri && (
                    <button
                      onClick={() => copy(uri)}
                      className="shrink-0 px-3 py-1.5 rounded-lg text-xs bg-white/10 hover:bg-white/20 transition-colors"
                      title="Copy gs:// to clipboard"
                    >
                      Copy
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Model folder listing */}
        {modelFiles.length > 0 && (
          <div className="card space-y-4 mb-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Unlearned Model Files</h2>
              <span className="text-xs text-gray-500 truncate max-w-md">
                {artifacts?.unlearned_model_uri}
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead className="text-gray-500">
                  <tr>
                    <th className="text-left px-3 py-2 border-b border-gray-800">Name</th>
                    <th className="text-left px-3 py-2 border-b border-gray-800">Size</th>
                    <th className="text-left px-3 py-2 border-b border-gray-800">Updated</th>
                    <th className="text-left px-3 py-2 border-b border-gray-800">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {modelFiles.map((f) => {
                    const gs = `gs://${f.__bucket}/${f.name}`;
                    return (
                      <tr key={f.name} className="hover:bg-white/5 transition-colors">
                        <td className="px-3 py-2 border-b border-gray-800/50 break-all text-gray-300">
                          {f.name}
                        </td>
                        <td className="px-3 py-2 border-b border-gray-800/50 text-gray-400">
                          {f.size || "—"}
                        </td>
                        <td className="px-3 py-2 border-b border-gray-800/50 text-gray-400">
                          {f.updated || "—"}
                        </td>
                        <td className="px-3 py-2 border-b border-gray-800/50">
                          <a
                            className="text-xs text-gray-400 hover:text-white transition-colors underline"
                            href={`${API}/assets/stream?uri=${encodeURIComponent(gs)}`}
                          >
                            Download
                          </a>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <div className="text-xs text-gray-500 mt-2">Items: {modelFiles.length}</div>
            </div>
          </div>
        )}
      </div>

      {/* Styles */}
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

export default Unlearn;
