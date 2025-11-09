import React, { useState } from "react";
import { getUploadUrl, putToSignedUrl, commitAsset, sha256 } from "../lib/uploads";

function UploadCard({ title, accept, type }) {
  const [progress, setProgress] = useState(0);
  const [assetId, setAssetId] = useState("");
  const [status, setStatus] = useState("");

  const onChoose = async (file) => {
    try {
      setStatus("Preparing...");
      const checksum = await sha256(file);

      const { uploadUrl, assetId } = await getUploadUrl(type, file.name);
      setStatus("Uploading...");
      await putToSignedUrl(uploadUrl, file, setProgress);

      setStatus("Finalizing...");
      const resp = await commitAsset({
        assetId,
        type,
        filename: file.name,
        bytes: file.size,
        checksum,
        meta: { userFacingName: title },
      });

      setAssetId(assetId);
      setStatus("Uploaded ✅");
    } catch (e) {
      console.error(e);
      setStatus(`Error: ${e.message}`);
    }
  };

  return (
    <div className="rounded-2xl p-6 feature-card border border-white/10">
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-white/60 text-sm mb-4">
        {type === "modelZip" ? "Upload a HuggingFace checkpoint ZIP." : "Upload CSV/JSON dataset."}
      </p>

      <input
        id={`${type}-input`}
        type="file"
        accept={accept}
        onChange={(e) => e.target.files?.[0] && onChoose(e.target.files[0])}
        className="hidden"
      />
      <label
        htmlFor={`${type}-input`}
        className="inline-flex items-center px-4 py-2 rounded-xl bg-cyan-500/20 border border-cyan-400/30 hover:bg-cyan-500/30 cursor-pointer"
      >
        Select File
      </label>

      {progress > 0 && (
        <div className="mt-4 h-2 w-full bg-white/10 rounded">
          <div className="h-full bg-cyan-400 rounded" style={{ width: `${progress}%` }} />
        </div>
      )}

      <div className="mt-3 text-sm text-white/70">
        {status}
        {assetId && <div className="mt-1 text-cyan-300">assetId: {assetId}</div>}
      </div>
    </div>
  );
}

export default function PrivacyPatchUploads() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl w-full">
      <UploadCard title="Model ZIP" type="modelZip" accept=".zip" />
      <UploadCard title="Train Dataset" type="train" accept=".csv,.json,.jsonl" />
      <UploadCard title="Forget Dataset" type="forget" accept=".csv,.json,.jsonl" />
    </div>
  );
}
