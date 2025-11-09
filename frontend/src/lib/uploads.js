export async function getUploadUrl(type, filename) {
  const r = await fetch("/api/assets/upload-url", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type, filename }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json(); // { uploadUrl, assetId, gcsPath }
}

export async function putToSignedUrl(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url, true);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded * 100) / e.total));
    };
    xhr.onload = () => (xhr.status >= 200 && xhr.status < 300) ? resolve() : reject(new Error(xhr.responseText || `PUT failed: ${xhr.status}`));
    xhr.onerror = () => reject(new Error("Network error during upload"));
    xhr.send(file);
  });
}

export async function commitAsset({ assetId, type, filename, bytes, checksum, meta }) {
  const r = await fetch("/api/assets/commit", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assetId, type, filename, bytes, checksum, meta }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function sha256(file) {
  const buf = await file.arrayBuffer();
  const hash = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, "0")).join("");
}
