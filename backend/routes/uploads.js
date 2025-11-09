import express from "express";
import { Storage } from "@google-cloud/storage";
import dotenv from "dotenv";
dotenv.config();

const router = express.Router();


function getUserId(req) {
  return "demo-user-001"; // temporary
}

const storage = new Storage({
  projectId: process.env.GCP_PROJECT_ID,
  keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,
});
const bucket = storage.bucket(process.env.GCS_BUCKET);

const VALID = {
  train: [".csv", ".json", ".jsonl"],
  forget: [".csv", ".json", ".jsonl"],
};

const folder = (t) => "datasets";
const uuid = () =>
  Math.random().toString(16).slice(2) + Date.now().toString(16);

const extOk = (t, name) =>
  (VALID[t] || []).some((e) => name.toLowerCase().endsWith(e));

async function signedPutUrl(file) {
  const [url] = await file.getSignedUrl({
    version: "v4",
    action: "write",
    expires: Date.now() + 30 * 60 * 1000, // 30 min
    // no contentType binding
  });
  return url;
}


router.post("/assets/upload-url", async (req, res) => {
  try {
    const { type, filename } = req.body || {};
    if (!type || !filename) {
      return res.status(400).json({ error: "type and filename required" });
    }
    if (!extOk(type, filename)) {
      return res.status(400).json({ error: `Invalid file extension for ${type}` });
    }

    const userId = getUserId(req); // or "demo-user"
    const assetId = uuid();
    const key = `users/${userId}/${folder(type)}/${assetId}/${filename}`;
    const file = bucket.file(key);
    const uploadUrl = await signedPutUrl(file);

    return res.json({ uploadUrl, assetId, gcsPath: `gs://${process.env.GCS_BUCKET}/${key}` });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Failed to create signed URL" });
  }
});

router.post("/assets/commit", async (req, res) => {
  try {
    const { assetId, type, filename } = req.body || {};
    if (!assetId || !type || !filename) {
      return res.status(400).json({ error: "assetId, type, filename required" });
    }

    const userId = getUserId(req);
    const key = `users/${userId}/${folder(type)}/${assetId}/${filename}`;
    const file = bucket.file(key);
    const [exists] = await file.exists();
    if (!exists) {
      return res.status(400).json({ error: "Object not found in GCS. Did you PUT to the signed URL?" });
    }

    const metaKey = `users/${userId}/${folder(type)}/${assetId}/meta.json`;
    const metaFile = bucket.file(metaKey);
    const metaPayload = {
      assetId,
      type,
      filename,
      createdAt: new Date().toISOString(),
      gcsPath: `gs://${process.env.GCS_BUCKET}/${key}`,
      userId,
    };

    await metaFile.save(JSON.stringify(metaPayload), { contentType: "application/json" });
    return res.json({ status: "READY", asset: metaPayload });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Commit failed" });
  }
});


// Parse "gs://bucket/path/to/file" => { bucket, key }
function parseGsUri(uri) {
  if (!uri || !uri.startsWith("gs://")) throw new Error("Not a gs:// URI");
  const rest = uri.slice(5);
  const idx = rest.indexOf("/");
  if (idx === -1) return { bucket: rest, key: "" };
  return { bucket: rest.slice(0, idx), key: rest.slice(idx + 1) };
}

/**
 * GET /api/assets/sign?uri=gs://bucket/path/file.png&expires=300
 * Returns a short-lived signed URL (great for <img src=...>)
 */
router.get("/assets/sign", async (req, res) => {
  try {
    const { uri } = req.query;
    const expires = Number(req.query.expires || 300);
    const { bucket: bkt, key } = parseGsUri(uri);
    const [url] = await storage.bucket(bkt).file(key).getSignedUrl({
      version: "v4",
      action: "read",
      expires: Date.now() + expires * 1000,
    });
    res.json({ url, expires });
  } catch (e) {
    console.error(e);
    res.status(400).json({ error: e.message || "Failed to sign URI" });
  }
});

/**
 * GET /api/assets/stream?uri=gs://bucket/path/file.csv
 * Streams the object through your server (good for downloads & CSV)
 */
router.get("/assets/stream", async (req, res) => {
  try {
    const { uri } = req.query;
    const { bucket: bkt, key } = parseGsUri(uri);
    const file = storage.bucket(bkt).file(key);
    const [meta] = await file.getMetadata();
    res.setHeader("Content-Type", meta.contentType || "application/octet-stream");
    file.createReadStream()
      .on("error", (err) => { console.error(err); res.status(500).end("Stream error"); })
      .pipe(res);
  } catch (e) {
    console.error(e);
    res.status(400).json({ error: e.message || "Failed to stream" });
  }
});

/**
 * GET /api/assets/csv?uri=gs://bucket/path/file.csv&limit=300
 * Reads CSV into JSON rows (preview)
 */
import { parse } from "csv-parse";
router.get("/assets/csv", async (req, res) => {
  try {
    const { uri } = req.query;
    const limit = Number(req.query.limit || 300);
    const { bucket: bkt, key } = parseGsUri(uri);
    const file = storage.bucket(bkt).file(key);

    const rows = [];
    const parser = file.createReadStream().pipe(parse({
      columns: true,
      skip_empty_lines: true,
      relax_column_count: true,
    }));
    parser.on("data", (row) => { if (rows.length < limit) rows.push(row); });
    parser.on("end", () => res.json({ rows, limit }));
    parser.on("error", (err) => {
      console.error(err);
      res.status(500).json({ error: "CSV parse failed" });
    });
  } catch (e) {
    console.error(e);
    res.status(400).json({ error: e.message || "Failed to read CSV" });
  }
});

/**
 * GET /api/assets/list?prefix=gs://bucket/path/to/folder/
 * Lists objects under a prefix (for model folder)
 */
router.get("/assets/list", async (req, res) => {
  try {
    const { prefix } = req.query;
    const { bucket: bkt, key } = parseGsUri(prefix);
    const [files] = await storage.bucket(bkt).getFiles({ prefix: key });
    res.json({
      bucket: bkt,
      prefix: key,
      items: files.map((f) => ({
        name: f.name, // full key including prefix
        size: f.metadata?.size,
        contentType: f.metadata?.contentType,
        updated: f.metadata?.updated,
      })),
    });
  } catch (e) {
    console.error(e);
    res.status(400).json({ error: e.message || "Failed to list objects" });
  }
});


export default router;
