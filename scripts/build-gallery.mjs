/** Build actual face-api.js descriptors for the first LFW image of each subject.
 *
 * Run in GitHub Actions; do not invent descriptors, and never silently replace failures.
 * The output is a privacy-sensitive biometric template matrix.
 */
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { createRequire } from "node:module";
import { createProfiles, photoUrl } from "../src/core.js";

const require = createRequire(import.meta.url);
const faceapi = require("face-api.js");
const canvas = require("canvas");
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const MODEL_URL = "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights";
const MODEL_NAME = "face_recognition_model-weights_manifest.json";
const OUT_DIR = process.env.OUT_DIR || "./build";
const LIMIT = Number(process.env.LIMIT || "0");
const retryDelay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function getBytes(url, attempts = 5) {
  let failure;
  for (let n = 0; n < attempts; n += 1) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(30000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return Buffer.from(await res.arrayBuffer());
    } catch (error) {
      failure = error;
      if (n + 1 < attempts) await retryDelay(700 * (n + 1) ** 2);
    }
  }
  throw new Error(`No se pudo cargar ${url}: ${failure?.message}`);
}

async function downloadWeights() {
  const modelPath = await fs.mkdtemp(path.join(os.tmpdir(), "lfw-model-"));
  const manifest = JSON.parse((await getBytes(`${MODEL_URL}/${MODEL_NAME}`)).toString("utf8"));
  await fs.writeFile(path.join(modelPath, MODEL_NAME), JSON.stringify(manifest));
  const weightFiles = [...new Set(manifest.flatMap((group) => group.paths))];
  for (const file of weightFiles) {
    console.log(`Downloading model shard: ${file}`);
    await fs.writeFile(path.join(modelPath, file), await getBytes(`${MODEL_URL}/${file}`));
  }
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  return { model: MODEL_NAME, weightFiles };
}

async function main() {
  const start = Date.now();
  const index = JSON.parse(await fs.readFile(new URL("../data/lfw-identities.json", import.meta.url), "utf8"));
  const all = createProfiles(index.identities);
  const subjects = LIMIT > 0 ? all.slice(0, LIMIT) : all;
  console.log(`Starting ${subjects.length} real descriptors of ${all.length} LFW subjects`);
  const weights = await downloadWeights();
  const embeddings = new Float32Array(subjects.length * 128);
  const failed = [];
  let completed = 0;
  for (let i = 0; i < subjects.length; i += 1) {
    const person = subjects[i];
    try {
      const bytes = await getBytes(photoUrl(person, 1));
      const image = await canvas.loadImage(bytes);
      const descriptor = await faceapi.computeFaceDescriptor(image);
      if (!descriptor || descriptor.length !== 128 || !Array.from(descriptor).every(Number.isFinite)) {
        throw new Error("face-api.js returned a missing or invalid descriptor");
      }
      embeddings.set(descriptor, i * 128);
      completed += 1;
    } catch (error) {
      failed.push({ index: i, id: person.id, reason: String(error.message || error) });
      // NaN signals a missing descriptor; never interpret it as a 0-vector match.
      embeddings.fill(NaN, i * 128, (i + 1) * 128);
    }
    if ((i + 1) % 50 === 0 || i + 1 === subjects.length) {
      const seconds = (Date.now() - start) / 1000;
      const rate = (i + 1) / Math.max(seconds, 0.01);
      const eta = Math.round((subjects.length - i - 1) / Math.max(rate, 0.001));
      console.log(`Processed ${i + 1}/${subjects.length}; valid=${completed}, failed=${failed.length}; elapsed=${Math.round(seconds)}s, ETA=${eta}s`);
    }
  }
  await fs.mkdir(OUT_DIR, { recursive: true });
  const bytes = Buffer.from(embeddings.buffer, embeddings.byteOffset, embeddings.byteLength);
  await fs.writeFile(path.join(OUT_DIR, "lfw-gallery.f32"), bytes);
  const report = {
    source: "Labeled Faces in the Wild, first photo per identity",
    model: "face-api.js 0.22.2 face_recognition_model",
    operation: "faceapi.computeFaceDescriptor(image)",
    dimensions: 128,
    order: "same index as data/lfw-identities.json",
    subjects: subjects.length,
    valid: completed,
    failed,
    bytes: bytes.length,
    modelWeights: weights,
    elapsedSeconds: Math.round((Date.now() - start) / 1000),
  };
  await fs.writeFile(path.join(OUT_DIR, "lfw-gallery-report.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({subjects: report.subjects, valid: report.valid, failed: failed.length, elapsedSeconds: report.elapsedSeconds}, null, 2));
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
