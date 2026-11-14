/** Offline runner using the same face-api.js chain as the browser. */
import { createRequire } from "node:module";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { MODEL_MANIFESTS, computeAlignedDescriptor } from "../src/face-pipeline.js";

const require = createRequire(import.meta.url);
const api = require("face-api.js");
const canvas = require("canvas");
api.env.monkeyPatch({ Canvas: canvas.Canvas, Image: canvas.Image, ImageData: canvas.ImageData });
const ROOT = "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights";

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function downloadBytes(url, attempts = 5) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      lastError = error;
      if (attempt + 1 < attempts) await delay(500 * (attempt + 1) ** 2);
    }
  }
  throw new Error(`Failed downloading ${url}: ${lastError?.message}`);
}

export async function loadFaceModels() {
  const folder = await fs.mkdtemp(path.join(os.tmpdir(), "perfilado-aligned-"));
  const weights = new Set();
  for (const filename of MODEL_MANIFESTS) {
    const bytes = await downloadBytes(`${ROOT}/${filename}`);
    const manifest = JSON.parse(bytes.toString("utf8"));
    await fs.writeFile(path.join(folder, filename), bytes);
    for (const part of manifest.flatMap((entry) => entry.paths)) weights.add(part);
  }
  for (const file of weights) {
    await fs.writeFile(path.join(folder, file), await downloadBytes(`${ROOT}/${file}`));
  }
  await Promise.all([
    api.nets.ssdMobilenetv1.loadFromDisk(folder),
    api.nets.faceLandmark68Net.loadFromDisk(folder),
    api.nets.faceRecognitionNet.loadFromDisk(folder),
  ]);
  return api;
}

export async function extractRemoteFaceDescriptor(imageUrl) {
  const bytes = await downloadBytes(imageUrl);
  const image = await canvas.loadImage(bytes);
  return computeAlignedDescriptor(api, image);
}
