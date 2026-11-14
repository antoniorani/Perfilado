/** Reassemble independently computed, ordered and validated LFW descriptors. */
import fs from "node:fs/promises";
import path from "node:path";

const PARTS = "./build/parts";
const OUT = "./build/final";
const ROWS = 5749;
const WIDTH = 128;

async function main() {
  const entries = (await fs.readdir(PARTS, { withFileTypes: true })).filter((e) => e.isDirectory()).sort((a,b)=>a.name.localeCompare(b.name));
  if (!entries.length) throw new Error("No batch artifacts");
  const matrix = new Float32Array(ROWS * WIDTH);
  matrix.fill(NaN);
  const seen = new Uint8Array(ROWS);
  const failures = [];
  let model = null, count = 0;
  const batches = [];
  for (const entry of entries) {
    const base = path.join(PARTS, entry.name);
    const report = JSON.parse(await fs.readFile(path.join(base, "lfw-gallery-report.json"), "utf8"));
    const bytes = await fs.readFile(path.join(base, "lfw-gallery.f32"));
    if (report.bytes !== bytes.length || bytes.length !== report.globalIndices.length * WIDTH * 4) {
      throw new Error(`Size mismatch in ${entry.name}`);
    }
    if (report.originalIdentityCount !== ROWS || (model != null && model !== report.model)) {
      throw new Error(`Incompatible model or catalog in ${entry.name}`);
    }
    model = report.model;
    const arr = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    for (let i = 0; i < report.globalIndices.length; i += 1) {
      const globalIndex = report.globalIndices[i];
      if (!Number.isInteger(globalIndex) || globalIndex < 0 || globalIndex >= ROWS || seen[globalIndex]) {
        throw new Error(`Duplicate or invalid global index ${globalIndex}`);
      }
      seen[globalIndex] = 1;
      matrix.set(arr.subarray(i * WIDTH, (i + 1) * WIDTH), globalIndex * WIDTH);
      count += 1;
    }
    failures.push(...report.failed);
    batches.push({ batch: entry.name, valid: report.valid, failed: report.failed.length, seconds: report.elapsedSeconds });
  }
  if (count !== ROWS || seen.some((x) => x !== 1)) {
    throw new Error(`Missing subjects: ${count} out of ${ROWS} present`);
  }
  const valid = Array.from({ length: ROWS }, (_, i) => matrix.subarray(i * WIDTH, (i + 1) * WIDTH))
    .filter((v) => v.every(Number.isFinite)).length;
  if (valid + failures.length !== ROWS) throw new Error("Inconsistent successful and failed references");
  await fs.mkdir(OUT, { recursive: true });
  await fs.writeFile(path.join(OUT, "lfw-gallery.f32"), Buffer.from(matrix.buffer));
  const report = {
    source: "LFW, first photograph for each of 5,749 people",
    model, operation: "faceapi.computeFaceDescriptor(image)",
    order: "same as data/lfw-identities.json", dimensions: WIDTH,
    subjects: ROWS, valid, failed: failures.length,
    failures, batches, bytes: matrix.byteLength
  };
  await fs.writeFile(path.join(OUT, "lfw-gallery-report.json"), JSON.stringify(report, null, 2));
  console.log("Gallery combined:", JSON.stringify({ subjects: ROWS, valid, failed: failures.length, bytes: matrix.byteLength }));
  if (failures.length) console.log("Failed IDs:", failures.slice(0, 30));
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
