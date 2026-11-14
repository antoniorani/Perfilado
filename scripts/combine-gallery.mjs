/** Assemble aligned reference vectors with model metadata and full integrity checks. */
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { createProfiles } from "../src/core.js";
import { PIPELINE_ID } from "../src/face-pipeline.js";
import { encodeGallery } from "../src/gallery-format.js";

const partsDir = process.env.PARTS_DIR || "./build/parts";
const outDir = process.env.OUT_DIR || "./build/final";
const roster = JSON.parse(await fs.readFile(new URL("../data/lfw-identities.json",import.meta.url),"utf8"));
const profiles = createProfiles(roster.identities);
const rows = profiles.length, width = 128;
const directories = (await fs.readdir(partsDir,{withFileTypes:true}))
  .filter(x=>x.isDirectory()).map(x=>x.name).sort();
if (!directories.length) throw new Error("Missing aligned gallery batch artifacts");
const matrix = new Float32Array(rows*width);
matrix.fill(NaN);
const seen = new Uint8Array(rows);
const failures = [];
let found = 0;
const batches = [];
for (const dir of directories) {
  const base = path.join(partsDir,dir);
  const report = JSON.parse(await fs.readFile(path.join(base,"lfw-gallery-part-report.json"),"utf8"));
  const bytes = await fs.readFile(path.join(base,"lfw-gallery-part.f32"));
  if (report.pipeline !== PIPELINE_ID || report.originalIdentityCount !== rows
    || report.dimensions !== width || report.bytes !== bytes.length
    || bytes.length !== report.globalIndices.length*width*4) {
    throw new Error(`Incompatible batch format: ${dir}`);
  }
  const values = new Float32Array(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  for (let i=0;i<report.globalIndices.length;i+=1) {
    const index = report.globalIndices[i];
    if (!Number.isInteger(index)||index<0||index>=rows||seen[index]) {
      throw new Error(`Duplicate or out-of-range gallery index: ${index} from ${dir}`);
    }
    seen[index] = 1;
    matrix.set(values.subarray(i*width,(i+1)*width),index*width);
    found += 1;
  }
  failures.push(...report.failed);
  batches.push({name:dir,valid:report.valid,failed:report.failed.length,seconds:report.elapsedSeconds});
}
if (found !== rows || seen.some(x=>x!==1)) throw new Error(`Incomplete gallery: ${found}/${rows}`);
const valid = Array.from({length:rows},(_,index)=>
  matrix.subarray(index*width,(index+1)*width)).filter(x=>x.every(Number.isFinite)).length;
if (valid + failures.length !== rows) throw new Error("Failed count and invalid rows mismatch");
const buffer = encodeGallery(matrix,profiles);
const bytes = Buffer.from(buffer);
const sha256 = createHash("sha256").update(bytes).digest("hex");
const report = {
  pipeline:PIPELINE_ID,
  operation:"ssdMobilenetv1 + 68 facial landmarks + aligned 128D descriptor",
  model:"face-api.js 0.22.2",
  source:"LFW first photo per person",
  artifact:"lfw-gallery-v2.pfg",
  format:"v2 header (64 bytes) + little-endian Float32 matrix",
  dimensions:width,
  subjects:rows,
  valid,
  failures,
  failed:failures.length,
  bytes:bytes.length,
  sha256,
  batches,
};
await fs.mkdir(outDir,{recursive:true});
await fs.writeFile(path.join(outDir,report.artifact),bytes);
await fs.writeFile(path.join(outDir,"lfw-gallery-v2-report.json"),JSON.stringify(report,null,2));
console.log("ALIGNED_GALLERY_RESULT",JSON.stringify({subjects:rows,valid,failed:failures.length,bytes:bytes.length,sha256}));
