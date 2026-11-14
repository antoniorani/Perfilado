/** Merge 302 real recovered descriptors into original 5,749-row LFW matrix.
 * Existing 5,447 aligned vectors stay BIT-FOR-BIT identical.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { createProfiles } from "../src/core.js";
import { encodeGallery, GALLERY_VERSION } from "../src/gallery-format.js";

const orig = "./build/base";
const parts = "./build/patches";
const out = "./build/final";
const roster = JSON.parse(await fs.readFile(new URL("../data/lfw-identities.json",import.meta.url),"utf8"));
const profiles = createProfiles(roster.identities);
const oldReport = JSON.parse(await fs.readFile(path.join(orig,"lfw-gallery-v2-report.json"),"utf8"));
const oldBytes = await fs.readFile(path.join(orig,"lfw-gallery-v2.pfg"));
const expectedSha = oldReport.sha256;
const foundSha = createHash("sha256").update(oldBytes).digest("hex");
if (foundSha !== expectedSha || oldReport.subjects !== 5749 || oldReport.failed !== 302) {
  throw new Error("Unexpected original v2 artifact hash or count");
}
const old = new Float32Array(oldBytes.buffer.slice(
  oldBytes.byteOffset + 64, oldBytes.byteOffset + oldBytes.byteLength));
const pending = new Set(oldReport.failures.map((entry)=>entry.index));
if (pending.size!==302) throw new Error("Original failure indices are inconsistent");
const recoveredMethods = new Map();
const duplicates = new Set();
const stillMissing = [];
const dirs = (await fs.readdir(parts,{withFileTypes:true}))
  .filter((entry)=>entry.isDirectory()).map((entry)=>entry.name).sort();
if (dirs.length!==8) throw new Error(`Expected 8 patch artifacts, got ${dirs.length}`);
for (const dir of dirs) {
  const manifest = JSON.parse(await fs.readFile(path.join(parts,dir,"recovered.json"),"utf8"));
  const count = manifest.recovered.length + manifest.failures.length;
  if (count !== manifest.expected || manifest.batches !== 8) {
    throw new Error(`Incomplete recovery shard ${dir}`);
  }
  for (const entry of manifest.recovered) {
    const index = entry.index;
    if (!pending.has(index)||duplicates.has(index)) {
      throw new Error(`Unexpected duplicate or non-failed person index ${index}`);
    }
    if (!Array.isArray(entry.descriptor)||entry.descriptor.length!==128 ||
      !entry.descriptor.every(Number.isFinite)) {
      throw new Error(`Invalid recovered descriptor at ${index}`);
    }
    if (profiles[index].id !== entry.id) throw new Error(`Index/name mismatch at ${index}`);
    old.set(entry.descriptor,index*128);
    recoveredMethods.set(index,entry.method);
    duplicates.add(index);
  }
  for (const miss of manifest.failures) {
    if (!pending.has(miss.index)||duplicates.has(miss.index)) throw new Error("Unexpected failure index");
    stillMissing.push(miss);
    duplicates.add(miss.index);
  }
}
if (duplicates.size!==302) throw new Error(`Not all original failures checked (${duplicates.size}/302)`);
const total = profiles.length,valid = Array.from({length:total},(_,i)=>old.subarray(i*128,(i+1)*128))
  .filter(row=>row.every(Number.isFinite)).length;
if (valid !== total-stillMissing.length) throw new Error("Final valid count mismatch");
const buffer = Buffer.from(encodeGallery(old, profiles));
const sha256 = createHash("sha256").update(buffer).digest("hex");
const methodCounts = {};
for(const method of recoveredMethods.values()) methodCounts[method]=(methodCounts[method]||0)+1;
const report={
  galleryVersion:GALLERY_VERSION,
  baseSha256:expectedSha,
  baseValid:5749-302,
  attemptedRecovery:302,
  recovered:recoveredMethods.size,
  methods:methodCounts,
  caveat:"direct-unverified: output of pretrained face recognition network without confirmed face detection/landmark alignment. Not equivalent in quality to confirmed aligned descriptors.",
  valid,remainingFailed:stillMissing.length,remainingFailures:stillMissing,
  fallbackIndices:[...recoveredMethods].map(([index,method])=>({index,id:profiles[index].id,method})),
  rows:total,dimensions:128,bytes:buffer.length,sha256,
};
await fs.mkdir(out,{recursive:true});
await fs.writeFile(path.join(out,"lfw-gallery-v3.pfg"),buffer);
await fs.writeFile(path.join(out,"lfw-gallery-v3-report.json"),JSON.stringify(report,null,2));
console.log("RECOVERED_GALLERY_RESULT",JSON.stringify({
  total,valid,recovered:recoveredMethods.size,remainingFailed:stillMissing.length,
  methods:methodCounts,sha256,
}));
if(stillMissing.length)process.exitCode=1;
