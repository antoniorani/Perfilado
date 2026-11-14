/** Recover only the LFW references that failed standard SSD recognition.
 * First retry two actual face detectors, then document a direct image descriptor
 * separately (not a verified/fully aligned face embedding).
 */
import fs from "node:fs/promises";
import { createProfiles, photoUrl } from "../src/core.js";
import { loadFaceModels, loadTinyFaceModel, downloadBytes, faceapi, canvas } from "./node-face-model.mjs";

const BATCH_COUNT = Number(process.env.SHARD_COUNT || 8);
const BATCH_ID = Number(process.env.SHARD_INDEX || 0);
if (!Number.isInteger(BATCH_COUNT) || !Number.isInteger(BATCH_ID) ||
    BATCH_COUNT < 1 || BATCH_ID < 0 || BATCH_ID >= BATCH_COUNT) {
  throw new Error("Invalid recovery partition");
}
const source = "./build/base/lfw-gallery-v2-report.json";
const roster = JSON.parse(await fs.readFile(new URL("../data/lfw-identities.json",import.meta.url), "utf8"));
const profiles = createProfiles(roster.identities);
const baseline = JSON.parse(await fs.readFile(source,"utf8"));
if (baseline.pipeline !== "faceapi0222-ssd68-aligned-v2" || baseline.subjects !== 5749) {
  throw new Error("Recovery requires verified v2 original reference gallery");
}
const missing = baseline.failures
  .filter((entry)=>entry.index % BATCH_COUNT === BATCH_ID);
const started = Date.now();
console.log("RECOVERY_START",JSON.stringify({batch:BATCH_ID,total:missing.length}));
await loadFaceModels();
await loadTinyFaceModel();

async function aligned(detection) {
  const result = await detection.withFaceLandmarks().withFaceDescriptor();
  return result?.descriptor || null;
}
function valid(vector) {
  return vector?.length === 128 && Array.from(vector).every(Number.isFinite);
}
const recovered = [];
const failures = [];
for (const [i,missingEntry] of missing.entries()) {
  const person = profiles[missingEntry.index];
  if (person.id !== missingEntry.id) throw new Error("Catalog index mismatch");
  try {
    const bytes = await downloadBytes(photoUrl(person,1));
    const image = await canvas.loadImage(bytes);
    const methods = [
      ["ssd-lower-confidence",() => aligned(faceapi.detectSingleFace(image,
        new faceapi.SsdMobilenetv1Options({ minConfidence: 0.07 })))],
      ["tiny-aligned",() => aligned(faceapi.detectSingleFace(image,
        new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.10 })))],
      ["ssd-upscaled",() => {
        const upscaled = canvas.createCanvas(image.width*2,image.height*2);
        upscaled.getContext("2d").drawImage(image,0,0,upscaled.width,upscaled.height);
        return aligned(faceapi.detectSingleFace(upscaled,
          new faceapi.SsdMobilenetv1Options({ minConfidence: 0.07 })));
      }],
      // Real model output but unverified landmark alignment. Track explicitly.
      ["direct-unverified",()=>faceapi.computeFaceDescriptor(image)],
    ];
    let result;
    for (const [method,task] of methods) {
      try {
        const vector = await task();
        if (valid(vector)) {
          result = { index:missingEntry.index, id:person.id,method,
            descriptor:Array.from(vector) };
          break;
        }
      } catch(error) {
        console.warn(`Method ${method} failed on ${person.id}: ${error.message}`);
      }
    }
    if (!result) throw new Error("All face embedding extraction methods failed");
    recovered.push(result);
  } catch(error) {
    failures.push({index:missingEntry.index,id:person.id,reason:String(error.message||error)});
  }
  if ((i+1)%10===0||i+1===missing.length) {
    const methods = recovered.reduce((result,x)=>(result[x.method]=(result[x.method]||0)+1,result),{});
    console.log("RECOVERY_PROGRESS",JSON.stringify({done:i+1,total:missing.length,
      valid:recovered.length,failed:failures.length,methods}));
  }
}
await fs.mkdir("./build/recovery",{recursive:true});
const manifest = {
  source:"LFW photos missing from v2 SSD detector",
  batch:BATCH_ID,batches:BATCH_COUNT,expected:missing.length,
  recovered,failures,
  elapsedSeconds:Math.round((Date.now()-started)/1000),
};
await fs.writeFile("./build/recovery/recovered.json",JSON.stringify(manifest));
console.log("RECOVERY_RESULT",JSON.stringify({
  batch:BATCH_ID,expected:missing.length,
  recovered:recovered.length,failed:failures.length,
  breakdown:recovered.reduce((result,x)=>(result[x.method]=(result[x.method]||0)+1,result),{}),
}));
if (failures.length) {
  // A nonzero status is *not* used here: the assembly validates missing rows.
  console.warn("Unrecovered photos must remain explicitly marked, never zero-filled.");
}
