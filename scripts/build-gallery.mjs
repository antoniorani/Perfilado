/** Batch-compute real aligned face embeddings; no fabricated fallback vectors. */
import fs from "node:fs/promises";
import { createProfiles, photoUrl } from "../src/core.js";
import { FACE_API_VERSION, PIPELINE_ID } from "../src/face-pipeline.js";
import { loadFaceModels, extractRemoteFaceDescriptor } from "./node-face-model.mjs";

const out = process.env.OUT_DIR || "./build";
const limit = Number(process.env.LIMIT || 0);
const partitions = Number(process.env.SHARD_COUNT || 1);
const partition = Number(process.env.SHARD_INDEX || 0);
if (![limit,partitions,partition].every(Number.isInteger) || partitions < 1 || partition < 0 || partition >= partitions || limit < 0) {
  throw new RangeError("Invalid partition or limit");
}
const roster = JSON.parse(await fs.readFile(new URL("../data/lfw-identities.json", import.meta.url), "utf8"));
const profiles = createProfiles(roster.identities);
const chosen = profiles.map((person, index) => ({person,index}))
  .filter((x)=>x.index % partitions === partition);
const selected = limit ? chosen.slice(0,limit) : chosen;
console.log(`Aligned gallery batch ${partition + 1}/${partitions}, subjects: ${selected.length}`);
const started = Date.now();
await loadFaceModels();

const vectors = new Float32Array(selected.length * 128);
const failed = [];
let valid = 0;
for (let i = 0; i < selected.length; i += 1) {
  const { person, index } = selected[i];
  try {
    const descriptor = await extractRemoteFaceDescriptor(photoUrl(person,1));
    vectors.set(descriptor,i*128);
    valid += 1;
  } catch(error) {
    failed.push({index,id:person.id,reason:String(error.message||error)});
    vectors.fill(NaN,i*128,(i+1)*128);
  }
  if ((i+1)%25===0 || i+1===selected.length) {
    console.log(`Aligned ${i+1}/${selected.length}: valid=${valid}, failed=${failed.length}; elapsed=${Math.round((Date.now()-started)/1000)}s`);
  }
}
await fs.mkdir(out,{recursive:true});
await fs.writeFile(`${out}/lfw-gallery-part.f32`,Buffer.from(vectors.buffer));
const report = {
  source:"LFW original 250px photos; first photograph for each subject",
  pipeline:PIPELINE_ID,model:`face-api.js ${FACE_API_VERSION}`,
  operation:"detectSingleFace(ssdMobilenetv1).withFaceLandmarks().withFaceDescriptor()",
  referenceCatalog:"data/lfw-identities.json",
  shardIndex:partition,shardCount:partitions,originalIdentityCount:profiles.length,
  globalIndices:selected.map((x)=>x.index),
  dimensions:128,subjects:selected.length,valid,failed,
  bytes:vectors.byteLength,elapsedSeconds:Math.round((Date.now()-started)/1000),
};
await fs.writeFile(`${out}/lfw-gallery-part-report.json`,JSON.stringify(report,null,2));
console.log("BATCH_RESULT",JSON.stringify({subjects:report.subjects,valid,failures:failed.length}));
