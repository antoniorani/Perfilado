/** Independent real-photo evaluation of the full aligned LFW reference gallery. */
import fs from "node:fs/promises";
import { createProfiles, photoUrl } from "../src/core.js";
import { chooseProbePhotos, readGallery, nearestCandidates, classifyProbe, summariseProbes } from "../src/full-gallery.js";
import { PIPELINE_ID } from "../src/face-pipeline.js";
import { loadFaceModels, extractRemoteFaceDescriptor } from "./node-face-model.mjs";

const dataDir = process.env.OUT_DIR || "./build/final";
const roster = JSON.parse(await fs.readFile(new URL("../data/lfw-identities.json", import.meta.url), "utf8"));
const profiles = createProfiles(roster.identities);
const raw = await fs.readFile(`${dataDir}/lfw-gallery-v2.pfg`);
const {gallery,missing} = readGallery(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength),profiles);
const samples = chooseProbePhotos(profiles,100);
console.log(`EVALUATION_START ${PIPELINE_ID}; gallery=${gallery.length}, missing=${missing.length}, probes=${samples.length}`);
await loadFaceModels();
const probes=[], failures=[];
for (let i=0;i<samples.length;i+=1) {
  const photo=samples[i];
  try {
    const desc=await extractRemoteFaceDescriptor(photoUrl(photo.person,photo.imageNumber));
    probes.push({...photo, candidates:nearestCandidates(desc,gallery,5)});
  } catch(error) {
    failures.push({index:i,id:photo.person.id,reason:String(error.message || error)});
  }
  if ((i+1)%10===0) console.log(`EVALUATION_PROGRESS ${i+1}/100 processed=${probes.length} failed=${failures.length}`);
}
const thresholds=[0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.8];
const summaries=thresholds.map((threshold)=>({
  threshold,...summariseProbes(probes.map((p)=>classifyProbe(p,p.candidates,threshold)),failures.length),
}));
const rawTop1=probes.filter((p)=>p.candidates[0].person.id===p.person.id).length;
const top5=probes.filter((p)=>p.candidates.some((candidate)=>candidate.person.id===p.person.id)).length;
const report={
  model:PIPELINE_ID,source:"LFW",galleryReferences:gallery.length,
  galleryMissing:missing.length, attempted:100,processed:probes.length,failures,
  top1:rawTop1,top5,top1Accuracy:probes.length ? rawTop1/probes.length : null,
  top5Accuracy:probes.length ? top5/probes.length : null,
  thresholdOutcomes:summaries,
  // Aggregate evaluation only; no sensitive query embeddings in the report.
};
await fs.writeFile(`${dataDir}/lfw-evaluation-100.json`,JSON.stringify(report,null,2));
console.log("EVALUATION_RESULT", JSON.stringify({
  processed:report.processed,top1:rawTop1,top5,
  accuracyTop1:report.top1Accuracy,failed:failures.length,
  threshold055:summaries.find((x)=>x.threshold===0.55),
}));
if (probes.length<85||report.top1Accuracy < 0.45) {
  throw new Error("Full-gallery validation requires inspection: too many failures or low top-1 performance");
}
