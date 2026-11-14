/** CI smoke-test: real matched LFW pairs must be separable after alignment. */
import fs from "node:fs/promises";
import { createProfiles, photoUrl } from "../src/core.js";
import { chooseProbePhotos, nearestCandidates } from "../src/full-gallery.js";
import { PIPELINE_ID } from "../src/face-pipeline.js";
import { loadFaceModels, extractRemoteFaceDescriptor } from "./node-face-model.mjs";

const count = Number(process.env.PAIR_COUNT || "24");
const target = (process.env.OUT_DIR || "./build/validation");
const index = JSON.parse(await fs.readFile(new URL("../data/lfw-identities.json", import.meta.url), "utf8"));
const profiles = createProfiles(index.identities);
const probePhotos = chooseProbePhotos(profiles, count);
await loadFaceModels();
const gallery = [], validProbes = [], failures = [];
for (let i = 0; i < probePhotos.length; i += 1) {
  const {person, imageNumber} = probePhotos[i];
  try {
    const reference = await extractRemoteFaceDescriptor(photoUrl(person, 1));
    const probe = await extractRemoteFaceDescriptor(photoUrl(person, imageNumber));
    gallery.push({ person, descriptor: reference });
    validProbes.push({ person, descriptor: probe });
  } catch(error) {
    failures.push({ index: i, id: person.id, reason: error.message });
  }
  console.log(`Validated pair ${i + 1}/${count}; usable=${gallery.length}, failed=${failures.length}`);
}
const distances = validProbes.map((probe) => {
  const candidates = nearestCandidates(probe.descriptor, gallery, 5);
  return {
    identity: probe.person.id,
    top1Correct: candidates[0].person.id === probe.person.id,
    nearest: candidates[0].distance,
    genuineDistance: Math.hypot(...probe.descriptor.map((value,index) =>
      value - gallery.find((x) => x.person.id === probe.person.id).descriptor[index])),
    top5Correct: candidates.some((x) => x.person.id === probe.person.id),
  };
});
function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a,b)=>a-b);
  const middle = Math.floor(sorted.length/2);
  return sorted.length%2 ? sorted[middle] : (sorted[middle-1]+sorted[middle])/2;
}
const correct = distances.filter((d) => d.top1Correct).length;
const report = {
  pipeline: PIPELINE_ID, attempted: count, processed: gallery.length, failures,
  correctTop1: correct, accuracyTop1: distances.length ? correct / distances.length : null,
  top5: distances.filter((x)=>x.top5Correct).length,
  medianGenuineDistance: median(distances.map((x)=>x.genuineDistance)),
  medianNearestDistance: median(distances.map((x)=>x.nearest)),
  examples: distances.slice(0,10),
};
await fs.mkdir(target,{recursive:true});
await fs.writeFile(`${target}/aligned-pairs-validation.json`, JSON.stringify(report,null,2));
console.log("VALIDATION_SUMMARY", JSON.stringify({
  attempted:report.attempted,processed:report.processed,
  top1:correct,accuracyTop1:report.accuracyTop1,
  medianGenuineDistance:report.medianGenuineDistance,
  medianNearestDistance:report.medianNearestDistance,
  failures:failures.length,
}));
if (report.processed < Math.ceil(count * .80) || report.accuracyTop1 < .55) {
  console.error("VALIDATION FAILED: aligned embeddings are not reliable enough to proceed");
  process.exitCode = 1;
}
