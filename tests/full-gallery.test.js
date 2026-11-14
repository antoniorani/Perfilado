import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createProfiles } from "../src/core.js";
import { encodeGallery, decodeGallery } from "../src/gallery-format.js";
import { computeAlignedDescriptor, PIPELINE_ID } from "../src/face-pipeline.js";
import {
  DESCRIPTOR_SIZE, chooseProbePhotos, readGallery,
  nearestCandidates, classifyProbe, summariseProbes,
} from "../src/full-gallery.js";

const data = JSON.parse(readFileSync(new URL("../data/lfw-identities.json", import.meta.url), "utf8"));
const profiles = createProfiles(data.identities);

test("exactly one second photograph is sampled for 100 different enrolled identities", () => {
  const sample = chooseProbePhotos(profiles, 100);
  assert.deepEqual(chooseProbePhotos(profiles, 100), sample);
  assert.equal(sample.length, 100);
  assert.equal(new Set(sample.map((s) => s.person.id)).size, 100);
  assert.ok(sample.every((s) => s.imageNumber === 2 && s.person.imageCount >= 2));
  assert.throws(() => chooseProbePhotos(profiles, 5000), RangeError);
});

test("float32 gallery corresponds to all profiles in the documented order", () => {
  const matrix = new Float32Array(profiles.length * DESCRIPTOR_SIZE);
  matrix[0] = 2;
  matrix[DESCRIPTOR_SIZE] = 3;
  const result = readGallery(encodeGallery(matrix, profiles), profiles);
  assert.equal(result.gallery.length, 5749);
  assert.equal(result.gallery[0].person.id, "PF-00001");
  assert.equal(result.gallery[1].descriptor[0], 3);
  assert.equal(result.missing.length, 0);
  assert.throws(() => readGallery(new ArrayBuffer(64), profiles), /antigua|incompatible/i);
  assert.equal(PIPELINE_ID, "faceapi0222-ssd68-aligned-v2");
});

test("corrupt descriptors are excluded instead of interpreted as zero vectors", () => {
  const matrix = new Float32Array(profiles.length * DESCRIPTOR_SIZE);
  matrix[256] = NaN;
  const result = readGallery(encodeGallery(matrix, profiles), profiles);
  assert.equal(result.gallery.length, 5748);
  assert.deepEqual(result.missing, [profiles[2].id]);
});

test("nearest-neighbor search returns sorted top-5 distances over genuine vectors", () => {
  const zeros = (first) => {
    const vector = new Float32Array(DESCRIPTOR_SIZE);
    vector[0] = first;
    return vector;
  };
  const refs = profiles.slice(0, 7).map((person, index) => ({ person, descriptor: zeros(index) }));
  const results = nearestCandidates(zeros(2.1), refs, 5);
  assert.equal(results.length, 5);
  assert.equal(results[0].person.id, refs[2].person.id);
  assert.ok(results.every((c, index) => index === 0 || c.distance >= results[index - 1].distance));
  assert.throws(() => nearestCandidates([1,2], refs), /invalid|inválido/);
});

test("top-1 accuracy does not change with threshold, while accepts and rejects do", () => {
  const a = profiles[0], b = profiles[1];
  const raw = [
    { person: a, candidates: [{ person: a, distance: 0.55 }, { person: b, distance: 0.75 }] },
    { person: b, candidates: [{ person: a, distance: 0.45 }, { person: b, distance: 0.52 }] },
    { person: b, candidates: [{ person: b, distance: 0.91 }, { person: a, distance: 1.0 }] },
  ];
  const at60 = summariseProbes(raw.map((x) => classifyProbe(x, x.candidates, 0.60)), 0);
  const at100 = summariseProbes(raw.map((x) => classifyProbe(x, x.candidates, 1.00)), 0);
  assert.equal(at60.top1, 2);
  assert.equal(at100.top1, 2);
  assert.deepEqual([at60.correct, at60.wrong, at60.rejected], [1, 1, 1]);
  assert.deepEqual([at100.correct, at100.wrong, at100.rejected], [2, 1, 0]);
  assert.equal(at60.attempted, 3);
});

test("v2 gallery rejects legacy files, changed model and mismatched catalog", () => {
  const input = new Float32Array(profiles.length * DESCRIPTOR_SIZE);
  const valid = encodeGallery(input, profiles);
  assert.equal(decodeGallery(valid, profiles).length, input.length);
  assert.throws(() => readGallery(input.buffer, profiles), /antigua|incompatible/i);
  const differentModel = valid.slice(0);
  new Uint8Array(differentModel)[16] = "!".charCodeAt(0);
  assert.throws(() => readGallery(differentModel, profiles), /otro modelo|preprocesamiento/i);
  assert.throws(() => readGallery(valid, profiles.slice(0, -1)), /identidades/i);
});

test("aligned descriptor uses SSD detection, face landmarks and recognition sequence", async () => {
  let called = [];
  class Options { constructor(config) { called.push(["options", config.minConfidence]); } }
  const api = {
    SsdMobilenetv1Options: Options,
    detectSingleFace(image, options) {
      called.push(["detector", image.id, options instanceof Options]);
      return {
        withFaceLandmarks() {
          called.push(["landmarks"]);
          return {
            async withFaceDescriptor() {
              called.push(["descriptor"]);
              return { descriptor: new Float32Array(128).fill(0.1) };
            },
          };
        },
      };
    },
  };
  const descriptor = await computeAlignedDescriptor(api, {id:"sample"});
  assert.equal(descriptor.length,128);
  assert.deepEqual(called.map((x) => x[0]),["options","detector","landmarks","descriptor"]);
  assert.deepEqual(called[1],["detector","sample",true]);
});
