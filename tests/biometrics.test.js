import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createProfiles, photoUrl } from "../src/core.js";
import { createCohort, euclideanDistance, identify, evaluateMatches } from "../src/biometric-core.js";

const roster = JSON.parse(readFileSync(new URL("../data/lfw-identities.json", import.meta.url), "utf8"));
const profiles = createProfiles(roster.identities);

test("a deterministic sample uses one real reference and new photos of enrolled identities", () => {
  const first = createCohort(profiles, 8);
  assert.deepEqual(createCohort(profiles, 8), first);
  assert.equal(first.gallery.length, 8);
  assert.equal(first.probes.length, 12);
  assert.ok(first.gallery.every((item) => item.imageNumber === 1 && item.person.imageCount > 1));
  assert.ok(first.probes.filter((item) => item.enrolled).every((item) => item.imageNumber === 2));
  assert.ok(first.probes.filter((item) => !item.enrolled).every((item) => item.imageNumber === 1));
  assert.equal(new Set(first.gallery.map((item) => item.person.id)).size, 8);
  const enrolledIds = new Set(first.gallery.map((item) => item.person.id));
  assert.ok(first.probes.every((item) => item.enrolled === enrolledIds.has(item.person.id)));
});

test("gallery sizes outside the documented workshop range are rejected", () => {
  for (const size of [1, 0, -1, 31, 1.5]) {
    assert.throws(() => createCohort(profiles, size), RangeError);
  }
});

test("LFW image URLs for later photographs respect actual 1,000-file shard boundaries", () => {
  const index = profiles.find((p) => p.imageOffset < 1000 && p.imageOffset + p.imageCount > 1000);
  assert.ok(index, "expected an identity straddling the 1,000th image boundary");
  const before = 1000 - index.imageOffset;
  assert.match(photoUrl(index, before), /\/images\/000\//);
  assert.match(photoUrl(index, before + 1), /\/images\/001\//);
  assert.throws(() => photoUrl(index, index.imageCount + 1), RangeError);
});

test("euclideanDistance accepts equal-dimensional embeddings only", () => {
  assert.equal(euclideanDistance([0, 0], [3, 4]), 5);
  assert.throws(() => euclideanDistance([1], [1, 2]), TypeError);
  assert.throws(() => euclideanDistance([NaN], [1]), TypeError);
});

test("identification separates correct matches, wrong identities, misses and false alerts", () => {
  const a = { id: "a", name: "A" }, b = { id: "b", name: "B" };
  const c = { id: "c", name: "C" }, d = { id: "d", name: "D" };
  const gallery = [
    { person: a, descriptor: [0, 0] },
    { person: b, descriptor: [1, 1] },
  ];
  const probes = [
    { person: a, enrolled: true, descriptor: [0.1, 0.1] },
    { person: a, enrolled: true, descriptor: [1, 1] },
    { person: b, enrolled: true, descriptor: [4, 4] },
    { person: c, enrolled: false, descriptor: [0.2, 0.2] },
    { person: d, enrolled: false, descriptor: [4, 3] },
  ];
  const metrics = evaluateMatches(identify(probes, gallery, 0.6));
  assert.deepEqual(metrics, {
    total: 5, enrolled: 3, unknown: 2, correct: 1, wrongIdentity: 1,
    missed: 1, falseAlerts: 1, correctRejections: 1,
    precision: 1 / 3, recall: 1 / 3, unknownFalsePositiveRate: 1 / 2,
    globalAccuracy: 2 / 5,
  });
  const stricter = evaluateMatches(identify(probes, gallery, 0.05));
  assert.equal(stricter.falseAlerts, 0);
  assert.ok(stricter.missed > metrics.missed);
});

test("empty gallery or zero-accepted cases are handled without invented results", () => {
  assert.deepEqual(identify([{ enrolled: true, descriptor: [1] }], [], 0.5), []);
  const metrics = evaluateMatches([]);
  assert.equal(metrics.total, 0);
  assert.equal(metrics.precision, null);
  assert.equal(metrics.recall, null);
});
