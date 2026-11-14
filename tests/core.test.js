import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createProfiles, getSummary, filterProfiles, referenceImageUrl } from "../src/core.js";

const source = JSON.parse(readFileSync(new URL("../data/lfw-identities.json", import.meta.url), "utf8"));
const profiles = createProfiles(source.identities);

test("official LFW roster has exactly 5,749 identities and 13,233 photographs", () => {
  const summary = getSummary(profiles);
  assert.equal(summary.profiles, 5749);
  assert.equal(summary.photographs, 13233);
  assert.equal(summary.references, 5749);
  assert.equal(summary.repeatIdentities, 1680);
});

test("each identity receives one unique reference and stable fictitious profile", () => {
  assert.equal(new Set(profiles.map((p) => p.id)).size, profiles.length);
  assert.deepEqual(createProfiles(source.identities).slice(0, 12), profiles.slice(0, 12));
  assert.ok(profiles.every((p) => p.referenceFile.endsWith("_0001.jpg")));
  assert.equal(profiles[0].name, "AJ Cook");
  assert.ok(profiles.every((p) => p.age >= 21 && p.age <= 66));
});

test("filters work by profile ID, fictitious zone, and age", () => {
  const candidate = profiles[41];
  assert.deepEqual(filterProfiles(profiles, { query: candidate.id.toLowerCase() }).map((p) => p.id), [candidate.id]);
  assert.ok(filterProfiles(profiles, { zone: candidate.zone }).every((p) => p.zone === candidate.zone));
  assert.deepEqual(filterProfiles(profiles, { query: candidate.name.toLowerCase() }).map((p) => p.id), [candidate.id]);
  const sorted = filterProfiles(profiles, { order: "age" });
  assert.ok(sorted.every((p, index) => index === 0 || sorted[index - 1].age <= p.age));
});

test("LFW photograph URLs use the correct 1,000-image shard boundaries", () => {
  assert.equal(
    referenceImageUrl(profiles[0]),
    "https://huggingface.co/datasets/marcelohaps/lfw/resolve/main/train/images/000/AJ_Cook_0001.jpg",
  );
  const asif = profiles.find((p) => p.lfwIdentity === "Asif_Hanif");
  assert.equal(asif.referenceShard, "001");
  assert.match(referenceImageUrl(asif), /images\/001\/Asif_Hanif_0001\.jpg$/);
  assert.equal(profiles.at(-1).referenceShard, "013");
});

test("invented zones are deliberately imbalanced, but every zone exists", () => {
  const summary = getSummary(profiles);
  const counts = Object.values(summary.byZone);
  assert.equal(counts.length, 5);
  assert.ok(counts.every((v) => v > 0));
  assert.ok(counts[0] > counts[1] && counts[1] > counts[2]);
  assert.ok(counts[2] > counts[3] && counts[3] > counts[4]);
});
