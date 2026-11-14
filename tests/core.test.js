import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createProfiles, getSummary, filterProfiles, exportProfilesCsv } from "../src/core.js";

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
  assert.ok(profiles.every((p) => p.age >= 21 && p.age <= 66));
});

test("filters work by profile ID, fictitious zone, and age", () => {
  const candidate = profiles[41];
  assert.deepEqual(filterProfiles(profiles, { query: candidate.id.toLowerCase() }).map((p) => p.id), [candidate.id]);
  assert.ok(filterProfiles(profiles, { zone: candidate.zone }).every((p) => p.zone === candidate.zone));
  const sorted = filterProfiles(profiles, { order: "age" });
  assert.ok(sorted.every((p, index) => index === 0 || sorted[index - 1].age <= p.age));
});

test("export CSV includes exactly one reference per profile, without political comments", () => {
  const csv = exportProfilesCsv(profiles);
  assert.equal(csv.trim().split("\r\n").length, 5750);
  assert.match(csv, /imagen_referencia/);
  assert.doesNotMatch(csv, /opinion_politica|comentario|postura/i);
});
