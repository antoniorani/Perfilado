import test from "node:test";
import assert from "node:assert/strict";
import { combinedRisk } from "../src/combined-risk.js";

test("two-stage scenario uses the independence assumption explicitly", () => {
  const result = combinedRisk(80, 53);
  assert.ok(Math.abs(result.simultaneous - 0.424) < 1e-12);
  assert.ok(Math.abs(result.atLeastOneFailure - 0.576) < 1e-12);
  assert.ok(Math.abs(result.lowerBound - 0.33) < 1e-12);
  assert.ok(Math.abs(result.upperBound - 0.53) < 1e-12);
});

test("bounds bracket the intersection regardless of marginals", () => {
  for (const [face, text] of [[33, 56], [100, 99], [0, 77], [10, 20], [100, 100]]) {
    const {simultaneous,lowerBound,upperBound} = combinedRisk(face, text);
    assert.ok(simultaneous <= upperBound + 1e-12);
    assert.ok(simultaneous >= lowerBound - 1e-12);
  }
});

test("boundary cases are unambiguous", () => {
  assert.deepEqual(combinedRisk(100, 100), {
    simultaneous: 1, atLeastOneFailure: 0, lowerBound: 1, upperBound: 1,
  });
  assert.deepEqual(combinedRisk(0, 100), {
    simultaneous: 0, atLeastOneFailure: 1, lowerBound: 0, upperBound: 0,
  });
});

test("rejects invalid marginal rates", () => {
  for (const values of [[null, 80], ["80", 60], [80, undefined], [NaN, 80],
    [Infinity, 80], [-1, 20], [30, 101]]) {
    assert.throws(() => combinedRisk(...values), RangeError);
  }
});
