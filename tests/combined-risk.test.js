import test from "node:test";
import assert from "node:assert/strict";
import { combinedRisk } from "../src/combined-risk.js";

test("illustrative cascade applies the independence assumption only to its product", () => {
  const stats = combinedRisk(80, 70, 53);
  assert.ok(Math.abs(stats.simultaneous - 0.2968) < 1e-12);
  assert.ok(Math.abs(stats.atLeastOneFailure - 0.7032) < 1e-12);
  assert.ok(Math.abs(stats.lowerBound - 0.03) < 1e-12);
  assert.equal(stats.upperBound, 0.53);
});

test("simultaneous correct results cannot exceed the least accurate stage", () => {
  for (const values of [[33, 56, 87], [100, 100, 99], [0, 77, 90], [10, 20, 30]]) {
    const result = combinedRisk(...values);
    assert.ok(result.simultaneous <= result.upperBound + 1e-12);
    assert.ok(result.simultaneous >= result.lowerBound - 1e-12);
    assert.ok(result.atLeastOneFailure >= 0 && result.atLeastOneFailure <= 1);
  }
});

test("boundary cases and percentages are explicit", () => {
  assert.deepEqual(combinedRisk(100, 100, 100), {
    simultaneous: 1, atLeastOneFailure: 0, lowerBound: 1, upperBound: 1,
  });
  assert.deepEqual(combinedRisk(0, 100, 100), {
    simultaneous: 0, atLeastOneFailure: 1, lowerBound: 0, upperBound: 0,
  });
});

test("rejects missing, text, non-finite, and out-of-range rates", () => {
  for (const values of [[null, 80, 90], ["80", 60, 70], [80, undefined, 10],
    [NaN, 80, 90], [Infinity, 80, 90], [-1, 20, 30], [30, 101, 40]]) {
    assert.throws(() => combinedRisk(...values), RangeError);
  }
});
