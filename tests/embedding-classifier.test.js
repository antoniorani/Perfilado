import test from "node:test";
import assert from "node:assert/strict";
import { trainEmbeddingClassifier, predictEmbedding } from "../src/embedding-classifier.js";

test("a real browser-compatible embedding head learns all three stance labels", () => {
  const samples = [
    [[1, 0, 0], "AGAINST"], [[.9,.1,0], "AGAINST"], [[.85,.15,0], "AGAINST"],
    [[0, 1, 0], "FAVOR"], [[.1,.9,0], "FAVOR"], [[.15,.85,0], "FAVOR"],
    [[0, 0, 1], "NONE"], [[0,.1,.9], "NONE"], [[.1,0,.9], "NONE"],
  ];
  const model = trainEmbeddingClassifier(samples.map(s=>Float32Array.from(s[0])),samples.map(s=>s[1]),{epochs:30});
  assert.equal(model.dimension,3);
  for(const [vector,label] of [[[.98,.02,0],"AGAINST"], [[0,.98,.02],"FAVOR"],[[.02,0,.98],"NONE"]]) {
    assert.equal(predictEmbedding(model,vector).label,label);
  }
});

test("semantic training refuses missing classes, incompatible and invalid embeddings", () => {
  assert.throws(()=>trainEmbeddingClassifier([[1,0],[0,1],[1,1]],["FAVOR","NONE","NONE"]),/Falta una clase/);
  assert.throws(()=>trainEmbeddingClassifier([[1,0],[0,1],[1,NaN]],["FAVOR","NONE","AGAINST"]),/no finito/);
  assert.throws(()=>trainEmbeddingClassifier([[1,0],[0,1],[1,0,0]],["FAVOR","NONE","AGAINST"]),/incompatible/);
  const model=trainEmbeddingClassifier([[1,0],[0,1],[-1,0]],["FAVOR","NONE","AGAINST"],{epochs:2});
  assert.throws(()=>predictEmbedding(model,[1]),/incompatible/);
});
