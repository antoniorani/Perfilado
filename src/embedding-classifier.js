/**
 * Browser-trained supervised head for L2-normalized sentence embeddings.
 * The encoder is pretrained, but these coefficients are learned exclusively
 * from the RUW train partition in this browser session.
 */
import { LABELS } from "./ruw-data.js";

const CLASS_INDEX = new Map(LABELS.map((label, i) => [label, i]));

export function predictEmbedding(model, vector) {
  if (!vector || vector.length !== model.dimension) throw new Error("Embedding incompatible");
  const scores = Array.from(model.bias);
  for (let c = 0; c < LABELS.length; c++) {
    const offset = c * model.dimension;
    for (let j = 0; j < model.dimension; j++) scores[c] += model.weights[offset + j] * vector[j];
  }
  let best = 0;
  for (let c = 1; c < scores.length; c++) if (scores[c] > scores[best]) best = c;
  return { label: LABELS[best], scores, knownTerms: model.dimension };
}

/** Train a 3-way softmax classifier with deterministic mini-batch Adam. */
export function trainEmbeddingClassifier(vectors, labels, {
  epochs = 35, batchSize = 64, learningRate = 0.02, regularization = 0.001,
} = {}) {
  if (!Array.isArray(vectors) || vectors.length !== labels?.length || vectors.length < 3) {
    throw new Error("Embeddings de entrenamiento inválidos");
  }
  const dimension = vectors[0]?.length;
  if (!Number.isInteger(dimension) || dimension < 2 || dimension > 4096
    || ![epochs, batchSize].every(n => Number.isInteger(n) && n >= 1 && n <= 300)
    || !Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1
    || !Number.isFinite(regularization) || regularization < 0 || regularization > 1) {
    throw new RangeError("Configuración de entrenamiento inválida");
  }
  const counts = new Int32Array(LABELS.length);
  const targets = labels.map(label => {
    const index = CLASS_INDEX.get(label);
    if (index === undefined) throw new Error("Etiqueta de entrenamiento desconocida");
    counts[index]++;
    return index;
  });
  if (counts.some(value => value === 0)) throw new Error("Falta una clase en entrenamiento");
  for (const vector of vectors) {
    if (vector?.length !== dimension || Array.from(vector).some(v => !Number.isFinite(v))) {
      throw new Error("Embedding incompatible o no finito");
    }
  }

  const size = dimension * LABELS.length;
  const weights = new Float32Array(size);
  const bias = new Float32Array(LABELS.length);
  const m = new Float64Array(size + LABELS.length);
  const v = new Float64Array(size + LABELS.length);
  const gradients = new Float64Array(m.length);
  const logits = new Float64Array(LABELS.length);
  const probabilities = new Float64Array(LABELS.length);
  const order = Uint32Array.from({length: vectors.length}, (_, i) => i);
  let seed = 0x42c0ffee, step = 0;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (let start = 0; start < order.length; start += batchSize) {
      gradients.fill(0);
      const end = Math.min(start + batchSize, order.length);
      for (let i = start; i < end; i++) {
        const index = order[i], vector = vectors[index];
        let max = -Infinity;
        for (let c = 0; c < LABELS.length; c++) {
          let score = bias[c], offset = c * dimension;
          for (let j = 0; j < dimension; j++) score += weights[offset + j] * vector[j];
          logits[c] = score;
          max = Math.max(max, score);
        }
        let denominator = 0;
        for (let c = 0; c < LABELS.length; c++) {
          probabilities[c] = Math.exp(logits[c] - max);
          denominator += probabilities[c];
        }
        for (let c = 0; c < LABELS.length; c++) {
          const error = probabilities[c] / denominator - Number(targets[index] === c);
          const offset = c * dimension;
          for (let j = 0; j < dimension; j++) gradients[offset + j] += error * vector[j];
          gradients[size + c] += error;
        }
      }
      step++;
      const inv = 1 / (end - start);
      const correctedM = 1 - Math.pow(0.9, step), correctedV = 1 - Math.pow(0.999, step);
      for (let i = 0; i < m.length; i++) {
        const isBias = i >= size;
        const weight = isBias ? bias[i - size] : weights[i];
        const gradient = gradients[i] * inv + (isBias ? 0 : regularization * weight);
        m[i] = 0.9 * m[i] + 0.1 * gradient;
        v[i] = 0.999 * v[i] + 0.001 * gradient * gradient;
        const next = weight - learningRate * (m[i] / correctedM) / (Math.sqrt(v[i] / correctedV) + 1e-8);
        if (isBias) bias[i - size] = next;
        else weights[i] = next;
      }
    }
  }
  return { dimension, weights, bias };
}
