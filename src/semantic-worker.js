/**
 * On-device RUW sentence embeddings + in-session classifier training.
 * No texts are sent to an inference service. The browser downloads a public
 * pretrained encoder, then runs ONNX locally (WASM). Texts/vectors stay in RAM.
 */
import { trainEmbeddingClassifier, predictEmbedding } from "./embedding-classifier.js";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const ENCODER = "Xenova/all-MiniLM-L6-v2";
const BATCH_SIZE = 12;
let extractorPromise;
let classifier;
let activeTarget;
let busy = false;
const cache = new Map();

function progress(id, message) {
  self.postMessage({ id, type: "progress", message });
}

async function getExtractor(id) {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      progress(id, "Descargando el encoder MiniLM y preparando ONNX en el navegador…");
      const { pipeline } = await import(TRANSFORMERS_URL);
      return pipeline("feature-extraction", ENCODER, { device: "wasm" });
    })().catch(error => { extractorPromise = undefined; throw error; });
  }
  return extractorPromise;
}

async function encode(texts, id, phase, { remember = true } = {}) {
  const extractor = await getExtractor(id);
  const pending = [...new Set(texts.filter(text => !cache.has(text)))];
  for (let i = 0; i < pending.length; i += BATCH_SIZE) {
    const batch = pending.slice(i, i + BATCH_SIZE);
    const output = await extractor(batch, { pooling: "mean", normalize: true });
    const dimension = output.dims.at(-1);
    if (dimension !== 384 || output.data.length !== batch.length * dimension) {
      throw new Error("Dimensiones del encoder semántico inesperadas");
    }
    for (let n = 0; n < batch.length; n++) {
      const vector = Float32Array.from(output.data.subarray(n * dimension, (n + 1) * dimension));
      cache.set(batch[n], vector); // Temporary for order-preserving output assembly.
    }
    output.dispose?.();
    progress(id, `${phase}: ${Math.min(i + batch.length, pending.length)} de ${pending.length} embeddings nuevos…`);
  }
  const vectors = texts.map(text => cache.get(text));
  if (!remember) for (const text of pending) cache.delete(text);
  return vectors;
}

self.onmessage = async ({data}) => {
  const { id, type } = data;
  if (busy) {
    self.postMessage({id, type:"error", message:"Hay otra operación de embeddings en curso"});
    return;
  }
  busy = true;
  try {
    if (type === "train") {
      const {train, val, target} = data;
      if (!["russia", "ukraine"].includes(target) || !Array.isArray(train) || !Array.isArray(val)) {
        throw new Error("Datos de entrenamiento incompatibles");
      }
      const trainVectors = await encode(train.map(row => row.text), id, "Entrenamiento");
      const valVectors = await encode(val, id, "Validación");
      progress(id, "Ajustando los pesos de una regresión logística sobre los embeddings…");
      const next = trainEmbeddingClassifier(trainVectors, train.map(row => row.label));
      const predictions = valVectors.map(vector => predictEmbedding(next, vector).label);
      classifier = next;
      activeTarget = target;
      self.postMessage({id, type:"result", result:{predictions, dimension:next.dimension}});
    } else if (type === "test") {
      if (!classifier || activeTarget !== data.target) throw new Error("Entrena primero el modelo semántico");
      const vectors = await encode(data.texts, id, "Prueba reservada");
      self.postMessage({id, type:"result", result:{predictions:vectors.map(v => predictEmbedding(classifier, v).label)}});
    } else if (type === "manual") {
      if (!classifier || activeTarget !== data.target) throw new Error("Entrena primero el modelo semántico");
      const [vector] = await encode([data.text], id, "Texto de prueba", {remember:false});
      self.postMessage({id, type:"result", result:predictEmbedding(classifier, vector)});
    } else {
      throw new Error("Operación semántica desconocida");
    }
  } catch (error) {
    self.postMessage({id, type:"error", message:error?.message || String(error)});
  } finally {
    busy = false;
  }
};
