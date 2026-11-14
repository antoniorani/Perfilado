/** Browser-side detector, landmark alignment and descriptor extraction. */
import { computeAlignedDescriptor, FACE_API_VERSION } from "./face-pipeline.js";

const MODEL_ROOT =
  `https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@${FACE_API_VERSION}/weights`;
let modelPromise;

export function ensureModels() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const api = window.faceapi;
      if (!api?.nets?.ssdMobilenetv1 || !api?.nets?.faceLandmark68Net
          || !api?.nets?.faceRecognitionNet) {
        throw new Error("No se ha podido cargar face-api.js desde el proveedor.");
      }
      await Promise.all([
        api.nets.ssdMobilenetv1.loadFromUri(MODEL_ROOT),
        api.nets.faceLandmark68Net.loadFromUri(MODEL_ROOT),
        api.nets.faceRecognitionNet.loadFromUri(MODEL_ROOT),
      ]);
      return api;
    })().catch((error) => {
      modelPromise = undefined;
      throw error;
    });
  }
  return modelPromise;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.referrerPolicy = "no-referrer";
    let finished = false;
    const done = (error) => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      if (error) { image.src = ""; reject(error); }
      else resolve(image);
    };
    const timeout = setTimeout(() =>
      done(new Error("La carga de la fotografía excedió 30 segundos.")), 30000);
    image.onload = () => done();
    image.onerror = () =>
      done(new Error("No se pudo leer la fotografía de LFW (red o CORS)."));
    image.src = url;
  });
}

export async function extractDescriptor(url) {
  const api = await ensureModels();
  const image = await loadImage(url);
  try {
    const descriptor = await computeAlignedDescriptor(api, image);
    return Array.from(descriptor);
  } finally {
    image.src = "";
  }
}
