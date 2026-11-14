/** Real browser face descriptors from LFW crops, identical to the offline generator. */
const MODEL_ROOT = "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights";
let modelPromise;

/** Fetch pretrained models only after a user explicitly launches the exercise. */
export function ensureModels() {
  if (!modelPromise) {
    modelPromise = (async () => {
      const api = window.faceapi;
      if (!api?.nets?.faceRecognitionNet) {
        throw new Error("No se ha podido cargar la biblioteca face-api.js desde el CDN.");
      }
      await api.nets.faceRecognitionNet.loadFromUri(MODEL_ROOT);
      return api;
    })().catch((error) => {
      modelPromise = undefined; // permit retry after transient network errors
      throw error;
    });
  }
  return modelPromise;
}

/** HTMLImageElement with anonymous CORS: failures are explicit, never replaced with fake data. */
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.referrerPolicy = "no-referrer";
    const timeout = setTimeout(() => reject(new Error("La imagen ha tardado demasiado en cargar.")), 25000);
    image.onload = () => { clearTimeout(timeout); resolve(image); };
    image.onerror = () => {
      clearTimeout(timeout);
      reject(new Error("No se pudo leer la imagen de LFW (conexión o restricciones CORS)."));
    };
    image.src = url;
  });
}

/** A real 128-dimensional descriptor of the pre-cropped LFW photo. */
export async function extractDescriptor(url) {
  const api = await ensureModels();
  const image = await loadImage(url);
  try {
    const result = await api.computeFaceDescriptor(image);
    if (!result || result.length !== 128 || !Array.from(result).every(Number.isFinite)) {
      throw new Error("El modelo no ha podido calcular un embedding válido.");
    }
    return Array.from(result);
  } finally {
    image.src = ""; // release HTML image memory
  }
}
