/** Single, versioned face pipeline shared by Node precomputation and the web. 
 * Do not change detector, alignment or weights without bumping PIPELINE_ID.
 */
export const FACE_API_VERSION = "0.22.2";
export const PIPELINE_ID = "faceapi0222-ssd68-aligned-v2";
export const MODEL_MANIFESTS = Object.freeze([
  "ssd_mobilenetv1_model-weights_manifest.json",
  "face_landmark_68_model-weights_manifest.json",
  "face_recognition_model-weights_manifest.json",
]);
export const DESCRIPTOR_LENGTH = 128;
export const DETECTION_MIN_CONFIDENCE = 0.3;

/** Same exact API call for offline precomputation and browser queries. */
export async function computeAlignedDescriptor(faceapi, image) {
  const result = await faceapi.detectSingleFace(
    image,
    new faceapi.SsdMobilenetv1Options({ minConfidence: DETECTION_MIN_CONFIDENCE }),
  ).withFaceLandmarks().withFaceDescriptor();
  if (!result?.descriptor || result.descriptor.length !== DESCRIPTOR_LENGTH
      || !Array.from(result.descriptor).every(Number.isFinite)) {
    throw new Error("No se detectó un rostro válido o no se pudo calcular su embedding.");
  }
  return result.descriptor;
}
