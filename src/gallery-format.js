import { DESCRIPTOR_LENGTH } from "./face-pipeline.js";
export const GALLERY_VERSION = "faceapi0222-fallback68-v3";

const HEADER_BYTES = 64;
const MAGIC = "PFLFW2\0\0";
const VERSION_OFFSET = 16;
const MAX_VERSION_BYTES = 32;

/** Light integrity fingerprint of ordered names and counts; not cryptographic. */
export function catalogFingerprint(profiles) {
  let hash = 2166136261;
  for (const item of profiles) {
    const entry = `${item.lfwIdentity}:${item.imageCount}\n`;
    for (let i = 0; i < entry.length; i += 1) {
      hash ^= entry.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
  }
  return hash >>> 0;
}

/** Binary embeds pipeline ID, catalog checksum, shape, and Float32 descriptors. */
export function encodeGallery(values, profiles) {
  if (!(values instanceof Float32Array) || values.length !== profiles.length * DESCRIPTOR_LENGTH) {
    throw new Error("Invalid descriptor matrix size");
  }
  if (GALLERY_VERSION.length > MAX_VERSION_BYTES) throw new Error("Pipeline ID exceeds header capacity");
  const buffer = new ArrayBuffer(HEADER_BYTES + values.byteLength);
  const header = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < MAGIC.length; i += 1) bytes[i] = MAGIC.charCodeAt(i);
  header.setUint32(8, profiles.length, true);
  header.setUint32(12, DESCRIPTOR_LENGTH, true);
  for (let i = 0; i < GALLERY_VERSION.length; i += 1) bytes[VERSION_OFFSET + i] = GALLERY_VERSION.charCodeAt(i);
  header.setUint32(48, catalogFingerprint(profiles), true);
  new Float32Array(buffer, HEADER_BYTES).set(values);
  return buffer;
}

/** Fails closed for legacy unaligned matrices, wrong model or wrong catalog. */
export function decodeGallery(buffer, profiles) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < HEADER_BYTES) {
    throw new Error("El archivo de embeddings es inválido o incompleto.");
  }
  const bytes = new Uint8Array(buffer);
  if ([...MAGIC].some((c, i) => bytes[i] !== c.charCodeAt(0))) {
    throw new Error("Galería antigua o incompatible. Necesitas los nuevos embeddings v3.");
  }
  const header = new DataView(buffer);
  const rows = header.getUint32(8, true);
  const width = header.getUint32(12, true);
  const version = Array.from(bytes.subarray(VERSION_OFFSET, VERSION_OFFSET + MAX_VERSION_BYTES))
    .map((b) => String.fromCharCode(b)).join("").replace(/\0+$/, "");
  const fingerprint = header.getUint32(48, true);
  if (version !== GALLERY_VERSION) throw new Error("La galería se calculó con otro modelo o preprocesamiento.");
  if (rows !== profiles.length || width !== DESCRIPTOR_LENGTH
      || fingerprint !== catalogFingerprint(profiles)) {
    throw new Error("La galería no corresponde a la lista de identidades actual.");
  }
  if (buffer.byteLength !== HEADER_BYTES + rows * width * 4) {
    throw new Error("Galería truncada o tamaño incompatible.");
  }
  return new Float32Array(buffer, HEADER_BYTES, rows * width);
}
