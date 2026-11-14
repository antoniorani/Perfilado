/** Pure logic for the 5,749-person gallery / 100-photo LFW experiment. */
export const DESCRIPTOR_SIZE = 128;
export const DEFAULT_PROBES = 100;

export function chooseProbePhotos(profiles, requested = DEFAULT_PROBES) {
  if (!Number.isInteger(requested) || requested < 1) throw new RangeError("Número de pruebas inválido");
  const eligible = profiles.filter((person) => person.imageCount >= 2);
  if (eligible.length < requested) throw new RangeError("No hay suficientes identidades con fotos adicionales");
  return Array.from({ length: requested }, (_, index) => ({
    person: eligible[Math.floor((index + 0.5) * eligible.length / requested)],
    imageNumber: 2,
  }));
}

/** The Float32 file is ordered identically to data/lfw-identities.json. */
export function readGallery(buffer, profiles) {
  const bytes = profiles.length * DESCRIPTOR_SIZE * Float32Array.BYTES_PER_ELEMENT;
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== bytes) {
    throw new Error(`Galería biométrica incompatible: se esperaban ${bytes} bytes`);
  }
  const values = new Float32Array(buffer);
  const missing = [];
  const gallery = profiles.map((person, index) => {
    const descriptor = values.subarray(index * DESCRIPTOR_SIZE, (index + 1) * DESCRIPTOR_SIZE);
    if (!descriptor.every(Number.isFinite)) {
      missing.push(person.id);
      return null;
    }
    return { person, descriptor };
  }).filter(Boolean);
  if (gallery.length < 2) throw new Error("La galería no contiene suficientes embeddings válidos");
  return { gallery, missing };
}

/** Full 1:N nearest-neighbor search. Distances are Euclidean, NOT probabilities. */
export function nearestCandidates(probe, gallery, count = 5) {
  if (!probe || probe.length !== DESCRIPTOR_SIZE || !probe.every(Number.isFinite)) {
    throw new Error("Embedding de prueba inválido");
  }
  if (!Number.isInteger(count) || count < 1) throw new RangeError("Número de candidatos inválido");
  const best = [];
  for (const reference of gallery) {
    const desc = reference.descriptor;
    if (!desc || desc.length !== DESCRIPTOR_SIZE) throw new Error("Galería corrupta");
    let sum = 0;
    for (let index = 0; index < DESCRIPTOR_SIZE; index += 1) {
      const diff = probe[index] - desc[index];
      sum += diff * diff;
    }
    if (best.length < count || sum < best[best.length - 1].squared) {
      const next = { person: reference.person, squared: sum };
      const at = best.findIndex((item) => item.squared > sum);
      best.splice(at < 0 ? best.length : at, 0, next);
      if (best.length > count) best.pop();
    }
  }
  return best.map((item) => ({ person: item.person, distance: Math.sqrt(item.squared) }));
}

export function classifyProbe(probe, candidates, threshold) {
  if (!Number.isFinite(threshold) || threshold < 0) throw new RangeError("Umbral inválido");
  if (!candidates.length) throw new Error("La búsqueda necesita candidatos");
  const nearest = candidates[0];
  const top1Correct = nearest.person.id === probe.person.id;
  const accepted = nearest.distance <= threshold;
  return {
    ...probe, candidates,
    nearest, top1Correct, accepted,
    outcome: !accepted ? "rejected" : top1Correct ? "correct" : "wrong",
  };
}

/** Top-1 is computed independently of the acceptance threshold. */
export function summariseProbes(classified, failures = 0) {
  const count = (outcome) => classified.filter((probe) => probe.outcome === outcome).length;
  const correct = count("correct"), wrong = count("wrong"), rejected = count("rejected");
  const top1 = classified.filter((probe) => probe.top1Correct).length;
  const processed = classified.length;
  return {
    attempted: processed + failures, processed, failures, top1,
    top1Rate: processed ? top1 / processed : null,
    correct, wrong, rejected,
    accepted: correct + wrong,
    accuracyAfterRejection: processed ? correct / processed : null,
  };
}
