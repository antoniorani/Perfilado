/** Pure, side-effect-free evaluation of an LFW 1:N face-identification experiment. */

/** Pick a reproducible, geographically neutral sample from the entire LFW roster.
 * The base gallery has ONE real reference photograph per enrolled identity.
 * Additional photos and outsiders are hidden until this exercise.
 */
export function createCohort(profiles, enrolledCount = 8) {
  if (!Number.isInteger(enrolledCount) || enrolledCount < 2 || enrolledCount > 30) {
    throw new RangeError("El tamaño del experimento debe estar entre 2 y 30");
  }
  const eligible = profiles.filter((person) => person.imageCount > 1);
  const outsiders = profiles.filter((person) => person.imageCount === 1);
  if (eligible.length < enrolledCount || outsiders.length < Math.ceil(enrolledCount / 2)) {
    throw new RangeError("El catálogo de LFW no contiene suficientes identidades");
  }
  const select = (items, count) => Array.from({ length: count }, (_, index) =>
    items[Math.floor((index + 0.5) * items.length / count)]);

  const gallery = select(eligible, enrolledCount);
  const unknown = select(outsiders, Math.ceil(enrolledCount / 2));
  return {
    gallery: gallery.map((person) => ({ person, imageNumber: 1 })),
    probes: [
      ...gallery.map((person) => ({ person, imageNumber: 2, enrolled: true })),
      ...unknown.map((person) => ({ person, imageNumber: 1, enrolled: false })),
    ],
  };
}

/** Raw Euclidean distance in the 128-dimensional descriptor space, not a probability. */
export function euclideanDistance(left, right) {
  if (!left || !right || left.length !== right.length || left.length === 0) {
    throw new TypeError("Los embeddings deben tener la misma dimensión");
  }
  let sum = 0;
  for (let index = 0; index < left.length; index += 1) {
    if (!Number.isFinite(left[index]) || !Number.isFinite(right[index])) {
      throw new TypeError("Embeddings con valores no finitos");
    }
    const difference = left[index] - right[index];
    sum += difference * difference;
  }
  return Math.sqrt(sum);
}

/** Each stored item is { person, imageNumber, descriptor }. */
export function identify(probes, gallery, threshold = 0.6) {
  if (!Number.isFinite(threshold) || threshold < 0) throw new RangeError("Umbral inválido");
  if (gallery.length === 0) return [];
  return probes.map((probe) => {
    let nearest = null;
    let distance = Infinity;
    for (const reference of gallery) {
      const candidate = euclideanDistance(probe.descriptor, reference.descriptor);
      if (candidate < distance) {
        nearest = reference;
        distance = candidate;
      }
    }
    const accepted = distance <= threshold;
    const correct = probe.enrolled && accepted && nearest.person.id === probe.person.id;
    let outcome;
    if (probe.enrolled) {
      outcome = !accepted ? "missed" : correct ? "correct" : "wrong-identity";
    } else {
      outcome = accepted ? "false-alert" : "correct-rejection";
    }
    return {
      ...probe,
      nearest: nearest.person,
      distance,
      accepted,
      outcome,
    };
  });
}

export function evaluateMatches(matches) {
  const get = (key) => matches.filter((result) => result.outcome === key).length;
  const correct = get("correct");
  const wrongIdentity = get("wrong-identity");
  const missed = get("missed");
  const falseAlerts = get("false-alert");
  const correctRejections = get("correct-rejection");
  const enrolled = correct + wrongIdentity + missed;
  const unknown = falseAlerts + correctRejections;
  const matchesAccepted = correct + wrongIdentity + falseAlerts;
  return {
    total: matches.length, enrolled, unknown,
    correct, wrongIdentity, missed, falseAlerts, correctRejections,
    precision: matchesAccepted ? correct / matchesAccepted : null,
    recall: enrolled ? correct / enrolled : null,
    unknownFalsePositiveRate: unknown ? falseAlerts / unknown : null,
    globalAccuracy: matches.length ? (correct + correctRejections) / matches.length : null,
  };
}
