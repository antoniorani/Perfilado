/**
 * Illustrative intersection of two correct stages in a classroom exercise.
 * This assumes a fictional face-to-comment association, never a real identity link.
 * The product is a scenario only, valid under independence; the bounds
 * hold for any dependence structure if the marginal rates are comparable.
 */
export function combinedRisk(facePercent, textPercent) {
  const values = [facePercent, textPercent];
  if (values.some(value => typeof value !== "number" || !Number.isFinite(value)
      || value < 0 || value > 100)) {
    throw new RangeError("Las tasas deben ser números entre 0 y 100");
  }
  const [face, text] = values.map(value => value / 100);
  const simultaneous = face * text;
  return {
    simultaneous,
    atLeastOneFailure: 1 - simultaneous,
    lowerBound: Math.max(0, face + text - 1),
    upperBound: Math.min(face, text),
  };
}
