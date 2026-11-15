/**
 * Didactic, hypothetical error cascade. No real identity-to-post linkage.
 * Each percentage is a user-supplied illustrative marginal success rate.
 * The product is valid ONLY under the explicitly stated independence assumption.
 * Fréchet bounds show the range without such an assumption.
 */
export function combinedRisk(facePercent, linkPercent, textPercent) {
  const values = [facePercent, linkPercent, textPercent];
  if (values.some(value => typeof value !== "number" || !Number.isFinite(value)
      || value < 0 || value > 100)) {
    throw new RangeError("Las tasas deben ser números entre 0 y 100");
  }
  const [face, link, text] = values.map(value => value / 100);
  const simultaneous = face * link * text;
  return {
    simultaneous,
    atLeastOneFailure: 1 - simultaneous,
    lowerBound: Math.max(0, face + link + text - 2),
    upperBound: Math.min(face, link, text),
  };
}
