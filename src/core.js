/** Pure helpers for the student-facing, static dataset explorer. */
export const ZONES = ["Sector Alfa", "Sector Bravo", "Sector Charlie", "Sector Delta", "Sector Eco"];
export const LANGUAGES = ["Español", "Inglés", "Francés", "Ucraniano", "Ruso"];

function hash(input) {
  let value = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    value ^= input.charCodeAt(i);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

/** LFW identity and image count are factual; every profile field is invented. */
export function createProfiles(identities) {
  if (!Array.isArray(identities)) throw new TypeError("Expected LFW identities array");
  return identities.map(([identity, imageCount], index) => {
    if (typeof identity !== "string" || !Number.isInteger(imageCount) || imageCount < 1) {
      throw new TypeError("Invalid LFW identity entry");
    }
    const seed = hash(identity);
    return {
      id: `PF-${String(index + 1).padStart(5, "0")}`,
      lfwIdentity: identity,
      referenceFile: `${identity}_0001.jpg`,
      imageCount,
      age: 21 + (seed % 46),
      zone: ZONES[(seed >>> 4) % ZONES.length],
      language: LANGUAGES[(seed >>> 10) % LANGUAGES.length],
    };
  });
}

export function getSummary(profiles) {
  const count = (values, property) => values.reduce((totals, profile) => {
    const key = profile[property];
    totals[key] = (totals[key] || 0) + 1;
    return totals;
  }, {});
  const ageBands = { "21–30": 0, "31–40": 0, "41–50": 0, "51–60": 0, "61–66": 0 };
  for (const profile of profiles) {
    const band = profile.age <= 30 ? "21–30" : profile.age <= 40 ? "31–40"
      : profile.age <= 50 ? "41–50" : profile.age <= 60 ? "51–60" : "61–66";
    ageBands[band] += 1;
  }
  return {
    profiles: profiles.length,
    photographs: profiles.reduce((total, profile) => total + profile.imageCount, 0),
    references: profiles.length,
    repeatIdentities: profiles.filter((profile) => profile.imageCount > 1).length,
    byZone: count(profiles, "zone"),
    byLanguage: count(profiles, "language"),
    byAge: ageBands,
  };
}

export function filterProfiles(profiles, { query = "", zone = "all", order = "id" } = {}) {
  const needle = query.trim().toLocaleLowerCase("es");
  const result = profiles.filter((profile) => {
    return (zone === "all" || profile.zone === zone)
      && (!needle || profile.id.toLocaleLowerCase("es").includes(needle));
  });
  return result.sort((a, b) => {
    if (order === "age") return a.age - b.age || a.id.localeCompare(b.id);
    if (order === "zone") return a.zone.localeCompare(b.zone, "es") || a.id.localeCompare(b.id);
    return a.id.localeCompare(b.id);
  });
}

export function exportProfilesCsv(profiles) {
  const columns = ["perfil_id", "imagen_referencia", "edad_sintetica", "ubicacion_sintetica", "idioma_sintetico"];
  const quote = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const rows = profiles.map((p) => [p.id, p.referenceFile, p.age, p.zone, p.language].map(quote).join(","));
  return "\uFEFF" + [columns.join(","), ...rows].join("\r\n");
}
