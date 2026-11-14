/** Pure helpers for the student-facing, static dataset explorer. */
export const ZONES = ["Sector Alfa", "Sector Bravo", "Sector Charlie", "Sector Delta", "Sector Eco"];
export const LANGUAGES = ["Español", "Inglés", "Francés", "Ucraniano", "Ruso"];
// Deliberately uneven, purely invented distribution (approx. 52/26/13/7/2%).
const ZONE_CUMULATIVE = [5200, 7800, 9100, 9800, 10000];
const PHOTO_SOURCE = "https://huggingface.co/datasets/marcelohaps/lfw/resolve/main/train/images";

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
  let photoOffset = 0;
  return identities.map(([identity, imageCount], index) => {
    if (typeof identity !== "string" || !Number.isInteger(imageCount) || imageCount < 1) {
      throw new TypeError("Invalid LFW identity entry");
    }
    const seed = hash(identity);
    const zoneDraw = (seed >>> 4) % 10000;
    const zoneIndex = ZONE_CUMULATIVE.findIndex((cutoff) => zoneDraw < cutoff);
    const imageOffset = photoOffset;
    const referenceShard = String(Math.floor(imageOffset / 1000)).padStart(3, "0");
    photoOffset += imageCount;
    return {
      id: `PF-${String(index + 1).padStart(5, "0")}`,
      lfwIdentity: identity,
      name: identity.replaceAll("_", " "),
      referenceFile: `${identity}_0001.jpg`,
      referenceShard,
      imageOffset,
      imageCount,
      age: 21 + (seed % 46),
      zone: ZONES[zoneIndex],
      language: LANGUAGES[(seed >>> 10) % LANGUAGES.length],
    };
  });
}

/** Lazily display the original first image from a public LFW mirror; not rehosted. */
export function photoUrl(profile, number = 1) {
  if (!Number.isInteger(number) || number < 1 || number > profile.imageCount) {
    throw new RangeError("Número de imagen fuera del catálogo de esta identidad");
  }
  const shard = String(Math.floor((profile.imageOffset + number - 1) / 1000)).padStart(3, "0");
  const filename = `${profile.lfwIdentity}_${String(number).padStart(4, "0")}.jpg`;
  return `${PHOTO_SOURCE}/${shard}/${encodeURIComponent(filename)}`;
}

export function referenceImageUrl(profile) {
  return photoUrl(profile, 1);
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
      && (!needle || `${profile.id} ${profile.name}`.toLocaleLowerCase("es").includes(needle));
  });
  return result.sort((a, b) => {
    if (order === "age") return a.age - b.age || a.id.localeCompare(b.id);
    if (order === "zone") return a.zone.localeCompare(b.zone, "es") || a.id.localeCompare(b.id);
    return a.id.localeCompare(b.id);
  });
}
