/**
 * RUW: official 6,000-post, two-target stance dataset.
 * Source texts remain in the original RUW repository, never committed here.
 */
export const RUW_REVISION = "ecb2a4579197c5be03092a1b8fa21ead359809ce";
export const RUW_BASE = `https://raw.githubusercontent.com/multi-target/RUW/${RUW_REVISION}`;
export const RUW_FILES = Object.freeze({
  train: "raw_train_Russia_Ukraine.csv",
  val: "raw_val_Russia_Ukraine.csv",
  test: "raw_test_Russia_Ukraine.csv",
});
export const LABELS = Object.freeze(["AGAINST", "FAVOR", "NONE"]);
export const LABEL_NAMES = Object.freeze({
  AGAINST: "En contra",
  FAVOR: "A favor",
  NONE: "Sin postura clara",
});
export const TARGETS = Object.freeze({
  russia: "Rusia",
  ukraine: "Ucrania",
});

/** RFC-4180-compatible CSV parser, including quoted newlines and escaped quotes. */
export function parseCSV(source) {
  if (typeof source !== "string") throw new TypeError("CSV no válido");
  const text = source.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [], field = "", quoted = false, afterQuote = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
        afterQuote = true;
      } else field += char;
    } else if (char === '"') {
      if (field !== "" || afterQuote) throw new Error("Comillas inesperadas en CSV");
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = ""; afterQuote = false;
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = []; field = ""; afterQuote = false;
    } else if (char === "\r" && text[index + 1] === "\n") {
      // CRLF outside of a quoted field.
    } else {
      if (afterQuote && char.trim() !== "") throw new Error("Texto después de cierre de comillas");
      field += char;
    }
  }
  if (quoted) throw new Error("CSV con comillas sin cerrar");
  if (row.length || field.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

/**
 * Hide handles, emails and URLs when displaying/training on third-party posts.
 * This is data minimization, not a claim of full anonymization.
 */
export function redactTweet(text) {
  return String(text)
    .replace(/\bhttps?:\/\/[^\s<>"']+|\bwww\.[^\s<>"']+/giu, " [enlace] ")
    .replace(/\b[\p{L}\p{N}._%+-]+@[\p{L}\p{N}.-]+\.[\p{L}]{2,}\b/giu, " [correo] ")
    .replace(/(^|[^\p{L}\p{N}])@[\p{L}\p{N}_]{1,30}/gu, "$1[usuario]")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s+/g, " ").trim();
}

export function normalizedText(text) {
  return redactTweet(text).normalize("NFKC").toLocaleLowerCase("en").trim();
}

export function parseRUWSplit(source, split) {
  if (!Object.hasOwn(RUW_FILES, split)) throw new Error("Partición RUW desconocida");
  const rows = parseCSV(source);
  const expected = ["Tweet", "Target 1", "Stance 1", "Target 2", "Stance 2", "Test/Train/Dev"];
  if (rows.length < 2 || JSON.stringify(rows[0]) !== JSON.stringify(expected)) {
    throw new Error(`Encabezados inesperados en RUW (${split})`);
  }
  const expectedSplit = { train: "Train", val: "Dev", test: "Test" }[split];
  return rows.slice(1).filter((r) => r.length === 6 && r.some(Boolean)).map((fields, index) => {
    const [rawText, rTarget, russia, uTarget, ukraine, originalSplit] = fields;
    if (rTarget !== "Russia" || uTarget !== "Ukraine"
        || originalSplit !== expectedSplit || !LABELS.includes(russia)
        || !LABELS.includes(ukraine)) {
      throw new Error(`Etiqueta, objetivo o partición no válida en RUW ${split}, fila ${index + 2}`);
    }
    return {
      id: `RUW-${split.toUpperCase()}-${String(index + 1).padStart(4, "0")}`,
      text: redactTweet(rawText),
      russia, ukraine, split,
    };
  });
}

/** 
 * Keep the original RUW train/validation/test split. Remove duplicate *texts* to
 * avoid optimistic held-out scores: test > validation > train takes precedence.
 * No stance labels from validation/test are used for feature selection/training.
 */
export function prepareRUW(data) {
  for (const key of Object.keys(RUW_FILES)) if (!Array.isArray(data[key])) {
    throw new TypeError(`Falta la partición ${key}`);
  }
  const seen = new Set(), removed = { train: 0, val: 0, test: 0 };
  const clean = {};
  for (const split of ["test", "val", "train"]) {
    const result = [];
    for (const row of data[split]) {
      const signature = normalizedText(row.text);
      if (!signature || seen.has(signature)) {
        removed[split] += 1;
        continue;
      }
      seen.add(signature);
      result.push(row);
    }
    clean[split] = result;
  }
  return {
    ...clean,
    originalCounts: Object.fromEntries(Object.keys(RUW_FILES).map(key=>[key,data[key].length])),
    removed,
  };
}

export function labelCounts(rows, target) {
  if (!Object.hasOwn(TARGETS, target)) throw new Error("Objetivo desconocido");
  const counts = Object.fromEntries(LABELS.map(label=>[label,0]));
  for (const row of rows) counts[row[target]] += 1;
  return counts;
}

export async function loadRUW(fetcher = fetch) {
  const splits = {};
  await Promise.all(Object.entries(RUW_FILES).map(async ([split, name]) => {
    const response = await fetcher(`${RUW_BASE}/${name}`);
    if (!response.ok) throw new Error(`No se ha podido descargar RUW ${split} (HTTP ${response.status})`);
    splits[split] = parseRUWSplit(await response.text(), split);
  }));
  if (splits.train.length !== 4200 || splits.val.length !== 600 || splits.test.length !== 1200) {
    throw new Error("El corpus RUW descargado no tiene el tamaño esperado (4200/600/1200)");
  }
  return prepareRUW(splits);
}
