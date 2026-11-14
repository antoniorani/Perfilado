import { createProfiles, photoUrl, getSummary } from "./core.js";
import { createCohort, identify, evaluateMatches } from "./biometric-core.js";
import { ensureModels, extractDescriptor } from "./face-engine.js";
import { setupAnswers } from "./answers.js";

const $ = (selector) => document.querySelector(selector);
const formatter = new Intl.NumberFormat("es-ES", { maximumFractionDigits: 1 });
const elements = {
  size: $("#cohort-size"), galleryCount: $("#gallery-count"), probeCount: $("#probe-count"),
  run: $("#run-model"), status: $("#model-status"), description: $("#model-description"),
  progress: $("#model-progress"), preview: $("#preview-gallery"), results: $("#results-section"),
  threshold: $("#distance-threshold"), thresholdValue: $("#threshold-value"),
  cases: $("#cases"), caseFilter: $("#case-filter"), caseSummary: $("#case-summary"),
  precision: $("#precision-text"), scope: $("#metrics-scope"),
};

let profiles = [];
let cohort;
let processed = null;
const embeddingCache = new Map();

function imageElement(person, imageNumber, className = "face-preview") {
  const img = document.createElement("img");
  img.src = photoUrl(person, imageNumber);
  img.alt = `LFW: ${person.name}, imagen ${imageNumber}`;
  img.referrerPolicy = "no-referrer";
  img.loading = "lazy";
  img.decoding = "async";
  img.className = className;
  img.addEventListener("error", () => {
    const placeholder = document.createElement("span");
    placeholder.className = `${className} image-missing`;
    placeholder.setAttribute("role", "img");
    placeholder.setAttribute("aria-label", "Fotografía no disponible");
    placeholder.textContent = "Foto no disponible";
    img.replaceWith(placeholder);
  }, { once: true });
  return img;
}

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function previewSample() {
  elements.galleryCount.textContent = `${cohort.gallery.length} referencias`;
  elements.probeCount.textContent = `${cohort.probes.length} fotografías`;
  const fragment = document.createDocumentFragment();
  for (const probe of cohort.probes.slice(0, 3)) {
    const card = createElement("article", "preview-card");
    const photos = createElement("div", "preview-pair");
    const reference = cohort.gallery.find((entry) => entry.person.id === probe.person.id);
    const left = reference || cohort.gallery[0];
    for (const [entry, label] of [[left, "REFERENCIA"], [probe, "NUEVA IMAGEN"]]) {
      const item = createElement("figure", "preview-photo");
      item.append(imageElement(entry.person, entry.imageNumber));
      item.append(createElement("figcaption", "", label));
      photos.append(item);
    }
    card.append(photos, createElement("p", "preview-card-caption",
      probe.enrolled ? "La identidad está en la galería." : "¿Está esta persona en la galería?"));
    fragment.append(card);
  }
  elements.preview.replaceChildren(fragment);
}

function configureCohort() {
  cohort = createCohort(profiles, Number(elements.size.value));
  processed = null;
  elements.results.hidden = true;
  elements.status.textContent = "Experimento preparado";
  elements.description.textContent =
    "Pulsa «Calcular embeddings». El navegador descargará el modelo y procesará la muestra seleccionada.";
  elements.progress.hidden = true;
  previewSample();
}

function setProgress(current, total, phase) {
  elements.status.textContent = phase;
  elements.description.textContent = `Imágenes examinadas: ${current} de ${total} (la detección puede fallar en algunas).`;
  elements.progress.hidden = false;
  elements.progress.value = Math.round(100 * current / total);
}

async function descriptor(entry) {
  const url = photoUrl(entry.person, entry.imageNumber);
  if (!embeddingCache.has(url)) {
    embeddingCache.set(url, extractDescriptor(url).catch((error) => {
      embeddingCache.delete(url);
      throw error;
    }));
  }
  return embeddingCache.get(url);
}

async function runExperiment() {
  elements.run.disabled = true;
  elements.size.disabled = true;
  elements.results.hidden = true;
  elements.progress.hidden = false;
  elements.progress.value = 0;
  elements.status.textContent = "Cargando los modelos de reconocimiento facial…";
  elements.description.textContent = "Se utiliza face-api.js con pesos reales, descargados desde el CDN.";

  try {
    await ensureModels();
    const total = cohort.gallery.length + cohort.probes.length;
    const references = [];
    const probes = [];
    const failed = [];
    let done = 0;

    for (const entry of cohort.gallery) {
      try {
        references.push({ ...entry, descriptor: await descriptor(entry) });
      } catch (error) {
        failed.push(`Referencia de ${entry.person.id}: ${error.message}`);
      }
      done += 1;
      setProgress(done, total, "Procesando fotografías de referencia…");
      await new Promise((resolve) => requestAnimationFrame(resolve));
      if (done === 2 && references.length === 0) {
        throw new Error("Las primeras imágenes no se han podido procesar. Comprueba que el proveedor LFW permite su lectura en el navegador (CORS).");
      }
    }

    const processedReferences = new Set(references.map((item) => item.person.id));
    for (const entry of cohort.probes) {
      if (entry.enrolled && !processedReferences.has(entry.person.id)) {
        failed.push(`Prueba de ${entry.person.id}: la referencia no se pudo procesar.`);
      } else {
        try {
          probes.push({ ...entry, descriptor: await descriptor(entry) });
        } catch (error) {
          failed.push(`Prueba de ${entry.person.id}: ${error.message}`);
        }
      }
      done += 1;
      setProgress(done, total, "Procesando fotografías de prueba…");
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    if (references.length < 2 || probes.length < 2) {
      throw new Error("No se han obtenido suficientes embeddings para evaluar. Revisa la conexión y vuelve a intentarlo.");
    }
    processed = { references, probes, failed };
    elements.status.textContent = "Embeddings calculados con un modelo real";
    elements.description.textContent = `${references.length} referencias y ${probes.length} fotografías de prueba procesadas. ${failed.length} fotografías excluidas por problemas de lectura o detección.`;
    elements.progress.hidden = true;
    elements.results.hidden = false;
    renderResults();
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error("Error en el experimento de reconocimiento facial:", error);
    elements.status.textContent = "No se ha podido completar el experimento";
    elements.description.textContent = `${error.message} Si el problema persiste, confirma que los CDN y las imágenes de Hugging Face son accesibles desde esta red.`;
    elements.progress.hidden = true;
  } finally {
    elements.run.disabled = false;
    elements.size.disabled = false;
  }
}

function percent(rate) {
  return rate == null ? "No calculable" : `${formatter.format(rate * 100)} %`;
}

function resultLabel(outcome) {
  const labels = {
    correct: "Identificación correcta",
    "wrong-identity": "Identidad equivocada",
    missed: "Persona registrada sin identificar",
    "false-alert": "Falsa identificación de un desconocido",
    "correct-rejection": "Desconocido rechazado correctamente",
  };
  return labels[outcome];
}

function createCase(result) {
  const card = createElement("article", `case-card outcome-${result.outcome}`);
  const photos = createElement("div", "case-photos");
  for (const [person, index, caption] of [
    [result.person, result.imageNumber, "FOTOGRAFÍA DE PRUEBA"],
    [result.nearest, 1, "CANDIDATO MÁS CERCANO"],
  ]) {
    const figure = createElement("figure", "case-photo");
    figure.append(imageElement(person, index, "case-image"));
    figure.append(createElement("figcaption", "", caption));
    photos.append(figure);
  }
  const content = createElement("div", "case-content");
  const state = createElement("span", "case-outcome", resultLabel(result.outcome));
  const title = createElement("h4", "", result.accepted
    ? `Propuesta: ${result.nearest.name}` : "Resultado: sin identificar");
  const name = result.enrolled ? result.person.name : "Identidad no registrada en la galería";
  const explanation = createElement("p", "", `Realidad LFW: ${name}. Distancia al candidato más cercano: ${result.distance.toFixed(3)}.`);
  content.append(state, title, explanation);
  card.append(photos, content);
  return card;
}

function renderResults() {
  if (!processed) return;
  const threshold = Number(elements.threshold.value);
  elements.thresholdValue.textContent = threshold.toFixed(3).replace(".", ",");
  const matches = identify(processed.probes, processed.references, threshold);
  const metrics = evaluateMatches(matches);
  const values = {
    "#metric-correct": metrics.correct,
    "#metric-wrong": metrics.wrongIdentity,
    "#metric-missed": metrics.missed,
    "#metric-rejected": metrics.correctRejections,
    "#metric-false-alert": metrics.falseAlerts,
  };
  for (const [selector, value] of Object.entries(values)) $(selector).textContent = String(value);

  elements.precision.textContent = `Precisión de las asignaciones: ${percent(metrics.precision)}`;
  elements.scope.textContent = `De ${metrics.total} pruebas procesadas (${metrics.enrolled} identidades registradas y ${metrics.unknown} desconocidas), ${metrics.correct} asignaciones fueron correctas. ${processed.failed.length} fotografías no entraron en la evaluación. No es una estimación del rendimiento en un despliegue real.`;
  const filter = elements.caseFilter.value;
  const visible = matches.filter((result) => filter === "all" ||
    (filter === "errors" ? !["correct", "correct-rejection"].includes(result.outcome) :
      ["correct", "correct-rejection"].includes(result.outcome)));
  elements.caseSummary.textContent = `${visible.length} de ${matches.length} resultados`;
  elements.cases.replaceChildren(...visible.map(createCase));
  if (!visible.length) elements.cases.append(createElement("p", "empty-cases", "No hay casos de esta clase con el umbral seleccionado."));
}

async function main() {
  setupAnswers({
    first: 6, last: 12, step: 2,
    title: "PERFILADO — PASO 2: IDENTIFICACIÓN BIOMÉTRICA",
    filename: "perfilado-paso2-respuestas.txt",
  });
  elements.run.disabled = true;
  try {
    const response = await fetch("./data/lfw-identities.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    profiles = createProfiles((await response.json()).identities);
    const stats = getSummary(profiles);
    if (stats.profiles !== 5749 || stats.photographs !== 13233) throw new Error("Índice incompleto");
    configureCohort();
    elements.run.disabled = false;
  } catch (error) {
    elements.status.textContent = "No se ha podido cargar el catálogo LFW";
    elements.description.textContent = `Revisa la conexión y vuelve a abrir la página: ${error.message}`;
    return;
  }
  elements.size.addEventListener("change", configureCohort);
  elements.run.addEventListener("click", runExperiment);
  elements.threshold.addEventListener("input", renderResults);
  elements.caseFilter.addEventListener("change", renderResults);
}
main();
