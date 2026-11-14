import { createProfiles, getSummary, photoUrl } from "./core.js";
import { chooseProbePhotos, readGallery, nearestCandidates, classifyProbe, summariseProbes } from "./full-gallery.js";
import { extractDescriptor, ensureModels } from "./face-engine.js";
import { setupAnswers } from "./answers.js";
import { GALLERY_VERSION } from "./gallery-format.js";

const $ = (selector) => document.querySelector(selector);
const formatter = new Intl.NumberFormat("es-ES", { maximumFractionDigits: 1 });
const PAGE_SIZE = 10;

const elements = {
  galleryStatus: $("#gallery-status"),
  galleryDescription: $("#gallery-description"),
  galleryFile: $("#gallery-file"),
  launch: $("#run-model"),
  modelStatus: $("#model-status"),
  modelDescription: $("#model-description"),
  progress: $("#model-progress"),
  preview: $("#preview-gallery"),
  results: $("#results-section"),
  threshold: $("#distance-threshold"),
  thresholdValue: $("#threshold-value"),
  caseFilter: $("#case-filter"),
  caseSummary: $("#case-summary"),
  cases: $("#cases"),
  prev: $("#cases-previous"),
  next: $("#cases-next"),
  pageCounter: $("#cases-counter"),
  pageSummary: $("#page-summary"),
  top1: $("#metric-top1"),
  top1Rate: $("#top1-rate"),
  correct: $("#metric-correct"),
  wrong: $("#metric-wrong"),
  rejected: $("#metric-rejected"),
  failed: $("#metric-failed"),
  precision: $("#precision-text"),
  scope: $("#metrics-scope"),
};

let allProfiles = [];
let gallery = [];
let probes = [];
let predictions = [];
let failures = [];
let displayed = [];
let page = 1;
let running = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function faceImage(person, imageNumber, className = "face-preview") {
  const image = document.createElement("img");
  image.className = className;
  image.src = photoUrl(person, imageNumber);
  image.alt = `Fotografía de referencia LFW, ${person.name}, imagen ${imageNumber}`;
  image.loading = "lazy";
  image.decoding = "async";
  image.referrerPolicy = "no-referrer";
  image.addEventListener("error", () => {
    const replacement = element("span", className + " image-missing", "Imagen no disponible");
    replacement.setAttribute("role", "img");
    replacement.setAttribute("aria-label", "La imagen no pudo cargarse");
    image.replaceWith(replacement);
  }, { once: true });
  return image;
}

function showPreview() {
  const fragment = document.createDocumentFragment();
  for (const photo of probes.slice(0, 3)) {
    const item = element("figure", "preview-card");
    item.append(faceImage(photo.person, photo.imageNumber));
    item.append(element("figcaption", "preview-card-caption", "Nueva fotografía · identificación pendiente"));
    fragment.append(item);
  }
  elements.preview.replaceChildren(fragment);
}

function galleryFeedback(title, detail, isError = false) {
  elements.galleryStatus.textContent = title;
  elements.galleryDescription.textContent = detail;
  $("#gallery-feedback").classList.toggle("gallery-warning", isError);
}

function activateGallery(buffer, origin) {
  const loaded = readGallery(buffer, allProfiles);
  gallery = loaded.gallery;
  elements.launch.disabled = false;
  galleryFeedback(
    `Galería cargada: ${formatter.format(loaded.gallery.length)} vectores de referencia reales`,
    loaded.missing.length
      ? `${loaded.missing.length} referencias no son válidas y quedan excluidas. Origen: ${origin}.`
      : `Todos los vectores están disponibles. Origen: ${origin}. Ya puedes identificar las 100 fotos nuevas.`,
  );
  elements.modelStatus.textContent = "Experimento preparado";
  elements.modelDescription.textContent = "Pulsa el botón para calcular 100 nuevos embeddings en este navegador.";
}

async function loadGallery() {
  try {
    const response = await fetch("./data/lfw-gallery-v3.pfg", { cache: "default" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    activateGallery(await response.arrayBuffer(), "galería precomputada");
  } catch (error) {
    elements.launch.disabled = true;
    galleryFeedback(
      "La galería precalculada no está disponible automáticamente",
      `Puedes seleccionar «lfw-gallery-v3.pfg», que incluye las referencias recuperadas. Las versiones anteriores (.f32 y v2.pfg) no son compatibles. Detalle: ${error.message}.`,
      true,
    );
  }
}

async function importGallery() {
  const file = elements.galleryFile.files?.[0];
  if (!file || running) return;
  try {
    activateGallery(await file.arrayBuffer(), "archivo local");
  } catch (error) {
    elements.launch.disabled = true;
    galleryFeedback("El archivo seleccionado no es una galería válida", error.message, true);
  }
}

function formatDistance(distance) {
  return distance.toFixed(3).replace(".", ",");
}

/** Catches incompatible model/image pipelines before running any student evaluation.
 * Same original photo, same weights and alignment -> extremely close embeddings.
 */
async function validateGalleryParity() {
  elements.modelStatus.textContent = "Verificando que la galería y el navegador usan el mismo modelo…";
  const controls = gallery.filter((item) => item.person.imageCount >= 10).slice(0, 4);
  let errors = 0;
  for (const reference of controls) {
    let vector;
    try {
      vector = await extractDescriptor(photoUrl(reference.person, 1));
    } catch (error) {
      errors += 1;
      console.warn("Control de modelo no procesado:", error);
      continue;
    }
    const discrepancy = Math.hypot(...vector.map((value, index) =>
      value - reference.descriptor[index]));
    if (discrepancy > 0.12) {
      throw new Error(`La galería (${GALLERY_VERSION}) y el navegador no producen el mismo vector para una fotografía de control (distancia ${formatDistance(discrepancy)}). Recalcula la galería con la versión actual.`);
    }
    elements.modelDescription.textContent =
      `Control superado: distancia de consistencia ${formatDistance(discrepancy)} para la misma fotografía.`;
    return;
  }
  throw new Error(`No fue posible verificar el modelo con ninguna fotografía de control (${errors} fallos). Revisa conexión y detección facial.`);
}

async function runExperiment() {
  if (running || !gallery.length) return;
  running = true;
  elements.launch.disabled = true;
  elements.galleryFile.disabled = true;
  elements.results.hidden = true;
  elements.progress.hidden = false;
  elements.progress.value = 0;
  elements.modelStatus.textContent = "Cargando el modelo de detección, alineación y reconocimiento facial…";
  elements.modelDescription.textContent = "El navegador descargará los pesos preentrenados una sola vez.";

  predictions = [];
  failures = [];
  try {
    await ensureModels();
    await validateGalleryParity();
    for (let i = 0; i < probes.length; i += 1) {
      const photo = probes[i];
      try {
        const descriptor = await extractDescriptor(photoUrl(photo.person, photo.imageNumber));
        const candidates = nearestCandidates(descriptor, gallery, 5);
        predictions.push({ ...photo, candidates });
      } catch (error) {
        failures.push({ ...photo, error: String(error.message || error) });
      }
      elements.progress.value = i + 1;
      elements.modelStatus.textContent = `Procesadas ${i + 1} de 100 fotografías`;
      elements.modelDescription.textContent =
        `${predictions.length} embeddings válidos · ${failures.length} imágenes no procesadas. Cada búsqueda compara contra ${formatter.format(gallery.length)} referencias.`;
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
    elements.progress.hidden = true;
    if (!predictions.length) {
      throw new Error("No se ha podido procesar ninguna fotografía. Comprueba la conexión y el acceso CORS a Hugging Face.");
    }
    elements.modelStatus.textContent = "Evaluación realizada con embeddings reales";
    elements.modelDescription.textContent = `Completadas ${predictions.length} búsquedas entre ${formatter.format(gallery.length)} referencias. ${failures.length} fotos no pudieron procesarse.`;
    elements.results.hidden = false;
    page = 1;
    showResults();
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error("Error al identificar las fotografías:", error);
    elements.modelStatus.textContent = "No se ha podido completar el experimento";
    elements.modelDescription.textContent = `${error.message} No se generaron resultados ficticios para suplir el error.`;
    elements.progress.hidden = true;
  } finally {
    running = false;
    elements.launch.disabled = !gallery.length;
    elements.galleryFile.disabled = false;
  }
}

function caseOutcome(outcome) {
  return {
    correct: "Identificación aceptada y correcta",
    wrong: "Identificación aceptada y equivocada",
    rejected: "Identificación rechazada por umbral",
  }[outcome];
}

function caseCard(result) {
  const card = element("article", `case-card outcome-${result.outcome}`);
  const pictures = element("div", "case-photos");
  const [best] = result.candidates;

  for (const [person, index, label] of [
    [result.person, result.imageNumber, "IMAGEN DE PRUEBA"],
    [best.person, 1, "CANDIDATO TOP-1"],
  ]) {
    const figure = element("figure", "case-photo");
    figure.append(faceImage(person, index, "case-image"), element("figcaption", "", label));
    pictures.append(figure);
  }

  const content = element("div", "case-content");
  const outcome = element("span", "case-outcome", caseOutcome(result.outcome));
  const heading = element("h4", "", result.accepted
    ? `Identificación propuesta: ${best.person.name}` : "Resultado: sin identificar");
  const explanation = element("p", "",
    `Identidad real: ${result.person.name}. Candidato top-1: ${best.person.name}. Distancia: ${formatDistance(best.distance)}.`);
  const topFive = element("details", "top-five");
  const summary = element("summary", "", "Ver las cinco coincidencias más próximas");
  const ordered = element("ol", "top-five-list");
  for (const candidate of result.candidates) {
    ordered.append(element("li", "", `${candidate.person.name} · distancia ${formatDistance(candidate.distance)}`));
  }
  topFive.append(summary, ordered);
  content.append(outcome, heading, explanation, topFive);
  card.append(pictures, content);
  return card;
}

function showPage() {
  const total = displayed.length;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if (page > pages) page = pages;
  const from = (page - 1) * PAGE_SIZE;
  const fragment = document.createDocumentFragment();
  for (const result of displayed.slice(from, from + PAGE_SIZE)) fragment.append(caseCard(result));
  if (!total) fragment.append(element("p", "empty-cases", "No hay casos de este tipo con el umbral seleccionado."));
  elements.cases.replaceChildren(fragment);
  elements.pageSummary.textContent = total
    ? `Mostrando ${from + 1}–${Math.min(from + PAGE_SIZE, total)} de ${total}`
    : "Sin resultados en este filtro";
  elements.caseSummary.textContent = `${total} de ${predictions.length} casos`;
  elements.pageCounter.textContent = `${page} / ${pages}`;
  elements.prev.disabled = page === 1;
  elements.next.disabled = page === pages;
}

function showResults() {
  if (!predictions.length) return;
  const threshold = Number(elements.threshold.value);
  elements.thresholdValue.textContent = formatDistance(threshold);
  const classified = predictions.map((probe) => classifyProbe(probe, probe.candidates, threshold));
  const metrics = summariseProbes(classified, failures.length);
  elements.top1.textContent = String(metrics.top1);
  elements.top1Rate.textContent = `de ${metrics.processed} procesadas (${formatter.format(100 * metrics.top1Rate)} %)`;
  elements.correct.textContent = String(metrics.correct);
  elements.wrong.textContent = String(metrics.wrong);
  elements.rejected.textContent = String(metrics.rejected);
  elements.failed.textContent = String(metrics.failures);
  elements.precision.textContent = `Top-1 sin umbral: ${metrics.top1}/${metrics.processed} · Tras aplicar el umbral: ${metrics.correct} correctas, ${metrics.wrong} erróneas y ${metrics.rejected} rechazadas.`;
  elements.scope.textContent = `De 100 fotos nuevas se procesaron ${metrics.processed} y ${metrics.failures} quedaron fuera de la evaluación. Todas las personas de prueba figuran en la galería. No se mide la respuesta ante desconocidos.`;

  const filter = elements.caseFilter.value;
  displayed = classified.filter((r) => filter === "all" || (
    filter === "errors" ? r.outcome === "wrong" || r.outcome === "rejected"
      : filter === "correct" ? r.outcome === "correct" : r.outcome === "rejected"
  ));
  showPage();
}

async function main() {
  setupAnswers({
    first: 6, last: 12, step: 2,
    title: "PERFILADO — PASO 2: IDENTIFICACIÓN BIOMÉTRICA",
    filename: "perfilado-paso2-respuestas.txt",
  });
  elements.progress.max = 100;
  try {
    const response = await fetch("./data/lfw-identities.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    allProfiles = createProfiles((await response.json()).identities);
    const stats = getSummary(allProfiles);
    if (stats.profiles !== 5749 || stats.photographs !== 13233) throw new Error("Índice LFW incompleto");
    probes = chooseProbePhotos(allProfiles, 100);
    showPreview();
    await loadGallery();
  } catch (error) {
    galleryFeedback("No se ha podido cargar el catálogo de LFW", error.message, true);
    return;
  }

  elements.galleryFile.addEventListener("change", importGallery);
  elements.launch.addEventListener("click", runExperiment);
  elements.threshold.addEventListener("input", () => { page = 1; showResults(); });
  elements.caseFilter.addEventListener("change", () => { page = 1; showResults(); });
  elements.prev.addEventListener("click", () => { if (page > 1) { page -= 1; showPage(); } });
  elements.next.addEventListener("click", () => {
    if (page * PAGE_SIZE < displayed.length) { page += 1; showPage(); }
  });
}
main();
