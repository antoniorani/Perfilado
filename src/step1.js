import { createProfiles, filterProfiles, getSummary, exportProfilesCsv, ZONES } from "./core.js";

const PAGE_SIZE = 10;
const TOTAL_EXPECTED = 5749;
const formatter = new Intl.NumberFormat("es-ES");
const $ = (selector) => document.querySelector(selector);

const elements = {
  search: $("#search"),
  zone: $("#zone"),
  sort: $("#sort"),
  body: $("#profiles-body"),
  dataStatus: $("#data-status"),
  total: $("#stat-identities"),
  references: $("#stat-references"),
  tableSummary: $("#table-summary"),
  counter: $("#page-counter"),
  next: $("#next"),
  previous: $("#previous"),
  csv: $("#download-csv"),
  photos: $("#photo-files"),
  fileStatus: $("#file-status"),
  zoneChart: $("#zone-chart"),
  ageChart: $("#age-chart"),
  exportAnswers: $("#export-answers"),
  saveState: $("#save-state"),
};

let profiles = [];
let filtered = [];
let currentPage = 1;
let imageFiles = new Map();
const photoUrls = new Map();

function setStatus(message, isError = false) {
  elements.dataStatus.hidden = !message;
  elements.dataStatus.textContent = message;
  elements.dataStatus.style.background = isError ? "#ffede8" : "";
  elements.dataStatus.style.color = isError ? "#923d29" : "";
}

function td(className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  return cell;
}

function makeImage(profile) {
  const container = document.createElement("div");
  container.className = "image-cell";
  const matchingFile = imageFiles.get(profile.referenceFile.toLowerCase());
  if (matchingFile) {
    let url = photoUrls.get(matchingFile);
    if (!url) {
      url = URL.createObjectURL(matchingFile);
      photoUrls.set(matchingFile, url);
    }
    const image = document.createElement("img");
    image.src = url;
    image.alt = `Fotografía de referencia del perfil ${profile.id}`;
    image.className = "photo-thumb";
    image.loading = "lazy";
    image.decoding = "async";
    container.append(image);
  } else {
    const placeholder = document.createElement("span");
    placeholder.className = "photo-thumb";
    placeholder.setAttribute("aria-label", "Imagen no cargada");
    placeholder.textContent = "IMG";
    container.append(placeholder);
  }
  const meta = document.createElement("span");
  meta.className = "thumb-meta";
  const top = document.createElement("strong");
  top.textContent = "Referencia 01";
  const bottom = document.createElement("small");
  bottom.textContent = matchingFile ? "Cargada localmente" : "Imagen no cargada";
  meta.append(top, bottom);
  container.append(meta);
  return container;
}

function renderTable() {
  const start = (currentPage - 1) * PAGE_SIZE;
  const visible = filtered.slice(start, start + PAGE_SIZE);
  const rows = document.createDocumentFragment();
  for (const profile of visible) {
    const row = document.createElement("tr");
    const photoCell = td();
    photoCell.append(makeImage(profile));

    const idCell = td("profile-code");
    idCell.textContent = profile.id;
    const ageCell = td();
    ageCell.textContent = String(profile.age);
    const zoneCell = td();
    zoneCell.textContent = profile.zone;
    const languageCell = td();
    languageCell.textContent = profile.language;

    const typeCell = td();
    const chips = document.createElement("span");
    chips.className = "type-chips";
    for (const text of ["1 foto LFW", "3 datos ficticios"]) {
      const chip = document.createElement("span");
      chip.textContent = text;
      chips.append(chip);
    }
    typeCell.append(chips);
    row.append(photoCell, idCell, ageCell, zoneCell, languageCell, typeCell);
    rows.append(row);
  }
  elements.body.replaceChildren(rows);
  const count = filtered.length;
  elements.tableSummary.textContent = count
    ? `Mostrando ${formatter.format(start + 1)}–${formatter.format(Math.min(start + PAGE_SIZE, count))} de ${formatter.format(count)} perfiles`
    : "No hay perfiles que coincidan con los filtros.";
  const pages = Math.max(1, Math.ceil(count / PAGE_SIZE));
  elements.counter.textContent = `${currentPage} / ${pages}`;
  elements.previous.disabled = currentPage === 1;
  elements.next.disabled = currentPage >= pages;
}

function updateFilters() {
  filtered = filterProfiles(profiles, {
    query: elements.search.value,
    zone: elements.zone.value,
    order: elements.sort.value,
  });
  currentPage = 1;
  renderTable();
}

function renderBarChart(target, values) {
  const maxValue = Math.max(1, ...Object.values(values));
  const fragment = document.createDocumentFragment();
  for (const [label, count] of Object.entries(values)) {
    const line = document.createElement("div");
    line.className = "chart-line";
    const caption = document.createElement("span");
    caption.className = "chart-line-label";
    caption.title = label;
    caption.textContent = label;
    const track = document.createElement("span");
    track.className = "chart-track";
    const fill = document.createElement("span");
    fill.className = "chart-fill";
    fill.style.display = "block";
    fill.style.width = `${100 * count / maxValue}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "chart-line-count";
    value.textContent = formatter.format(count);
    line.append(caption, track, value);
    fragment.append(line);
  }
  target.replaceChildren(fragment);
}

function renderStats() {
  const summary = getSummary(profiles);
  elements.total.textContent = formatter.format(summary.profiles);
  elements.references.textContent = formatter.format(summary.references);
  renderBarChart(elements.zoneChart, summary.byZone);
  renderBarChart(elements.ageChart, summary.byAge);
}

function downloadText(filename, content, type) {
  const file = new Blob([content], { type });
  const url = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function clearPhotoUrls() {
  for (const url of photoUrls.values()) URL.revokeObjectURL(url);
  photoUrls.clear();
}

function importPhotographs(files) {
  clearPhotoUrls();
  imageFiles = new Map();
  for (const file of files) {
    if (/\.jpe?g$/i.test(file.name)) imageFiles.set(file.name.toLowerCase(), file);
  }
  const matches = profiles.reduce((total, profile) =>
    total + Number(imageFiles.has(profile.referenceFile.toLowerCase())), 0);
  elements.fileStatus.textContent = `${formatter.format(imageFiles.size)} archivos JPG seleccionados · ${formatter.format(matches)} referencias reconocidas.`;
  renderTable();
}

function installQuestions() {
  for (let index = 1; index <= 5; index += 1) {
    const field = document.getElementById(`q${index}`);
    const key = `perfilado:step1:q${index}`;
    try { field.value = localStorage.getItem(key) || ""; } catch { /* storage disabled */ }
    field.addEventListener("input", () => {
      try {
        localStorage.setItem(key, field.value);
        elements.saveState.textContent = "Respuestas guardadas en este navegador.";
      } catch {
        elements.saveState.textContent = "No se pueden guardar automáticamente: exporta tus respuestas.";
      }
    });
  }
  elements.exportAnswers.addEventListener("click", () => {
    const entries = Array.from({ length: 5 }, (_, i) => {
      const field = document.getElementById(`q${i + 1}`);
      const prompt = document.querySelector(`label[for="q${i + 1}"]`).textContent.trim();
      return `${prompt}\n\n${field.value.trim() || "(Sin respuesta)"}`;
    });
    downloadText("perfilado-paso-1-respuestas.txt",
      `PERFILADO — PASO 1: EXPLORACIÓN DE DATOS\n\n${entries.join("\n\n----------------------------------------\n\n")}\n`,
      "text/plain;charset=utf-8");
  });
}

async function main() {
  installQuestions();
  for (const zone of ZONES) {
    const option = document.createElement("option");
    option.value = zone;
    option.textContent = zone;
    elements.zone.append(option);
  }

  try {
    const response = await fetch("./data/lfw-identities.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    profiles = createProfiles(data.identities);
    const summary = getSummary(profiles);
    if (summary.profiles !== TOTAL_EXPECTED || summary.photographs !== 13233) {
      throw new Error("El catálogo LFW no supera la verificación de integridad");
    }
    renderStats();
    updateFilters();
    setStatus("");
  } catch (error) {
    console.error("No se pudo cargar el catálogo de LFW:", error);
    setStatus("No se pudo cargar el catálogo. Abre la web mediante un servidor local o GitHub Pages y actualiza la página.", true);
    elements.csv.disabled = true;
    elements.previous.disabled = true;
    elements.next.disabled = true;
    return;
  }

  for (const control of [elements.search, elements.zone, elements.sort]) {
    control.addEventListener(control === elements.search ? "input" : "change", updateFilters);
  }
  elements.previous.addEventListener("click", () => {
    if (currentPage > 1) { currentPage -= 1; renderTable(); }
  });
  elements.next.addEventListener("click", () => {
    if (currentPage * PAGE_SIZE < filtered.length) { currentPage += 1; renderTable(); }
  });
  elements.csv.addEventListener("click", () => {
    downloadText("perfilado-paso1-perfiles.csv", exportProfilesCsv(filtered), "text/csv;charset=utf-8");
  });
  elements.photos.addEventListener("change", (event) => {
    importPhotographs(Array.from(event.target.files || []));
  });
  window.addEventListener("pagehide", clearPhotoUrls);
}
main();
