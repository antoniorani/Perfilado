import { createProfiles, filterProfiles, getSummary, referenceImageUrl, ZONES } from "./core.js";

const PAGE_SIZE = 10;
const formatter = new Intl.NumberFormat("es-ES");
const $ = (selector) => document.querySelector(selector);
const elements = {
  search: $("#search"), zone: $("#zone"), sort: $("#sort"),
  body: $("#profiles-body"), dataStatus: $("#data-status"),
  tableSummary: $("#table-summary"), counter: $("#page-counter"),
  next: $("#next"), previous: $("#previous"),
  zoneChart: $("#zone-chart"), ageChart: $("#age-chart"),
  exportAnswers: $("#export-answers"), saveState: $("#save-state"),
};

let profiles = [];
let filtered = [];
let currentPage = 1;

function setStatus(message, isError = false) {
  elements.dataStatus.hidden = !message;
  elements.dataStatus.textContent = message;
  elements.dataStatus.classList.toggle("data-status-error", isError);
}

function cell(value, className = "") {
  const element = document.createElement("td");
  element.textContent = value;
  if (className) element.className = className;
  return element;
}

function photo(profile) {
  const holder = document.createElement("div");
  holder.className = "image-cell";
  const img = document.createElement("img");
  img.src = referenceImageUrl(profile);
  img.alt = `Fotografía de ${profile.name} en el conjunto LFW`;
  img.className = "photo-thumb";
  img.loading = "lazy";
  img.decoding = "async";
  img.referrerPolicy = "no-referrer";
  img.width = 52;
  img.height = 52;
  img.addEventListener("error", () => {
    const fallback = document.createElement("span");
    fallback.className = "photo-thumb";
    fallback.setAttribute("role", "img");
    fallback.setAttribute("aria-label", "Fotografía no disponible");
    fallback.textContent = "—";
    img.replaceWith(fallback);
  }, { once: true });
  holder.append(img);
  return holder;
}

function renderTable() {
  const start = (currentPage - 1) * PAGE_SIZE;
  const fragment = document.createDocumentFragment();
  for (const profile of filtered.slice(start, start + PAGE_SIZE)) {
    const row = document.createElement("tr");
    const photograph = document.createElement("td");
    photograph.append(photo(profile));
    row.append(
      photograph,
      cell(profile.name, "real-name"),
      cell(profile.id, "profile-code"),
      cell(String(profile.age)),
      cell(profile.zone),
      cell(profile.language),
    );
    fragment.append(row);
  }
  elements.body.replaceChildren(fragment);
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
    query: elements.search.value, zone: elements.zone.value, order: elements.sort.value,
  });
  currentPage = 1;
  renderTable();
}

function renderChart(target, counts) {
  const max = Math.max(1, ...Object.values(counts));
  const fragment = document.createDocumentFragment();
  for (const [label, count] of Object.entries(counts)) {
    const row = document.createElement("div");
    row.className = "chart-line";
    const name = document.createElement("span");
    name.className = "chart-line-label";
    name.textContent = label;
    const track = document.createElement("span");
    track.className = "chart-track";
    const fill = document.createElement("span");
    fill.className = "chart-fill";
    fill.style.display = "block";
    fill.style.width = `${100 * count / max}%`;
    track.append(fill);
    const number = document.createElement("span");
    number.className = "chart-line-count";
    number.textContent = formatter.format(count);
    row.append(name, track, number);
    fragment.append(row);
  }
  target.replaceChildren(fragment);
}

function downloadAnswers() {
  const entries = Array.from({ length: 5 }, (_, index) => {
    const field = document.getElementById(`q${index + 1}`);
    const question = document.querySelector(`label[for="q${index + 1}"]`).textContent.trim();
    return `${question}\n\n${field.value.trim() || "(Sin respuesta)"}`;
  });
  const content = `PERFILADO — PASO 1: EXPLORACIÓN\n\n${entries.join("\n\n------------------------------------\n\n")}\n`;
  const url = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "perfilado-paso1-respuestas.txt";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setUpQuestions() {
  for (let index = 1; index <= 5; index += 1) {
    const field = document.getElementById(`q${index}`);
    const key = `perfilado:step1:q${index}`;
    try { field.value = localStorage.getItem(key) || ""; } catch { /* Storage may be disabled */ }
    field.addEventListener("input", () => {
      try {
        localStorage.setItem(key, field.value);
        elements.saveState.textContent = "Respuestas guardadas en este navegador.";
      } catch {
        elements.saveState.textContent = "Guardado automático no disponible. Exporta tus respuestas.";
      }
    });
  }
  elements.exportAnswers.addEventListener("click", downloadAnswers);
}

async function main() {
  setUpQuestions();
  for (const zone of ZONES) {
    const option = document.createElement("option");
    option.value = zone;
    option.textContent = zone;
    elements.zone.append(option);
  }
  try {
    const response = await fetch("./data/lfw-identities.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { identities } = await response.json();
    profiles = createProfiles(identities);
    const summary = getSummary(profiles);
    if (summary.profiles !== 5749 || summary.photographs !== 13233) {
      throw new Error("El índice de LFW no supera la verificación de integridad");
    }
    renderChart(elements.zoneChart, summary.byZone);
    renderChart(elements.ageChart, summary.byAge);
    updateFilters();
    setStatus("");
  } catch (error) {
    console.error("No se pudo cargar el índice de LFW:", error);
    setStatus("No se pudo cargar el catálogo. Abre la web con un servidor HTTP o desde GitHub Pages.", true);
    elements.previous.disabled = true;
    elements.next.disabled = true;
    return;
  }

  for (const field of [elements.search, elements.zone, elements.sort]) {
    field.addEventListener(field === elements.search ? "input" : "change", updateFilters);
  }
  elements.previous.addEventListener("click", () => {
    if (currentPage > 1) { currentPage -= 1; renderTable(); }
  });
  elements.next.addEventListener("click", () => {
    if (currentPage * PAGE_SIZE < filtered.length) { currentPage += 1; renderTable(); }
  });
}
main();
