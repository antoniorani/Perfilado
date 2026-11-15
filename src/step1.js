import { createProfiles, filterProfiles, getSummary, referenceImageUrl, ZONES } from "./core.js";
import { setupAnswers } from "./answers.js";

const PAGE_SIZE = 10;
const DISTRIBUTIONS = {
  zone: { title: "Perfiles por zona", key: "byZone" },
  age: { title: "Perfiles por tramo de edad", key: "byAge" },
  language: { title: "Perfiles por idioma", key: "byLanguage" },
};
const formatter = new Intl.NumberFormat("es-ES");
const $ = (selector) => document.querySelector(selector);
const elements = {
  search: $("#search"), zone: $("#zone"), sort: $("#sort"),
  body: $("#profiles-body"), dataStatus: $("#data-status"),
  tableSummary: $("#table-summary"), counter: $("#page-counter"),
  next: $("#next"), previous: $("#previous"),
  distributionSelect: $("#distribution-select"),
  distributionTitle: $("#distribution-title"), distributionChart: $("#distribution-chart"),
  photoDialog: $("#photo-dialog"), photoClose: $("#photo-close"),
  photoLarge: $("#photo-large"), photoCaption: $("#photo-caption"), photoError: $("#photo-error"),
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
  const button = document.createElement("button");
  button.type = "button";
  button.className = "photo-zoom";
  button.setAttribute("aria-label", `Ampliar fotografía de ${profile.name}`);
  button.dataset.photoUrl = referenceImageUrl(profile);
  button.dataset.photoName = profile.name;

  const img = document.createElement("img");
  img.src = button.dataset.photoUrl;
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
    button.replaceWith(fallback);
  }, { once: true });
  button.append(img);
  holder.append(button);
  return holder;
}

function openPhoto(button) {
  const name = button.dataset.photoName;
  const large = elements.photoLarge;
  elements.photoError.hidden = true;
  large.hidden = false;
  large.alt = `Fotografía ampliada de ${name} en LFW`;
  elements.photoCaption.textContent = name;
  large.onerror = () => {
    large.hidden = true;
    elements.photoError.hidden = false;
  };
  large.src = button.dataset.photoUrl;
  elements.photoDialog.showModal();
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

function showDistribution(summary) {
  const setting = DISTRIBUTIONS[elements.distributionSelect.value];
  if (!setting) return;
  elements.distributionTitle.textContent = setting.title;
  elements.distributionChart.setAttribute("aria-label", setting.title);
  renderChart(elements.distributionChart, summary[setting.key]);
}

async function main() {
  // Preserve drafts from the previously numbered "05 bis" prompt.
  try {
    const oldAnswer = localStorage.getItem("perfilado:step1:q5b");
    if (oldAnswer !== null && localStorage.getItem("perfilado:step1:q6") === null) {
      localStorage.setItem("perfilado:step1:q6", oldAnswer);
    }
  } catch { /* Local storage may be unavailable. */ }
  setupAnswers({ first: 1, last: 6, step: 1 });
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
    showDistribution(summary);
    elements.distributionSelect.addEventListener("change", () => showDistribution(summary));
    updateFilters();
    setStatus("");
  } catch (error) {
    console.error("No se pudo cargar el índice de LFW:", error);
    setStatus("No se pudo cargar el catálogo. Abre la web con un servidor HTTP o desde GitHub Pages.", true);
    elements.previous.disabled = true;
    elements.next.disabled = true;
    return;
  }

  elements.body.addEventListener("click", (event) => {
    const button = event.target.closest(".photo-zoom");
    if (button) openPhoto(button);
  });
  elements.photoClose.addEventListener("click", () => elements.photoDialog.close());
  elements.photoDialog.addEventListener("click", (event) => {
    if (event.target === elements.photoDialog) elements.photoDialog.close();
  });

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
