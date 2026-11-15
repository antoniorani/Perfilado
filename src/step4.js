import { combinedRisk } from "./combined-risk.js";
import { setupAnswers } from "./answers.js";

const percentage = new Intl.NumberFormat("es-ES", {
  style: "percent", maximumFractionDigits: 1,
});
const fields = {
  face: document.querySelector("#face-rate"),
  link: document.querySelector("#link-rate"),
  text: document.querySelector("#text-rate"),
};
const defaults = { face: 80, link: 70, text: 53 };
const output = {
  face: document.querySelector("#face-value"),
  link: document.querySelector("#link-value"),
  text: document.querySelector("#text-value"),
  joint: document.querySelector("#joint-result"),
  failure: document.querySelector("#failure-result"),
  bounds: document.querySelector("#bounds-result"),
  bar: document.querySelector("#joint-bar"),
};

function updateIllustration() {
  const face = Number(fields.face.value);
  const link = Number(fields.link.value);
  const text = Number(fields.text.value);
  const risk = combinedRisk(face, link, text);

  for (const [name, input] of Object.entries(fields)) {
    output[name].value = percentage.format(Number(input.value) / 100);
  }
  output.joint.textContent = percentage.format(risk.simultaneous);
  output.failure.textContent = percentage.format(risk.atLeastOneFailure);
  output.bounds.textContent =
    percentage.format(risk.lowerBound) + "–" + percentage.format(risk.upperBound);
  output.bar.style.width = (risk.simultaneous * 100).toFixed(2) + "%";
  output.bar.setAttribute("aria-valuenow", (risk.simultaneous * 100).toFixed(1));
  output.bar.setAttribute("aria-valuetext", percentage.format(risk.simultaneous));
}

for (const input of Object.values(fields)) input.addEventListener("input", updateIllustration);
document.querySelector("#reset-rates").addEventListener("click", () => {
  for (const [name, value] of Object.entries(defaults)) fields[name].value = value;
  updateIllustration();
});
setupAnswers({
  first: 20, last: 23, step: 4,
  title: "PERFILADO — PASO 4: EVALUACIÓN CONJUNTA",
  filename: "perfilado-paso4-respuestas.txt",
});
updateIllustration();
