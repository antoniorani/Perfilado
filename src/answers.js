/** Reusable, browser-local student worksheet. No server-side submission. */
export function setupAnswers({ first, last, step, title, filename, additionalIds = [] }) {
  const status = document.querySelector("#save-state");
  const exportButton = document.querySelector("#export-answers");
  const ids = [...Array.from({ length: last - first + 1 }, (_, index) => `q${first + index}`), ...additionalIds];
  for (const id of ids) {
    const answer = document.getElementById(id);
    const key = `perfilado:step${step}:${id}`;
    try { answer.value = localStorage.getItem(key) || ""; } catch { /* local storage disabled */ }
    answer.addEventListener("input", () => {
      try {
        localStorage.setItem(key, answer.value);
        if (status) status.textContent = "Respuestas guardadas en este navegador.";
      } catch {
        if (status) status.textContent = "No es posible guardar aquí. Exporta las respuestas para conservarlas.";
      }
    });
  }
  if (exportButton) exportButton.addEventListener("click", () => {
    const rows = ids.map((id) => {
      const question = document.querySelector(`label[for="${id}"]`).textContent.trim();
      const answer = document.getElementById(id).value.trim();
      return `${question}\n\n${answer || "(Sin respuesta)"}`;
    });
    const content = `${title}\n\n${rows.join("\n\n------------------------------------\n\n")}\n`;
    const url = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
