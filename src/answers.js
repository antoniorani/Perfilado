/** Reusable, browser-local student worksheet. No server-side submission. */
export function setupAnswers({ first, last, step, title, filename }) {
  const status = document.querySelector("#save-state");
  const exportButton = document.querySelector("#export-answers");
  for (let number = first; number <= last; number += 1) {
    const answer = document.getElementById(`q${number}`);
    const key = `perfilado:step${step}:q${number}`;
    try { answer.value = localStorage.getItem(key) || ""; } catch { /* local storage disabled */ }
    answer.addEventListener("input", () => {
      try {
        localStorage.setItem(key, answer.value);
        status.textContent = "Respuestas guardadas en este navegador.";
      } catch {
        status.textContent = "No es posible guardar aquí. Exporta las respuestas para conservarlas.";
      }
    });
  }
  exportButton.addEventListener("click", () => {
    const rows = Array.from({ length: last - first + 1 }, (_, index) => {
      const number = index + first;
      const question = document.querySelector(`label[for="q${number}"]`).textContent.trim();
      const answer = document.getElementById(`q${number}`).value.trim();
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
