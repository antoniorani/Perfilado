/** Step 2 Chromium smoke. The browser runs the real gallery/search UI with a
 * deterministic embedding stub to keep CI fast and independent from model CDNs.
 */
import { chromium } from "playwright";
import fs from "node:fs/promises";

const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
const page = await browser.newPage({ viewport: { width: 1365, height: 900 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));

try {
  await page.route("**/src/face-engine.js", route => route.fulfill({
    contentType: "text/javascript",
    body: `
      window.__modelCalls = { load: 0, photos: [] };
      export async function ensureModels() { window.__modelCalls.load += 1; }
      export async function extractDescriptor(url) {
        window.__modelCalls.photos.push(url);
        return Array(128).fill(0.08);
      }
    `,
  }));
  await page.route("https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js",
    route => route.fulfill({ status: 200, contentType: "application/javascript", body: "" }));
  const pixel = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0e2m8AAAAASUVORK5CYII=", "base64",
  );
  await page.route("https://huggingface.co/datasets/marcelohaps/lfw/resolve/**",
    route => route.fulfill({ status: 200, contentType: "image/png", body: pixel }));

  await page.goto("http://localhost:8000/biometria.html", { waitUntil: "domcontentloaded" });
  if (await page.locator(".process-grid-reference .process-card").count() !== 4 ||
      await page.locator(".process-grid-recognition .process-card").count() !== 2) {
    throw new Error("Expected four preparation steps and two identification steps");
  }
  if (await page.locator(".process-icon use").count() !== 6 ||
      await page.locator(".alignment-illustration svg").count() !== 1) {
    throw new Error("Missing model pipeline icons or visual alignment example");
  }
  const explanation = await page.locator(".biometric-procedure").innerText();
  for (const term of ["Face Landmark 68", "Face Recognition Net", "SSD MobileNet V1", "distancia euclidiana"]) {
    if (!explanation.includes(term)) throw new Error("Missing technical explanation: " + term);
  }
  if (await page.locator("#save-state, #export-answers, .question-actions").count()) {
    throw new Error("Step 2 should not display worksheet status/export controls");
  }
  if (await page.getByText("PROTECCIÓN DE DATOS", { exact: true }).count()) {
    throw new Error("Step 2 privacy label should be hidden");
  }

  await page.waitForFunction(() => !document.querySelector("#run-model").disabled,
    { timeout: 25000 });
  if (!(await page.locator("#gallery-feedback").evaluate(el => el.hidden))) {
    throw new Error("Redundant green gallery-status panel should be hidden when data loads");
  }
  await page.locator("#run-model").click();
  await page.locator("#results-section").waitFor({ state: "visible", timeout: 90000 });
  const calls = await page.evaluate(() => window.__modelCalls);
  if (calls.load !== 1 || calls.photos.length !== 100 ||
      !calls.photos.every(url => url.endsWith("_0002.jpg"))) {
    throw new Error("The run must extract only 100 new images, with no reference parity reprocessing: "
      + JSON.stringify({load:calls.load,photos:calls.photos.length,first:calls.photos[0]}));
  }

  // Render an explicit synthetic error, where TOP-2 is correct.
  // Real LFW labels are used only as face-matching test fixtures; no political texts.
  const expected = await page.evaluate(async () => {
    const { caseCard } = await import("./src/step2.js");
    const { createProfiles } = await import("./src/core.js");
    const { identities } = await (await fetch("./data/lfw-identities.json")).json();
    const [real, wrong, third, fourth, fifth] =
      createProfiles(identities).filter(person => person.imageCount >= 2).slice(0, 5);
    const candidates = [wrong, real, third, fourth, fifth].map((person, index) => ({
      person, distance: 0.3 + index * 0.06,
    }));
    const card = caseCard({
      person: real, imageNumber: 2, candidates, nearest: candidates[0],
      top1Correct: false, accepted: true, outcome: "wrong",
    });
    document.querySelector("#cases").replaceChildren(card);
    return {real:real.name,wrong:wrong.name};
  });
  const card = page.locator("#cases .case-card").first();
  if (await card.locator(".case-photo").count() !== 3) {
    throw new Error("Error review must show probe photo, TOP-1 and TOP-2");
  }
  const heading = await card.locator("h4").textContent();
  if (heading !== `Era ${expected.real} y el modelo ha dicho que era ${expected.wrong}`) {
    throw new Error("Unexpected prediction heading: " + heading);
  }
  if (await card.locator(".case-match-wrong").count() !== 1 ||
      await card.locator(".case-match-correct").count() !== 1) {
    throw new Error("TOP-1 must show a red cross and correct TOP-2 a green tick");
  }
  await card.locator(".top-five summary").click();
  if (await card.locator(".top-five-list .candidate-correct").count() !== 1 ||
      await card.locator(".top-five-list .candidate-wrong").count() !== 4) {
    throw new Error("The top five list must indicate all incorrect/correct matches");
  }

  await page.locator("#q12").fill("Se debe examinar la base jurídica y minimización.");
  await page.reload({ waitUntil: "domcontentloaded" });
  if (!(await page.locator("#q12").inputValue()).includes("base jurídica")) {
    throw new Error("Automatic student response saving has regressed");
  }
  await fs.mkdir("build", { recursive: true });
  await page.screenshot({ path: "build/step2-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "build/step2-mobile.png", fullPage: true });
  const dimensions = await page.evaluate(() => ({
    viewport: innerWidth, document: document.documentElement.scrollWidth,
  }));
  if (dimensions.document > dimensions.viewport + 2) {
    throw new Error("Step 2 mobile overflow: " + JSON.stringify(dimensions));
  }
  if (errors.length) throw new Error("Browser errors: " + errors.join(" | "));
  console.log("STEP2_BROWSER_SMOKE_SUCCESS", JSON.stringify({
    modelCalls:calls.load,probeEmbeddings:calls.photos.length,
    candidates:5,syntheticTop2Correct:true,dimensions,
  }));
} finally {
  await browser.close();
}
