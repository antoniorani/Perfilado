/** Intro + Step 1 browser smoke against the real static application. */
import { chromium } from "playwright";
import fs from "node:fs/promises";

const browser = await chromium.launch({headless:true,args:["--no-sandbox"]});
const page = await browser.newPage({viewport:{width:1365,height:900},acceptDownloads:true});
const pageErrors = [];
page.on("pageerror", error => pageErrors.push(error.message));

try {
  await page.goto("http://localhost:8000/index.html",{waitUntil:"domcontentloaded"});
  if (await page.locator(".journey-grid a").count() !== 0) {
    throw new Error("The introductory phase cards must not be navigable");
  }
  if (await page.locator(".journey-grid .journey-card").count() !== 4) {
    throw new Error("The introduction should describe all four phases");
  }
  if (await page.getByRole("link",{name:/Comenzar la práctica/}).count() !== 2) {
    throw new Error("Expected exactly two start buttons on the introduction");
  }
  if (await page.getByText("Abrir el expediente 01").count()) throw new Error("Obsolete entry point");
  if (!(await page.locator(".hero-author").textContent()).includes("Antonio Ramos Nieto")) {
    throw new Error("Author not visibly credited in hero");
  }

  // Avoid depending on a third-party image host in CI while testing the modal.
  const onePixel = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0e2m8AAAAASUVORK5CYII=", "base64");
  await page.route("https://huggingface.co/datasets/marcelohaps/lfw/resolve/**",
    route => route.fulfill({status:200, contentType:"image/png", body:onePixel}));

  await page.goto("http://localhost:8000/datos.html",{waitUntil:"domcontentloaded"});
  await page.locator("#profiles-body tr").first().waitFor({timeout:30000});
  if (await page.locator("#profiles-body tr").count() !== 10) {
    throw new Error("Initial dataset page should show ten profiles");
  }
  const facts = await page.locator(".dataset-facts").innerText();
  if (facts.includes("13.233") || !(facts.match(/5.749/g)?.length >= 2)) {
    throw new Error("Initial dataset description is not 5749 reference photos");
  }
  const source = await page.locator(".dataset-source a").getAttribute("href");
  if (source !== "https://huggingface.co/datasets/marcelohaps/lfw") {
    throw new Error("Unexpected LFW dataset source link");
  }
  if (await page.locator("#zone-chart, #age-chart").count()) {
    throw new Error("Fixed distribution charts should have been removed");
  }
  await page.locator("#distribution-select").selectOption("language");
  const language = await page.locator("#distribution-title").textContent();
  if (!language.includes("idioma") || await page.locator("#distribution-chart .chart-line").count() !== 5) {
    throw new Error("Language distribution does not update");
  }
  await page.locator("#distribution-select").selectOption("age");
  const age = await page.locator("#distribution-title").textContent();
  if (!age.includes("edad") || await page.locator("#distribution-chart .chart-line").count() !== 5) {
    throw new Error("Age distribution does not update");
  }

  const photoButton = page.locator(".photo-zoom").first();
  await photoButton.waitFor();
  const thumbnailSrc = await photoButton.locator("img").getAttribute("src");
  await photoButton.click();
  await page.locator("#photo-dialog").waitFor({state:"visible"});
  if ((await page.locator("#photo-large").getAttribute("src")) !== thumbnailSrc) {
    throw new Error("The enlarged image must be the same reference photo");
  }
  await page.locator("#photo-close").click();
  if (await page.locator("#photo-dialog").isVisible()) throw new Error("Photo modal did not close");

  if (await page.locator(".question-item").count() !== 6) {
    throw new Error("Step 1 must contain six reflection questions");
  }
  if ((await page.locator("label[for='q5']").textContent()).includes("PROTECCIÓN DE DATOS")) {
    throw new Error("Old privacy label still displayed on question 5");
  }
  await page.locator("#q5b").fill("Conservar solo datos indispensables y definir plazos.");
  await page.reload({waitUntil:"domcontentloaded"});
  if (!(await page.locator("#q5b").inputValue()).includes("Conservar solo datos")) {
    throw new Error("Added privacy response was not persisted");
  }
  const nextDownload = page.waitForEvent("download");
  await page.locator("#export-answers").click();
  const downloaded = await nextDownload;
  const exportedText = await fs.readFile(await downloaded.path(),"utf8");
  if (!exportedText.includes("Conservar solo datos indispensables") ||
      !exportedText.includes("05 bis")) {
    throw new Error("Sixth question missing from exported worksheet");
  }
  await fs.mkdir("build",{recursive:true});
  await page.screenshot({path:"build/step1-desktop.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:"build/step1-mobile.png",fullPage:true});
  const bounds = await page.evaluate(()=>({
    viewport: innerWidth, content:document.documentElement.scrollWidth,
  }));
  if (bounds.content > bounds.viewport + 2) throw new Error("Step 1 mobile horizontal overflow");
  if (pageErrors.length) throw new Error("Browser errors: "+pageErrors.join(" | "));
  console.log("STEP1_BROWSER_SMOKE_SUCCESS",JSON.stringify({
    rows:10, distributions:["zone","language","age"],questions:6,
    modal:true,exported:true,viewport:bounds,
  }));
} finally {
  await browser.close();
}
