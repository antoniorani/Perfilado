/** Smoke-test the actual public RUW origin with Chromium and the real UI. */
import { chromium } from "playwright";
import fs from "node:fs/promises";

const browser = await chromium.launch({ headless: true, args:["--no-sandbox"] });
const page = await browser.newPage({ viewport:{width:1365,height:900} });
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => { if (message.type()==="error") errors.push(message.text()); });
try {
  await page.goto("http://localhost:8000/textos.html", {waitUntil:"domcontentloaded"});
  await page.getByText("Corpus RUW cargado correctamente").waitFor({timeout:45000});
  const samples = await page.locator(".tweet-card").count();
  if (samples !== 5) throw new Error(`Expected 5 examples; got ${samples}`);
  // The default is an on-device semantic model, not an offline trained artifact.
  // For fast CI smoke testing, train the existing fully local TF-IDF baseline.
  if (await page.locator("#model-select").inputValue() !== "semantic") {
    throw new Error("The on-device semantic model should be the default");
  }
  await page.locator("#model-select").selectOption("nb");
  await page.locator("#train-model").click();
  await page.getByText("Modelo entrenado:", {exact:false}).waitFor({timeout:45000});
  const score = (await page.locator("#metric-accuracy").textContent()).trim();
  const matrixRows = await page.locator("#confusion-matrix tbody tr").count();
  const baselineAccuracy = Number(score.replace("%", "").replace(",", ".")) / 100;
  if (!(baselineAccuracy > .3 && baselineAccuracy < .8) || matrixRows !== 3) {
    throw new Error(`Invalid browser-trained Naive Bayes validation score/matrix: ${score} / ${matrixRows}`);
  }
  await page.locator("#model-select").selectOption("centroid");
  if (!(await page.locator("#validation-results").isHidden())) {
    throw new Error("Model changes must invalidate old evaluation");
  }
  await page.locator("#target-select").selectOption("ukraine");
  await page.locator("#ngram-select").selectOption("2");
  await page.locator("#train-model").click();
  await page.getByText("Modelo entrenado:", {exact:false}).waitFor({timeout:30000});
  await page.locator("#manual-text").fill("Support peace and humanitarian help.");
  await page.locator("#manual-run").click();
  if (!(await page.locator("#manual-result").textContent()).includes("Ucrania")) {
    throw new Error("Custom text classification did not use selected target");
  }
  await page.locator("#reveal-test").click();
  if (await page.locator("#test-results").isHidden()) throw new Error("Held-out test did not render");
  const testTotal = await page.locator("#test-total").textContent();
  if (Number(testTotal.replaceAll(".","")) < 1100) throw new Error(`Unexpected held-out test size: ${testTotal}`);
  const matrix2 = await page.locator("#test-confusion tbody tr").count();
  if (matrix2 !== 3) throw new Error("Missing 3x3 final test confusion matrix");
  await fs.mkdir("build",{recursive:true});
  await page.screenshot({path:"build/ruw-step3-desktop.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:"build/ruw-step3-mobile.png",fullPage:true});
  console.log("BROWSER_SMOKE_SUCCESS",JSON.stringify({baselineValidationScore:score,testTotal,samples,matrixRows,matrix2,errors}));
  if (errors.length) throw new Error(`Unexpected browser console errors: ${errors.join(" | ").slice(0,700)}`);
} finally {
  await browser.close();
}
