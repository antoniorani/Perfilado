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
  if (!(await page.locator('input[name="model"][value="semantic"]').isChecked())) {
    throw new Error("The on-device semantic model should be the default");
  }
  if (await page.locator('[data-models="semantic"].model-explainer').isHidden()) {
    throw new Error("The selected model must show its visual explanation");
  }
  await page.locator('input[name="model"][value="nb"]').check({force:true});
  if (await page.locator('[data-models="nb"].model-explainer').isHidden()) {
    throw new Error("Naive Bayes explanation did not appear");
  }
  await page.locator("#train-model").click();
  await page.getByText("Modelo entrenado:", {exact:false}).waitFor({timeout:45000});
  const score = (await page.locator("#metric-accuracy").textContent()).trim();
  const matrixRows = await page.locator("#confusion-matrix tbody tr").count();
  const baselineAccuracy = Number(score.replace("%", "").replace(",", ".")) / 100;
  if (!(baselineAccuracy > .3 && baselineAccuracy < .8) || matrixRows !== 3) {
    throw new Error(`Invalid browser-trained Naive Bayes validation score/matrix: ${score} / ${matrixRows}`);
  }
  await page.locator('input[name="model"][value="centroid"]').check({force:true});
  if (!(await page.locator("#validation-results").isHidden())) {
    throw new Error("Model changes must invalidate old evaluation");
  }
  await page.locator("#target-select").selectOption("ukraine");
  await page.locator("#ngram-select").selectOption("2");
  await page.locator("#split-range").fill("70");
  const trainCount = Number((await page.locator("#split-train-count").textContent()).replaceAll(".",""));
  const valCount = Number((await page.locator("#split-val-count").textContent()).replaceAll(".",""));
  if (trainCount + valCount < 5800 || Math.abs(trainCount / (trainCount + valCount) - .7) > .01) {
    throw new Error(`Student split was not applied: ${trainCount}/${valCount}`);
  }
  await page.locator("#train-model").click();
  await page.getByText("Modelo entrenado:", {exact:false}).waitFor({timeout:30000});
  await page.locator("#manual-text").fill("Support peace and humanitarian help.");
  await page.locator("#manual-run").click();
  if (!(await page.locator("#manual-result").textContent()).includes("Ucrania")) {
    throw new Error("Custom text classification did not use selected target");
  }
  // Real MiniLM + ONNX smoke test: small fixture, still trains in Chromium.
  // This checks the CDN, worker protocol and encoder tensor dimensions.
  const semantic = await page.evaluate(() => new Promise((resolve,reject) => {
    const worker = new Worker("./src/semantic-worker.js", {type:"module"});
    const timeout = setTimeout(() => {worker.terminate();reject(new Error("MiniLM worker timed out"));},180000);
    const finish = (error,result) => {
      clearTimeout(timeout);
      worker.terminate();
      error ? reject(error) : resolve(result);
    };
    worker.onerror = (event) => finish(new Error(event.message || "MiniLM worker error"));
    worker.onmessage = ({data}) => {
      if(data.type==="error")finish(new Error(data.message));
      if(data.type==="result")finish(null,data.result);
    };
    worker.postMessage({
      id:1,type:"train",target:"russia",
      train:[
        {text:"Condemn the violent invasion",label:"AGAINST"},
        {text:"I reject this invasion",label:"AGAINST"},
        {text:"Oppose the military attack",label:"AGAINST"},
        {text:"I support this action",label:"FAVOR"},
        {text:"I endorse this action",label:"FAVOR"},
        {text:"I am in favor of the action",label:"FAVOR"},
        {text:"It is sunny today",label:"NONE"},
        {text:"The weather is good",label:"NONE"},
        {text:"The event begins at noon",label:"NONE"},
      ],
      val:["I reject the military attack","I support this decision","The weather today"],
    });
  }));
  if(semantic.dimension!==384 || semantic.predictions.length!==3 ||
      semantic.predictions.some(label=>!["AGAINST","FAVOR","NONE"].includes(label))) {
    throw new Error("Real MiniLM worker did not produce valid embeddings/predictions");
  }
  console.log("BROWSER_SEMANTIC_SMOKE_SUCCESS",semantic.predictions);
  await fs.mkdir("build",{recursive:true});
  await page.screenshot({path:"build/ruw-step3-desktop.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:"build/ruw-step3-mobile.png",fullPage:true});
  console.log("BROWSER_SMOKE_SUCCESS",JSON.stringify({baselineValidationScore:score,trainCount,valCount,samples,matrixRows,errors}));
  if (errors.length) throw new Error(`Unexpected browser console errors: ${errors.join(" | ").slice(0,700)}`);
} finally {
  await browser.close();
}
