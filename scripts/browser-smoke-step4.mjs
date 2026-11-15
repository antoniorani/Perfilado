/** End-to-end check for static GitHub Pages Step 4, with no remote AI calls. */
import { chromium } from "playwright";
import fs from "node:fs/promises";

const browser = await chromium.launch({headless:true,args:["--no-sandbox"]});
const page = await browser.newPage({viewport:{width:1365,height:900},acceptDownloads:true});
const errors = [];
page.on("pageerror", error => errors.push(error.message));
try {
  await page.goto("http://localhost:8000/evaluacion.html",{waitUntil:"domcontentloaded"});
  await page.getByRole("heading",{name:"Una predicción. Muchas consecuencias."}).waitFor();

  if(await page.locator(".final-case").count() !== 3) throw new Error("Missing three fictitious cases");
  if(await page.locator(".question-item").count() !== 4) throw new Error("Missing four final questions");
  if((await page.locator("#joint-result").textContent()).trim() !== "29,7 %") {
    throw new Error("Default cascade result changed unexpectedly");
  }
  // Sliders must update the scenario and the dependence bounds.
  await page.locator("#face-rate").evaluate(el=>{el.value="50";el.dispatchEvent(new Event("input",{bubbles:true}));});
  await page.locator("#link-rate").evaluate(el=>{el.value="40";el.dispatchEvent(new Event("input",{bubbles:true}));});
  await page.locator("#text-rate").evaluate(el=>{el.value="30";el.dispatchEvent(new Event("input",{bubbles:true}));});
  const joint = (await page.locator("#joint-result").textContent()).trim();
  const bounds = (await page.locator("#bounds-result").textContent()).trim();
  if(!joint.startsWith("6") || !bounds.startsWith("0") || !bounds.includes("30")) {
    throw new Error("Incorrect dynamic cascade/bounds: "+joint+" / "+bounds);
  }

  // The student's answers stay in this browser and export through the shared worksheet.
  await page.locator("#q23").fill("Recomendación de prueba: no implantar con esta evidencia.");
  await page.reload({waitUntil:"domcontentloaded"});
  if(!(await page.locator("#q23").inputValue()).includes("no implantar")) {
    throw new Error("Step 4 answer did not persist locally");
  }
  const download = page.waitForEvent("download");
  await page.locator("#export-answers").click();
  const file = await download;
  if(file.suggestedFilename() !== "perfilado-paso4-respuestas.txt") {
    throw new Error("Step 4 worksheet download has unexpected name");
  }
  await fs.mkdir("build",{recursive:true});
  await page.screenshot({path:"build/step4-desktop.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:"build/step4-mobile.png",fullPage:true});
  const width = await page.evaluate(()=>({content:document.documentElement.scrollWidth,viewport:innerWidth}));
  if(width.content > width.viewport + 2) throw new Error("Horizontal overflow in Step 4 mobile layout: "+JSON.stringify(width));
  if(errors.length)throw new Error("Step 4 page errors: "+errors.join(" | "));
  console.log("STEP4_BROWSER_SMOKE_SUCCESS",JSON.stringify({cases:3,questions:4,joint,bounds,download:file.suggestedFilename(),mobileWidth:width}));
} finally {
  await browser.close();
}
