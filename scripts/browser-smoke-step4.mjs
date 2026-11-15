/** End-to-end check for static GitHub Pages Step 4, with no remote AI calls. */
import { chromium } from "playwright";
import fs from "node:fs/promises";

const browser = await chromium.launch({headless:true,args:["--no-sandbox"]});
const page = await browser.newPage({viewport:{width:1365,height:900},acceptDownloads:true});
const errors = [];
page.on("pageerror", error => errors.push(error.message));
try {
  await page.goto("http://localhost:8000/evaluacion.html",{waitUntil:"domcontentloaded"});
  await page.getByRole("heading",{name:"Ahora sabemos para qué era todo."}).waitFor();

  if(await page.locator(".final-sequence > div").count() !== 3) throw new Error("Missing three steps in the fictional brief");
  if(await page.locator("#link-rate").count() !== 0) throw new Error("Obsolete linkage slider is still present");
  if(await page.locator(".question-item").count() !== 4) throw new Error("Missing four final questions");
  if((await page.locator("#joint-result").textContent()).trim() !== "42,4 %") {
    throw new Error("Default cascade result changed unexpectedly");
  }
  // Sliders must update the scenario and the dependence bounds.
  await page.locator("#face-rate").evaluate(el=>{el.value="50";el.dispatchEvent(new Event("input",{bubbles:true}));});
  await page.locator("#text-rate").evaluate(el=>{el.value="30";el.dispatchEvent(new Event("input",{bubbles:true}));});
  const joint = (await page.locator("#joint-result").textContent()).trim();
  const bounds = (await page.locator("#bounds-result").textContent()).trim();
  if(!joint.startsWith("15") || !bounds.startsWith("0") || !bounds.includes("30")) {
    throw new Error("Incorrect dynamic cascade/bounds: "+joint+" / "+bounds);
  }

  // The student's answers stay in this browser without visible export/status controls.
  await page.locator("#q23").fill("Recomendación de prueba: no implantar con esta evidencia.");
  await page.reload({waitUntil:"domcontentloaded"});
  if(!(await page.locator("#q23").inputValue()).includes("no implantar")) {
    throw new Error("Step 4 answer did not persist locally");
  }
  if (await page.locator("#save-state, #export-answers, .question-actions").count()) {
    throw new Error("Step 4 should save silently and show no export controls");
  }
  if ((await page.locator("label[for='q22']").innerText()).includes("PROTECCIÓN DE DATOS")) {
    throw new Error("Redundant privacy label remains on the question");
  }
  await fs.mkdir("build",{recursive:true});
  await page.screenshot({path:"build/step4-desktop.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:"build/step4-mobile.png",fullPage:true});
  const width = await page.evaluate(()=>({content:document.documentElement.scrollWidth,viewport:innerWidth}));
  if(width.content > width.viewport + 2) throw new Error("Horizontal overflow in Step 4 mobile layout: "+JSON.stringify(width));
  if(errors.length)throw new Error("Step 4 page errors: "+errors.join(" | "));
  console.log("STEP4_BROWSER_SMOKE_SUCCESS",JSON.stringify({stages:3,questions:4,joint,bounds,silentSave:true,mobileWidth:width}));
} finally {
  await browser.close();
}
