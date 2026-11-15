/** Generate ephemeral, de-identified RUW splits for a model benchmark.
 * No dataset rows are committed or uploaded as artifacts.
 */
import fs from "node:fs/promises";
import { loadRUW } from "../src/ruw-data.js";
const corpus=await loadRUW();
await fs.mkdir("./build", {recursive:true});
await fs.writeFile("./build/ruw-benchmark-input.json",JSON.stringify(corpus));
console.log("RUW_DATA",JSON.stringify({
  original:corpus.originalCounts,
  clean:Object.fromEntries(["train","val","test"].map(split=>[split,corpus[split].length])),
  removed:corpus.removed
}));
