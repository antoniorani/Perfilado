/** Recompute DEV predictions from the distributed model in pure JavaScript.
 * Compare to scikit-learn predictions without using held-out test labels.
 */
import fs from "node:fs/promises";
import { loadRUW } from "../src/ruw-data.js";
import { createRidgeModel } from "../src/ruw-ridge.js";
import { evaluateClassifier } from "../src/text-classifier.js";

const data=await loadRUW();
const modelJson=JSON.parse(await fs.readFile(new URL("../data/ruw-joint-ridge-v1.json",import.meta.url),"utf8"));
if(data.train.length!==modelJson.trainSize || data.val.length!==modelJson.validationSize){
  throw new Error("RUW dataset and exported Ridge model are out of sync");
}
const model=createRidgeModel(modelJson,"russia");
const disagreement=[];
for(let i=0;i<data.val.length;i++){
  const actual=model.predictText(data.val[i].text);
  const expected=modelJson.devPredictions[i]?.split("/");
  if(!expected || actual.jointLabels.russia!==expected[0] || actual.jointLabels.ukraine!==expected[1]){
    disagreement.push({index:i,js:actual.jointLabels,sklearn:expected,featureCount:actual.knownTerms});
  }
}
console.log("MODEL_PARITY",JSON.stringify({validation:data.val.length,
  disagreements:disagreement.length,examples:disagreement.slice(0,8)}));
for(const target of ["russia","ukraine"]){
  const result=evaluateClassifier(createRidgeModel(modelJson,target),data.val,target);
  console.log("MODEL_TARGET_METRICS",JSON.stringify({target,accuracy:result.accuracy,macroF1:result.macroF1}));
  const expected=modelJson.devValidation[target];
  if(Math.abs(result.accuracy-expected.accuracy)>0.025 ||
     Math.abs(result.macroF1-expected.macroF1)>0.03){
    throw new Error(`Inconsistent JS model for ${target} versus scikit-learn`);
  }
}
if(disagreement.length>15)throw new Error("Too many JS/sklearn inference disagreements");
