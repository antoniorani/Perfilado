/** Live audit against the pinned, original 6,000-post RUW dataset.
 * Runs in CI; only aggregated scores are printed or kept.
 */
import { RUW_REVISION, loadRUW, LABELS, labelCounts } from "../src/ruw-data.js";
import { buildVectorizer, trainClassifier, evaluateClassifier, baselineLabel } from "../src/text-classifier.js";

const source=await loadRUW();
const counts=Object.fromEntries(["train","val","test"].map(name=>[name,source[name].length]));
const original=source.originalCounts;
if(original.train!==4200 || original.val!==600 || original.test!==1200)
  throw new Error("Original RUW split counts changed");

const report={revision:RUW_REVISION,original,clean:counts,removed:source.removed,
  targets:{},note:"Actual statistical NLP models, fitted ONLY on train. No test labels used in tuning."};
for(const target of ["russia","ukraine"]){
  const classes=labelCounts(source.train,target);
  const majority=baselineLabel(source.train,target);
  const targetModels={};
  for(const method of ["nb","centroid"]){
    const vectorizer=buildVectorizer(source.train,{ngrams:1,maxFeatures:2500});
    const model=trainClassifier(source.train,target,method,vectorizer,{alpha:1});
    const val=evaluateClassifier(model,source.val,target);
    targetModels[method]={accuracy:val.accuracy,macroF1:val.macroF1,
      correct:val.correct,total:val.total,vocabulary:vectorizer.size,
      perClass:val.perClass.map(x=>({label:x.label,support:x.support,f1:x.f1}))};
  }
  const baseline=source.val.filter(x=>x[target]===majority).length/source.val.length;
  report.targets[target]={classes,majority,baseline,models:targetModels};
  console.log("RUW_REAL_MODEL",JSON.stringify({target,baseline,models:targetModels}));
  if(!targetModels.nb.accuracy || !targetModels.centroid.accuracy)
    throw new Error("No valid classification scores produced");
}
console.log("RUW_DATASET_AUDIT",JSON.stringify({revision:RUW_REVISION,
  original,clean:counts,duplicatesExcluded:source.removed,labels:LABELS}));
