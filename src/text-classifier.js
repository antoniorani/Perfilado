/**
 * Real (non-generative) NLP models running entirely in the browser.
 * The sparse TF-IDF representation is fitted on TRAIN ONLY, never validation/test.
 * Naive Bayes and cosine nearest centroid are trained classifiers, not mock scores.
 */
import { LABELS } from "./ruw-data.js";

export const MODEL_NAMES = Object.freeze({
  nb: "Naive Bayes multinomial",
  centroid: "Prototipos por similitud coseno",
});
const LABEL_INDEX = new Map(LABELS.map((name, i) => [name, i]));

export function tokenize(text, ngrams = 1) {
  if (![1, 2].includes(ngrams)) throw new RangeError("N-gramas inválidos");
  const words = String(text).normalize("NFKC").toLocaleLowerCase("en")
    .match(/[\p{L}\p{N}]+(?:['’][\p{L}]+)?/gu)?.filter((word)=>word.length>=2) || [];
  return ngrams === 1 ? words : [...words, ...words.slice(1).map((w,i)=>`${w}·${words[i+1]}`)];
}

export function buildVectorizer(rows, { ngrams = 1, maxFeatures = 2500 } = {}) {
  if (!Number.isInteger(maxFeatures) || maxFeatures < 10 || maxFeatures > 10000) {
    throw new RangeError("Tamaño de vocabulario inválido");
  }
  if (!Array.isArray(rows) || !rows.length) throw new Error("Entrenamiento vacío");
  const documentFreq = new Map();
  for (const row of rows) {
    const unique = new Set(tokenize(row.text, ngrams));
    for (const word of unique) documentFreq.set(word,(documentFreq.get(word)||0)+1);
  }
  const vocab = [...documentFreq].filter(([,count])=>count>=2 && count<=rows.length*.98)
    .sort((a,b)=>b[1]-a[1] || a[0].localeCompare(b[0],"en"))
    .slice(0,maxFeatures);
  if (!vocab.length) throw new Error("No se encontraron términos repetidos en el entrenamiento");
  const terms = vocab.map(([word])=>word);
  const indices = new Map(terms.map((word,index)=>[word,index]));
  const idf = Float32Array.from(vocab,([,df])=>Math.log((1+rows.length)/(1+df))+1);

  return {
    terms, size:terms.length, ngrams, trainingSize:rows.length,
    transform(text) {
      const tf = new Map();
      for (const word of tokenize(text,ngrams)) {
        const index = indices.get(word);
        if (index!==undefined) tf.set(index,(tf.get(index)||0)+1);
      }
      let norm=0;
      const pairs=[];
      for (const [index,count] of tf) {
        const value=(1+Math.log(count))*idf[index];
        norm+=value*value;
        pairs.push([index,value]);
      }
      norm=Math.sqrt(norm)||1;
      pairs.sort((a,b)=>a[0]-b[0]);
      return { indices:Uint16Array.from(pairs,([index])=>index),
        values:Float32Array.from(pairs,([,value])=>value/norm) };
    },
  };
}

function countsFor(rows,target) {
  const counts= new Float64Array(LABELS.length);
  for(const row of rows) {
    const index=LABEL_INDEX.get(row[target]);
    if(index===undefined)throw new Error("Etiqueta de entrenamiento desconocida");
    counts[index]+=1;
  }
  if(counts.some((value)=>value===0))throw new Error("Falta una clase en entrenamiento");
  return counts;
}

export function trainClassifier(rows, target, modelName, vectorizer, {alpha=1}={}) {
  if(!["russia","ukraine"].includes(target))throw new Error("Objetivo desconocido");
  if(!Object.hasOwn(MODEL_NAMES,modelName))throw new Error("Clasificador desconocido");
  if(!Number.isFinite(alpha)||alpha<=0||alpha>100)throw new RangeError("Suavizado inválido");
  const n=vectorizer.size,k=LABELS.length;
  const classCounts=countsFor(rows,target);
  const weights= Array.from({length:k},()=>new Float64Array(n));
  const classTotals=new Float64Array(k);
  for(const row of rows){
    const c=LABEL_INDEX.get(row[target]);
    const embedding=vectorizer.transform(row.text);
    for(let i=0;i<embedding.indices.length;i++){
      const term=embedding.indices[i],w=embedding.values[i];
      weights[c][term]+=w;
      classTotals[c]+=w;
    }
  }

  if(modelName==="nb"){
    for(let c=0;c<k;c++){
      const denom=classTotals[c]+alpha*n;
      for(let j=0;j<n;j++) weights[c][j]=Math.log((weights[c][j]+alpha)/denom);
    }
  }else{
    // A centroid is the L2-normalized mean of each class's TF-IDF vectors.
    for(let c=0;c<k;c++){
      let norm=0;
      for(let j=0;j<n;j++)norm+=weights[c][j]*weights[c][j];
      norm=Math.sqrt(norm)||1;
      for(let j=0;j<n;j++)weights[c][j]/=norm;
    }
  }

  const priors= Array.from(classCounts,(count)=>Math.log(count/rows.length));
  return {
    name:modelName,target, vectorizer,
    predictText(text) {
      const v=vectorizer.transform(text);
      const scores=weights.map((classWeights,c)=>{
        let score=modelName==="nb"?priors[c]:0;
        for(let i=0;i<v.indices.length;i++)score+=v.values[i]*classWeights[v.indices[i]];
        return score;
      });
      let best=0;
      for(let c=1;c<k;c++)if(scores[c]>scores[best])best=c;
      return { label:LABELS[best],scores,knownTerms:v.indices.length };
    },
    topTerms(label, count=6) {
      const c=LABEL_INDEX.get(label);
      if(c===undefined)throw new Error("Clase desconocida");
      const ranked=vectorizer.terms.map((term,index)=>{
        const other=weights.reduce((sum,w,idx)=>sum+(idx===c?0:w[index]),0)/(k-1);
        return {term, weight:weights[c][index]-other};
      }).sort((a,b)=>b.weight-a.weight);
      return ranked.slice(0,count);
    },
  };
}

export function baselineLabel(rows,target) {
  const counts=countsFor(rows,target);
  let best=0;
  for(let i=1;i<counts.length;i++)if(counts[i]>counts[best])best=i;
  return LABELS[best];
}

export function evaluateClassifier(model, rows, target) {
  const matrix=Array.from({length:LABELS.length},()=>Array(LABELS.length).fill(0));
  const decisions=[];
  for(const row of rows){
    const prediction=model.predictText(row.text);
    const actual=LABEL_INDEX.get(row[target]);
    const predicted=LABEL_INDEX.get(prediction.label);
    if(actual===undefined||predicted===undefined)throw new Error("Etiqueta inesperada");
    matrix[actual][predicted]+=1;
    decisions.push({row,prediction:prediction.label,correct:actual===predicted,knownTerms:prediction.knownTerms});
  }
  const perClass=LABELS.map((label,index)=>{
    const tp=matrix[index][index];
    const support=matrix[index].reduce((sum,n)=>sum+n,0);
    const predicted=matrix.reduce((sum,row)=>sum+row[index],0);
    const precision=predicted?tp/predicted:0;
    const recall=support?tp/support:0;
    const f1=precision+recall?2*precision*recall/(precision+recall):0;
    return {label,support,predicted,precision,recall,f1};
  });
  const correct=decisions.filter((r)=>r.correct).length;
  const accuracy=decisions.length?correct/decisions.length:null;
  const macroF1=perClass.reduce((sum,r)=>sum+r.f1,0)/perClass.length;
  return {matrix,decisions,perClass,correct,total:decisions.length,accuracy,macroF1};
}
