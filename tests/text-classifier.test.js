import test from "node:test";
import assert from "node:assert/strict";
import { tokenize, buildVectorizer, trainClassifier, evaluateClassifier, baselineLabel } from "../src/text-classifier.js";

const classes=["AGAINST","FAVOR","NONE"];
const samples={
  AGAINST:["condemn invasion attack violence","oppose invasion condemn violence","against aggression and violent attack","condemn war violence now"],
  FAVOR:["support aid solidarity freedom","support peace and assistance","support solidarity freedom help","celebrate solidarity and support"],
  NONE:["weather news headline morning","news weather report today","morning news about meetings","report and news bulletin"],
};
const train=classes.flatMap((label)=>samples[label].map((text)=>({
  text,russia:label,ukraine:label==="AGAINST"?"FAVOR":label==="FAVOR"?"AGAINST":"NONE",
})));
const val=[
  {text:"condemn invasion violence",russia:"AGAINST",ukraine:"FAVOR"},
  {text:"support solidarity freedom",russia:"FAVOR",ukraine:"AGAINST"},
  {text:"weather news morning",russia:"NONE",ukraine:"NONE"},
];

test("tokenization works with Unicode and optional adjacent bigrams",()=>{
  assert.deepEqual(tokenize("Слава Україна #peace",1),["слава","україна","peace"]);
  assert.deepEqual(tokenize("hello world",2),["hello","world","hello·world"]);
});

test("TF-IDF vocabulary is fitted only on training, sparse vectors are finite and normalized",()=>{
  const vectorizer=buildVectorizer(train,{ngrams:2,maxFeatures:200});
  assert.ok(vectorizer.terms.includes("condemn"));
  assert.ok(!vectorizer.terms.includes("extraneousvalidationterm"));
  const v=vectorizer.transform("condemn violence");
  assert.ok(v.indices.length>0);
  const norm=Math.hypot(...v.values);
  assert.ok(Math.abs(norm-1)<1e-5);
  assert.equal(vectorizer.transform("extraneousvalidationterm").indices.length,0);
});

for(const name of ["nb","centroid"]){
  test(`real classifier ${name} learns 3 classes and produces confusion statistics`,()=>{
    const vectorizer=buildVectorizer(train,{ngrams:1,maxFeatures:100});
    const model=trainClassifier(train,"russia",name,vectorizer,{alpha:1});
    const result=evaluateClassifier(model,val,"russia");
    assert.equal(result.total,3);
    assert.equal(result.matrix.flat().reduce((sum,x)=>sum+x,0),3);
    assert.ok(result.accuracy>=2/3,`Unexpected score ${result.accuracy} for ${name}`);
    assert.ok(result.macroF1>=0 && result.macroF1<=1);
    assert.equal(result.perClass.length,3);
    assert.equal(model.topTerms("AGAINST",3).length,3);
  });
}

test("majority baseline, alternate target and input validation",()=>{
  assert.equal(baselineLabel(train,"russia"),"AGAINST");
  const v=buildVectorizer(train,{maxFeatures:100});
  const model=trainClassifier(train,"ukraine","nb",v);
  assert.ok(classes.includes(model.predictText("invasion condemn").label));
  assert.throws(()=>trainClassifier(train,"russia","invented",v),/desconocido/);
  assert.throws(()=>buildVectorizer(train,{maxFeatures:0}),RangeError);
});
