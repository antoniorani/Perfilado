import test from "node:test";
import assert from "node:assert/strict";
import { createRidgeModel } from "../src/ruw-ridge.js";
import { RUW_REVISION } from "../src/ruw-data.js";

function fixture() {
  const classNames=["AGAINST","FAVOR","NONE"].flatMap(r=>
    ["AGAINST","FAVOR","NONE"].map(u=>r+"/"+u));
  const features=4,raw=new Int16Array(classNames.length*features);
  raw[classNames.indexOf("FAVOR/AGAINST")*features]=1000;
  raw[classNames.indexOf("AGAINST/FAVOR")*features+1]=1000;
  return {
    format:"RUW-joint-ridge-1",revision:RUW_REVISION,
    classNames,
    representation:[
      {analyzer:"word",ngrams:[1,3],vocabulary:["support","condemn"],idf:[1,1],scale:1},
      {analyzer:"char_wb",ngrams:[3,5],vocabulary:[" sup"," con"],idf:[1,1],scale:.5},
    ],
    weightsI16:{shape:[classNames.length,features],
      scale:Array(classNames.length).fill(.001),
      bytes:Buffer.from(raw.buffer).toString("base64")},
    intercepts:Array(classNames.length).fill(0),
  };
}

test("Ridge inference reconstructs learned int16 weights and returns both stance targets",()=>{
  const payload=fixture();
  const russia=createRidgeModel(payload,"russia");
  const ukraine=createRidgeModel(payload,"ukraine");
  const support=russia.predictText("support");
  assert.equal(support.label,"FAVOR");
  assert.deepEqual(support.jointLabels,{russia:"FAVOR",ukraine:"AGAINST"});
  assert.equal(ukraine.predictText("support").label,"AGAINST");
  assert.deepEqual(russia.predictText("condemn").jointLabels,
    {russia:"AGAINST",ukraine:"FAVOR"});
  assert.equal(russia.vectorizer.size,4);
  assert.equal(russia.topTerms("FAVOR",1)[0].term,"support");
});

test("Ridge rejects changed dataset, malformed templates and unexpected targets",()=>{
  const payload=fixture();
  assert.throws(()=>createRidgeModel({...payload,revision:"other"},"russia"),/incompatible/);
  assert.throws(()=>createRidgeModel(payload,"wrong"),/incompatible/);
  assert.throws(()=>createRidgeModel({...payload,weightsI16:{...payload.weightsI16,
    shape:[9,5]}},"russia"),/Dimensiones/);
});
