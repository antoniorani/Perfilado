import test from "node:test";
import assert from "node:assert/strict";
import { parseCSV, parseRUWSplit, redactTweet, prepareRUW, labelCounts, normalizedText, splitTrainValidation } from "../src/ruw-data.js";

test("student split is reproducible, disjoint and covers the whole corpus", () => {
  const rows=Array.from({length:100},(_,i)=>({id:i}));
  const first=splitTrainValidation(rows,.8);
  assert.equal(first.train.length,80);
  assert.equal(first.val.length,20);
  assert.deepEqual(splitTrainValidation(rows,.8),first);
  const ids=[...first.train,...first.val].map(r=>r.id).sort((a,b)=>a-b);
  assert.deepEqual(ids,rows.map(r=>r.id));
  assert.notDeepEqual(first.train.map(r=>r.id),rows.slice(0,80).map(r=>r.id));
  assert.throws(()=>splitTrainValidation(rows,1),/inválida/);
});

test("RFC-4180 parser supports comma, quoted newline, escapes and CRLF", () => {
  const csv='Text,Value\r\n"one, two","a ""quote"""\r\n"two\nlines",NONE\r\n';
  assert.deepEqual(parseCSV(csv),[
    ["Text","Value"],["one, two",'a "quote"'],["two\nlines","NONE"],
  ]);
  assert.throws(()=>parseCSV('"broken'),/sin cerrar/);
});

test("RUW parser preserves two independent stance annotations", () => {
  const raw=[
    "Tweet,Target 1,Stance 1,Target 2,Stance 2,Test/Train/Dev",
    '"Text @user with https://example.org",Russia,AGAINST,Ukraine,FAVOR,Train',
    '"This is a title",Russia,NONE,Ukraine,NONE,Train',
  ].join("\r\n");
  const rows=parseRUWSplit(raw,"train");
  assert.equal(rows.length,2);
  assert.deepEqual([rows[0].russia,rows[0].ukraine],["AGAINST","FAVOR"]);
  assert.match(rows[0].text,/\[usuario\]/);
  assert.match(rows[0].text,/\[enlace\]/);
  assert.equal(rows[0].split,"train");
  assert.equal(rows[0].id,"RUW-TRAIN-0001");
  assert.throws(()=>parseRUWSplit(raw.replace("AGAINST","OTHER"),"train"),/no válida/);
});

test("redaction hides social handles, URLs and emails, without claiming full anonymity", () => {
  const result=redactTweet("Hi @someone https://example.org/x mail@example.com");
  assert.doesNotMatch(result,/@someone|https:|mail@example.com/);
  assert.equal(normalizedText(" YES \n  NO "), "yes no");
});

test("duplicate removal favors held-out test, then validation, before training", () => {
  const example=(text,split)=>({text,split,russia:"NONE",ukraine:"FAVOR"});
  const corpus=prepareRUW({
    train:[example("same","train"),example("training unique","train"),example("training unique","train")],
    val:[example("same","val"),example("validation unique","val")],
    test:[example("same","test"),example("test unique","test")],
  });
  assert.deepEqual(corpus.removed,{train:2,val:1,test:0});
  assert.deepEqual(corpus.train.map(x=>x.text),["training unique"]);
  assert.deepEqual(corpus.val.map(x=>x.text),["validation unique"]);
  assert.deepEqual(corpus.test.map(x=>x.text),["same","test unique"]);
  assert.deepEqual(labelCounts(corpus.test,"ukraine"),{AGAINST:0,FAVOR:2,NONE:0});
});
