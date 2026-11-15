"""Benchmark genuine pre-trained semantic sentence embeddings + supervised classifiers.
Validation-only; no held-out test labels or tweets are uploaded as artifacts.
"""
import json, time
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score

d=json.loads(Path("build/ruw-benchmark-input.json").read_text())
texts=[r["text"] for r in d["train"]+d["val"]]
models=[
  "sentence-transformers/all-MiniLM-L6-v2",
  "BAAI/bge-small-en-v1.5",
]
results=[]
for name in models:
  start=time.monotonic()
  print("ENCODER_START",name,flush=True)
  encoder=TextEmbedding(model_name=name)
  vectors=np.array(list(encoder.embed(texts,batch_size=32)),dtype=np.float32)
  xtrain=vectors[:len(d["train"])]
  xval=vectors[len(d["train"]):]
  elapsed=round(time.monotonic()-start,1)
  print("ENCODER_END",json.dumps({"name":name,"vectors":list(vectors.shape),"seconds":elapsed}),flush=True)
  for target in ["russia","ukraine"]:
    ytrain=np.array([r[target] for r in d["train"]])
    yval=np.array([r[target] for r in d["val"]])
    for method in ["svm","logistic","ridge"]:
      for C in [.1,1,10]:
        clf=(LinearSVC(C=C,max_iter=3000,dual=False)
          if method=="svm" else LogisticRegression(C=C,max_iter=300)
          if method=="logistic" else RidgeClassifier(alpha=C))
        clf.fit(xtrain,ytrain)
        pred=clf.predict(xval)
        score=dict(encoder=name,target=target,model=method,C=C,
          accuracy=round(float(accuracy_score(yval,pred)),5),
          macroF1=round(float(f1_score(yval,pred,average="macro")),5))
        print("SEMANTIC_SCORE",json.dumps(score),flush=True)
        results.append(score)
best={target:sorted((x for x in results if x["target"]==target),
 key=lambda v:(v["macroF1"],v["accuracy"]),reverse=True)[:5]
 for target in ["russia","ukraine"]}
print("SEMANTIC_BEST",json.dumps(best),flush=True)
Path("build/ruw-semantic-results.json").write_text(json.dumps(dict(best=best,all=results),indent=2))
