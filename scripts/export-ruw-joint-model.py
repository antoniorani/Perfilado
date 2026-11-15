"""Train a reproducible, public *model* (not posts) with real RUW train labels.
Selection was performed on DEV; TEST remains untouched by this artifact builder.
"""
import json, hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score,accuracy_score

DATA=Path("build/ruw-benchmark-input.json")
d=json.loads(DATA.read_text())
train=[r["text"] for r in d["train"]]
val=[r["text"] for r in d["val"]]
word=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True,max_features=40000)
char=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=40000)
wtr=word.fit_transform(train)
wval=word.transform(val)
ctr=char.fit_transform(train)
cval=char.transform(val)
X=hstack([wtr,ctr*.5],format="csr")
XV=hstack([wval,cval*.5],format="csr")
label=lambda r:r["russia"]+"/"+r["ukraine"]
clf=RidgeClassifier(alpha=1)
clf.fit(X,[label(r) for r in d["train"]])
pred=clf.predict(XV)
targets={}
for target,i in [("russia",0),("ukraine",1)]:
  true=[r[target] for r in d["val"]]
  guess=[row.split("/")[i] for row in pred]
  targets[target]=dict(accuracy=accuracy_score(true,guess),
    macroF1=f1_score(true,guess,average="macro"),
    correct=sum(a==b for a,b in zip(true,guess)),
    support=len(true))
print("MODEL_VALIDATION",json.dumps(targets),flush=True)
model={
 "format":"RUW-joint-ridge-1",
 "revision":"ecb2a4579197c5be03092a1b8fa21ead359809ce",
 "trainSize":len(train),
 "validationSize":len(val),
 "classNames":list(clf.classes_),
 "representation":[
  {"analyzer":"word","ngrams":[1,3],"vocabulary":word.get_feature_names_out().tolist(),
   "idf":np.round(word.idf_,7).tolist(),"scale":1},
  {"analyzer":"char_wb","ngrams":[3,5],"vocabulary":char.get_feature_names_out().tolist(),
   "idf":np.round(char.idf_,7).tolist(),"scale":.5},
 ],
 "coefficients":np.round(clf.coef_.astype(np.float32),7).tolist(),
 "intercepts":np.round(clf.intercept_.astype(np.float32),7).tolist(),
 "devValidation":targets,
 "devPredictions":list(map(str,pred)),
}
out=Path("data/ruw-joint-ridge-v1.json")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(model,ensure_ascii=False,separators=(",",":")))
digest=hashlib.sha256(out.read_bytes()).hexdigest()
print("MODEL_ARTIFACT",json.dumps({"path":str(out),"bytes":out.stat().st_size,
 "features":X.shape[1],"classes":len(clf.classes_),"sha256":digest}),flush=True)
