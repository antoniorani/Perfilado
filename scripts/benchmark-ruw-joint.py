"""Evaluate multi-target structure without using test labels.
Only the original RUW training and dev sets are used for selecting models.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

d=json.loads(Path("build/ruw-benchmark-input.json").read_text())
train=[r["text"] for r in d["train"]]
val=[r["text"] for r in d["val"]]
Ytr=np.array([[r["russia"],r["ukraine"]] for r in d["train"]])
Yval=np.array([[r["russia"],r["ukraine"]] for r in d["val"]])
joint_train=np.array(["/".join(row) for row in Ytr])
trlabels=list(sorted(set(joint_train)))
w=TfidfVectorizer(ngram_range=(1,3),min_df=2,sublinear_tf=True,max_features=40000)
c=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=40000)
wtr=w.fit_transform(train);wv=w.transform(val)
ctr=c.fit_transform(train);cv=c.transform(val)
reprs={
 "word_1_3":(wtr,wv),
 "charwb_3_5":(ctr,cv),
 "word_char_1_1":(hstack([wtr,ctr],format="csr"),hstack([wv,cv],format="csr")),
 "word_char_1_05":(hstack([wtr,ctr*.5],format="csr"),hstack([wv,cv*.5],format="csr")),
}
out=[]
for repr_name,(trainX,valX) in reprs.items():
 for family,C in [("svm",.1),("svm",.3),("svm",1),("ridge",1)]:
  clf=(LinearSVC(C=C,dual=False,max_iter=3000) if family=="svm" else RidgeClassifier(alpha=C))
  clf.fit(trainX,joint_train)
  p=clf.predict(valX)
  joint_pred=np.array([x.split("/") for x in p])
  result={"features":int(trainX.shape[1]),"representation":repr_name,"family":family,"C":C,
   "jointAccuracy":round(float(np.mean(np.all(joint_pred==Yval,axis=1))),5)}
  for t,index in [("russia",0),("ukraine",1)]:
   result[t]={"accuracy":round(float(accuracy_score(Yval[:,index],joint_pred[:,index])),5),
              "macroF1":round(float(f1_score(Yval[:,index],joint_pred[:,index],average="macro")),5)}
  out.append(result)
  print("MULTI_TARGET",json.dumps(result),flush=True)
ranked=sorted(out,key=lambda x:(x["russia"]["macroF1"]+x["ukraine"]["macroF1"])/2,reverse=True)
print("MULTI_TARGET_BEST",json.dumps(ranked[:8]),flush=True)
Path("build/ruw-multi-target-results.json").write_text(json.dumps(dict(all=out,ranked=ranked[:8]),indent=2))
