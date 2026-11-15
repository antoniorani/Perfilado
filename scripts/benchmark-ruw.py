"""Compare real text classifiers on the exact RUW train/dev split.
Never touch the held-out test labels in model selection. Output only aggregates.
"""
import json, time, warnings
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

data=json.loads(Path("build/ruw-benchmark-input.json").read_text())
train=[r["text"] for r in data["train"]]
valid=[r["text"] for r in data["val"]]
configs={
 "word_1_2":dict(analyzer="word",ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=20000),
 "word_1_3":dict(analyzer="word",ngram_range=(1,3),min_df=2,sublinear_tf=True,max_features=40000),
 "char_3_5":dict(analyzer="char",ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=50000),
 "charwb_3_5":dict(analyzer="char_wb",ngram_range=(3,5),min_df=3,sublinear_tf=True,max_features=50000),
 "word_1_2_5000":dict(analyzer="word",ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=5000),
}
models={
 "LinearSVC_C0.3":lambda: LinearSVC(C=.3,max_iter=3000,dual=False),
 "LinearSVC_C1":lambda: LinearSVC(C=1,max_iter=3000,dual=False),
 "LinearSVC_C3":lambda: LinearSVC(C=3,max_iter=3000,dual=False),
 "Logistic_C1":lambda: LogisticRegression(C=1,max_iter=300,solver="lbfgs"),
 "Logistic_C4":lambda: LogisticRegression(C=4,max_iter=300,solver="lbfgs"),
 "Ridge_a1":lambda: RidgeClassifier(alpha=1),
 "SGD_log":lambda: SGDClassifier(loss="log_loss",alpha=1e-4,max_iter=1000,random_state=42),
}
results=[]
for config_name, config in configs.items():
 start=time.monotonic()
 tfidf=TfidfVectorizer(**config)
 X_train=tfidf.fit_transform(train)
 X_val=tfidf.transform(valid)
 print("VECTOR",json.dumps({"representation":config_name,
  "features":len(tfidf.vocabulary_),"elapsed":round(time.monotonic()-start,2)}),flush=True)
 for target in ["russia","ukraine"]:
  ytrain=[r[target] for r in data["train"]]
  yval=[r[target] for r in data["val"]]
  for model_name,make in models.items():
   start=time.monotonic()
   with warnings.catch_warnings():
    warnings.simplefilter("ignore",FutureWarning)
    clf=make()
    clf.fit(X_train,ytrain)
   pred=clf.predict(X_val)
   row=dict(target=target,representation=config_name,model=model_name,
      accuracy=round(float(accuracy_score(yval,pred)),5),
      macroF1=round(float(f1_score(yval,pred,average="macro")),5),
      features=len(tfidf.vocabulary_),seconds=round(time.monotonic()-start,3))
   results.append(row)
   print("SCORE",json.dumps(row),flush=True)
ranked={target:sorted((r for r in results if r["target"]==target),
 key=lambda r:(r["macroF1"],r["accuracy"]),reverse=True)[:12] for target in ["russia","ukraine"]}
print("BEST_MODELS",json.dumps(ranked),flush=True)
Path("build/ruw-benchmark-results.json").write_text(json.dumps(dict(all=results,top=ranked),indent=2))
