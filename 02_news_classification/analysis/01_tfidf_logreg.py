"""뉴스 제목 TF-IDF + 로지스틱 회귀 주제 분류 (KLUE-YNAT, 7개 클래스)

학습: data/raw/ynat_train.json, 검증: data/raw/ynat_dev.json
입력 변수: 제목(title)만 사용. predefined_news_category, annotations, url, date, guid는
정답과 직접 연결되거나 식별자이므로 사용하지 않는다.
실행: 프로젝트 루트에서 python analysis/01_tfidf_logreg.py
"""
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
for p in (PROC, TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
CLASSES = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]


def load(name):
    with open(RAW / name, encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame({"guid": [r["guid"] for r in rows], "title": [r["title"] for r in rows],
                         "label": [r["label"] for r in rows]})


train, dev = load("ynat_train.json"), load("ynat_dev.json")

# ---- 데이터 점검 기록 (제외·대체한 데이터는 없음, 건수만 기록) ----
checks = pd.DataFrame([
    ("train 건수", len(train)), ("dev 건수", len(dev)),
    ("train 제목 결측/공백", int(train.title.fillna("").str.strip().eq("").sum())),
    ("dev 제목 결측/공백", int(dev.title.fillna("").str.strip().eq("").sum())),
    ("train 내 중복 제목(첫 건 제외)", int(train.title.duplicated().sum())),
    ("dev 제목 중 train에도 있는 건수", int(dev.title.isin(set(train.title)).sum())),
    ("train 클래스 수", train.label.nunique()), ("dev 클래스 수", dev.label.nunique()),
    ("제외한 행 수", 0),
], columns=["점검 항목", "값"])
checks.to_csv(TAB / "data_checks.csv", index=False, encoding="utf-8-sig")
print(checks.to_string(index=False))
assert set(train.label) == set(CLASSES) == set(dev.label)

X_train, y_train, X_dev, y_dev = train.title, train.label, dev.title, dev.label


def word_tfidf():
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def char_tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)


def make(kind, C):
    vec = {"word(1-2)": word_tfidf(), "char_wb(2-4)": char_tfidf(),
           "word+char": FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])}[kind]
    return Pipeline([("tfidf", vec), ("lr", LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE))])


# ---- 설정 선택: 학습 데이터 안에서만 3-fold 교차검증 (dev는 마지막 평가에만 사용) ----
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
rows = []
for kind in ["word(1-2)", "char_wb(2-4)", "word+char"]:
    for C in [10.0]:
        t0 = time.time()
        s = cross_val_score(make(kind, C), X_train, y_train, cv=cv, scoring="accuracy", n_jobs=3)
        rows.append((kind, C, s.mean(), s.std(), round(time.time() - t0)))
        print(kind, C, round(s.mean(), 4), f"{time.time() - t0:.0f}s", flush=True)
cvdf = pd.DataFrame(rows, columns=["TF-IDF 방식", "C", "교차검증 정답률 평균", "표준편차", "소요(초)"])
cvdf.to_csv(TAB / "cv_config_comparison.csv", index=False, encoding="utf-8-sig")
best = cvdf.sort_values("교차검증 정답률 평균", ascending=False).iloc[0]
print("선택:", best["TF-IDF 방식"], best["C"], flush=True)

# ---- 최종 학습(train 전체) 후 dev로 1회 평가 ----
model = make(best["TF-IDF 방식"], best["C"]).fit(X_train, y_train)
pred = model.predict(X_dev)
proba = model.predict_proba(X_dev)
acc = accuracy_score(y_dev, pred)
macro_f1 = f1_score(y_dev, pred, average="macro")
n_feat = len(model.named_steps["tfidf"].get_feature_names_out()) if hasattr(model.named_steps["tfidf"], "get_feature_names_out") else np.nan
print(f"dev accuracy={acc:.4f} macro-F1={macro_f1:.4f} features={n_feat}", flush=True)

pd.DataFrame([("TF-IDF 방식", best["TF-IDF 방식"]), ("로지스틱 회귀 C", best["C"]), ("특성 수", n_feat),
              ("학습 건수", len(train)), ("검증 건수", len(dev)),
              ("검증 정답률", round(acc, 4)), ("검증 macro F1", round(macro_f1, 4))],
             columns=["항목", "값"]).to_csv(TAB / "dev_summary.csv", index=False, encoding="utf-8-sig")

rep = pd.DataFrame(classification_report(y_dev, pred, labels=CLASSES, output_dict=True, zero_division=0)).T
rep.to_csv(TAB / "dev_classification_report.csv", encoding="utf-8-sig")

actual = y_dev.value_counts().reindex(CLASSES)
predicted = pd.Series(pred).value_counts().reindex(CLASSES).fillna(0).astype(int)
cnt = pd.DataFrame({"실제 건수": actual, "예측 건수": predicted})
cnt["예측-실제"] = cnt["예측 건수"] - cnt["실제 건수"]
cnt["예측 비율(%)"] = (cnt["예측 건수"] / cnt["예측 건수"].sum() * 100).round(1)
cnt.index.name = "주제"
cnt.loc["합계"] = [cnt["실제 건수"].sum(), cnt["예측 건수"].sum(), 0, 100.0]
cnt.to_csv(TAB / "dev_predicted_vs_actual_counts.csv", encoding="utf-8-sig")
print(cnt.to_string(), flush=True)

cm = confusion_matrix(y_dev, pred, labels=CLASSES)
pd.DataFrame(cm, index=[f"실제 {c}" for c in CLASSES], columns=[f"예측 {c}" for c in CLASSES]).to_csv(
    TAB / "dev_confusion_matrix.csv", encoding="utf-8-sig")

out = dev[["guid", "title", "label"]].copy()
out["pred"] = pred
out["pred_prob"] = proba.max(axis=1).round(4)
out["correct"] = out.label == out.pred
out.to_csv(PROC / "dev_predictions.csv", index=False, encoding="utf-8-sig")

plt.rcParams["font.family"] = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(7.5, 6.5))
cmn = cm / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(7), CLASSES, rotation=35, ha="right")
ax.set_yticks(range(7), CLASSES)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ax.set_xlabel("예측 주제")
ax.set_ylabel("실제 주제")
ax.set_title(f"검증 혼동행렬 (정답률 {acc:.1%})")
fig.colorbar(im, ax=ax, fraction=0.046, label="행 기준 비율")
fig.tight_layout()
fig.savefig(FIG / "dev_confusion_matrix.png", dpi=150)

pd.DataFrame([("python", platform.python_version()), ("scikit-learn", sklearn.__version__), ("pandas", pd.__version__),
              ("numpy", np.__version__), ("matplotlib", matplotlib.__version__), ("random_state", RANDOM_STATE)],
             columns=["패키지", "버전"]).to_csv(TAB / "environment.csv", index=False, encoding="utf-8-sig")
print("done", flush=True)
