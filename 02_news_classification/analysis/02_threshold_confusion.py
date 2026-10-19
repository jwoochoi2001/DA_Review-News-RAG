"""임계값별 혼동행렬 (검증 데이터)

01_tfidf_logreg.py 가 저장한 data/processed/dev_predictions.csv 를 사용한다.
예측 확률(가장 높은 클래스의 확률)이 임계값보다 낮으면 '보류'로 두고 혼동행렬의 별도 열에 센다.
실행: 프로젝트 루트에서 python analysis/02_threshold_confusion.py
"""
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
CLASSES = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
HOLD = "보류"
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

d = pd.read_csv(ROOT / "data" / "processed" / "dev_predictions.csv", encoding="utf-8-sig")
print("예측 확률 분위수:", d.pred_prob.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())


def matrix(th):
    pred = np.where(d.pred_prob >= th, d.pred, HOLD)
    cm = pd.crosstab(pd.Categorical(d.label, CLASSES), pd.Categorical(pred, CLASSES + [HOLD]), dropna=False)
    cm.index = [f"실제 {c}" for c in CLASSES]
    cm.columns = [f"예측 {c}" for c in CLASSES] + ["보류"]
    return cm, pred


rows, mats = [], {}
for th in THRESHOLDS:
    cm, pred = matrix(th)
    mats[th] = cm
    cm.to_csv(TAB / f"dev_confusion_th_{th:.1f}.csv", encoding="utf-8-sig")
    keep = pred != HOLD
    correct = (pred == d.label.values)
    rows.append((th, int(keep.sum()), int((~keep).sum()), round(keep.mean() * 100, 1),
                 round(correct[keep].mean() * 100, 1) if keep.any() else np.nan,
                 round(correct.mean() * 100, 1)))
summ = pd.DataFrame(rows, columns=["임계값", "분류 건수", "보류 건수", "분류 비율(%)", "분류된 건의 정답률(%)", "전체 정답률(보류=오답, %)"])
base = pd.DataFrame([(0.0, len(d), 0, 100.0, round(d.correct.mean() * 100, 1), round(d.correct.mean() * 100, 1))], columns=summ.columns)
summ = pd.concat([base, summ], ignore_index=True)
summ.to_csv(TAB / "dev_threshold_summary.csv", index=False, encoding="utf-8-sig")
print(summ.to_string(index=False))

# 보류 건이 어떤 실제 주제에 몰리는지
hold_rows = []
for th in THRESHOLDS:
    h = mats[th]["보류"]
    hold_rows.append([th] + [int(h.iloc[i]) for i in range(7)])
hold = pd.DataFrame(hold_rows, columns=["임계값"] + CLASSES)
hold.to_csv(TAB / "dev_threshold_hold_by_true_class.csv", index=False, encoding="utf-8-sig")
print(hold.to_string(index=False))

# 그림: 임계값별 혼동행렬 (행 기준 비율 색, 칸에는 건수)
plt.rcParams["font.family"] = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
ths = [0.0] + THRESHOLDS
fig, axes = plt.subplots(2, 3, figsize=(20, 12.5))
for ax, th in zip(axes.ravel(), ths):
    cm = mats[th] if th else matrix(0.0)[0]
    arr = cm.values
    norm = arr / arr.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8), CLASSES + [HOLD], rotation=35, ha="right")
    ax.set_yticks(range(7), CLASSES)
    for i in range(7):
        for j in range(8):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.5 else "black")
    r = summ[summ.임계값 == th].iloc[0]
    ax.set_title(f"임계값 {th:.1f} · 분류 {r['분류 비율(%)']:.0f}% · 분류된 건 정답률 {r['분류된 건의 정답률(%)']:.1f}%", fontsize=11)
    ax.set_xlabel("예측 주제")
    ax.set_ylabel("실제 주제")
fig.suptitle("검증 혼동행렬 (예측 확률이 임계값 미만이면 '보류')", fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "dev_confusion_by_threshold.png", dpi=130)
print("done")
