"""
make_figures.py — README에 들어가는 차트를 저장소의 결과 파일(CSV)에서 다시 그립니다.

실행(저장소 루트에서):  python tools/make_figures.py
입력: 각 폴더의 outputs/tables, data/ 아래 CSV  →  출력: docs/img/fig*.png
화면 캡처(dash_*, bot_*)는 Streamlit 앱을 실제로 띄워 찍은 것이라 이 스크립트가 만들지 않습니다.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "01_review_analysis"
NEWS = ROOT / "02_news_classification"
RAG = ROOT / "03_rag_faq_chatbot"
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})

SENT = ["긍정", "부정", "혼합", "중립", "판단 어려움"]
SENT_COLOR = {"긍정": "#2e9e6b", "부정": "#d1495b", "중립": "#8d99ae", "혼합": "#edae49", "판단 어려움": "#7b6cb5"}
CATS = [("장착/고정", "장착_고정"), ("접힘", "접힘"), ("파손", "파손"), ("열", "열"), ("휴대성", "휴대성"), ("건조 결과", "건조_결과")]


def split_reasons(s):
    """검토사유 문자열을 사유 이름 목록으로: 괄호 설명 제거, 쉼표로 나눔, ':' 앞부분만 사용, 리뷰당 중복 제거."""
    s = re.sub(r"\([^)]*\)", "", s)
    return sorted({p.strip().split(":")[0].strip() for p in s.split(",") if p.strip()})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def bar_labels(ax, bars, fmt="{:,.0f}", pad=3, **kw):
    """세로 막대 위에 값을 적습니다."""
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom",
                    xytext=(0, pad), textcoords="offset points", fontsize=8, **kw)


# ───────────────────────── 1. 데이터 정제
raw = pd.read_csv(REV / "data/raw/reviews.csv")
clean = pd.read_csv(REV / "data/processed/reviews_clean.csv")
tag, ent = re.compile(r"<[^>]+>"), re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
dup = int(raw.duplicated(["reviewer_id", "asin", "timestamp", "rating", "title", "text"]).sum())
n_tag = int(raw["text"].fillna("").str.contains(tag).sum())
n_ent = int(raw["text"].fillna("").str.contains(ent).sum())
n_out = int((~raw["rating"].between(1, 5)).sum())
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2.4]})
x = np.arange(1, 6)
r0, r1 = raw["rating"].value_counts().reindex(x, fill_value=0), clean["rating"].value_counts().reindex(x, fill_value=0)
b0 = ax[0].bar(x - 0.2, r0.values, 0.4, label=f"원본 {len(raw):,}건", color="#b8c2cc")
b1 = ax[0].bar(x + 0.2, r1.values, 0.4, label=f"정제 후 {len(clean):,}건", color="#3b82c4")
bar_labels(ax[0], list(b0) + list(b1))
ax[0].set(title="별점 분포 (원본 vs 정제 후)", xlabel="별점", ylabel="리뷰 수", xticks=x)
ax[0].legend(frameon=False)
labels = ["중복 리뷰 제거", "본문 HTML 태그 제거", "본문 HTML 특수문자 변환", "별점 범위(1~5) 이탈"]
vals = [dup, n_tag, n_ent, n_out]
bars = ax[1].barh(labels, vals, color=["#d1495b", "#edae49", "#edae49", "#8d99ae"])
ax[1].invert_yaxis()
for b, v in zip(bars, vals):
    ax[1].text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v}건", va="center", fontsize=9)
ax[1].set(title="정제 작업별 처리 건수", xlim=(0, max(vals) * 1.25))
save(fig, "fig01_cleaning.png")

# ───────────────────────── 2. 리뷰 단위 BERTopic (1~3점 562건)
ti = pd.read_csv(REV / "outputs/tables/topic_info.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
names = ["이상치(군집 없음)" if t == -1 else f"주제 {t}\n" + ", ".join(eval(r)[:3]) for t, r in zip(ti["Topic"], ti["Representation"])]
colors = ["#8d99ae" if t == -1 else "#3b82c4" for t in ti["Topic"]]
bars = ax.bar(names, ti["Count"], color=colors)
bar_labels(ax, bars)
ax.set(title="리뷰 단위 BERTopic — 군집별 리뷰 수 (1~3점 562건)", ylabel="리뷰 수")
save(fig, "fig02_review_level_topics.png")
(OUT / "fig02b_topics_scatter.png").write_bytes((REV / "outputs/figures/topics_scatter.png").read_bytes())

# ───────────────────────── 3. 문장 단위 토픽(21개) → 큰 주제(9개)
st = pd.read_csv(REV / "outputs/tables/sentence_topic_summary.csv").sort_values("n_sentences")
fig, ax = plt.subplots(figsize=(9, 7))
cmap = plt.get_cmap("RdYlGn")
bars = ax.barh([f"{t}  {s}" for t, s in zip(st["Topic"], st["suggested_theme"])], st["n_sentences"],
               color=[cmap((r - 1.3) / (2.6 - 1.3)) for r in st["avg_rating"]])
for b, n, r in zip(bars, st["n_sentences"], st["avg_rating"]):
    ax.text(n + 4, b.get_y() + b.get_height() / 2, f"{n}문장 · 평균 ★{r:.2f}", va="center", fontsize=8)
ax.set(title="문장 단위 토픽 20개 + 미분류 — 문장 수 (색: 평균 별점, 붉을수록 낮음)", xlabel="문장 수", xlim=(0, st["n_sentences"].max() * 1.25))
save(fig, "fig03_sentence_topics.png")

mt = pd.read_csv(REV / "outputs/tables/merged_theme_summary.csv").sort_values("n_sentences")
fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.barh([f"{t}. {n}" for t, n in zip(mt["theme"], mt["name"])], mt["n_sentences"], color="#3b82c4")
for b, n, rv, r in zip(bars, mt["n_sentences"], mt["n_reviews"], mt["avg_rating"]):
    ax.text(n + 6, b.get_y() + b.get_height() / 2, f"{n}문장 · 리뷰 {rv}건 · 평균 ★{r:.2f}", va="center", fontsize=8.5)
ax.set(title="비슷한 주제를 묶은 큰 주제 9개 (10개 미만)", xlabel="문장 수", xlim=(0, mt["n_sentences"].max() * 1.4))
save(fig, "fig04_merged_themes.png")

# ───────────────────────── 4. 하이퍼파라미터 탐색 50개 조합
tr = pd.read_csv(REV / "outputs/tables/tuning_results.csv")
nn, mc = sorted(tr["n_neighbors"].unique()), sorted(tr["min_cluster_size"].unique())
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
for i, model in enumerate(["MiniLM-L12", "mpnet-base"]):
    sub = tr[tr["model"] == model]
    for j, (col, ttl, cm) in enumerate([("n_clusters", "군집 수", "Blues"), ("outlier_pct", "미분류 비율(%)", "Oranges")]):
        a = axes[i, j]
        piv = sub.pivot(index="n_neighbors", columns="min_cluster_size", values=col).reindex(index=nn, columns=mc)
        im = a.imshow(piv.values, cmap=cm, aspect="auto")
        a.set_xticks(range(len(mc)), mc)
        a.set_yticks(range(len(nn)), nn)
        for (y, xx), v in np.ndenumerate(piv.values):
            a.text(xx, y, f"{v:.0f}" if col == "n_clusters" else f"{v:.1f}", ha="center", va="center", fontsize=8,
                   color="white" if v > np.nanmax(piv.values) * 0.65 else "black")
        a.set(title=f"{model} · {ttl}", xlabel="최소 군집 크기 (min_cluster_size)", ylabel="UMAP 이웃 수")
        a.spines[:].set_visible(False)
        if model == "MiniLM-L12":  # 추천 조합(이웃 5, 최소 40) 표시
            a.add_patch(plt.Rectangle((mc.index(40) - 0.5, nn.index(5) - 0.5), 1, 1, fill=False, ec="#d1495b", lw=2.5))
fig.suptitle("50개 조합 탐색 (빨간 테두리 = 추천 조합: MiniLM · 이웃 5 · 최소 40)", fontweight="bold")
save(fig, "fig05_tuning_grid.png")

top5 = pd.read_csv(REV / "outputs/tables/top5.csv")
stab = {(10, 40, "MiniLM-L12"): (0.71, 0.33), (15, 40, "mpnet-base"): (0.41, 0.36), (50, 30, "MiniLM-L12"): (0.79, 0.76),
        (15, 40, "MiniLM-L12"): (0.65, 0.30), (5, 40, "MiniLM-L12"): (0.85, 0.81)}  # 시드 바꿔 재실행한 ARI(평균, 최저)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
lab = [f"{r.model}\n이웃 {r.n_neighbors}·최소 {r.min_cluster_size}" for r in top5.itertuples()]
means = [stab[(r.n_neighbors, r.min_cluster_size, r.model)][0] for r in top5.itertuples()]
mins = [stab[(r.n_neighbors, r.min_cluster_size, r.model)][1] for r in top5.itertuples()]
xx = np.arange(len(lab))
ax.bar(xx - 0.2, top5["score"], 0.4, label="종합 점수 (높을수록 좋음)", color="#b8c2cc")
ax.bar(xx + 0.2, means, 0.4, label="시드 안정성 ARI 평균", color="#3b82c4")
ax.scatter(xx + 0.2, mins, color="#d1495b", zorder=3, label="ARI 최저")
ax.set_xticks(xx, lab, fontsize=8)
ax.set(title="점수 상위 5개 조합의 안정성 — 점수 1위는 시드에 따라 군집 수가 흔들림", ylim=(0, 1))
ax.legend(frameon=False, fontsize=8, loc="upper left")
save(fig, "fig06_tuning_stability.png")

# ───────────────────────── 5. 6개 항목 + 기타 분류와 감정
fin = pd.read_csv(REV / "outputs/tables/리뷰분류_최종.csv", keep_default_na=False)
rows = []
for name, c in CATS:
    vc = fin[f"감정_{c}"][fin[f"감정_{c}"] != ""].value_counts()
    rows.append([name] + [int(vc.get(s, 0)) for s in SENT])
vc = fin.loc[fin["기타"] == 1, "감정_전체"].value_counts()
rows.append(["기타"] + [int(vc.get(s, 0)) for s in SENT])
sent_df = pd.DataFrame(rows, columns=["항목"] + SENT).set_index("항목")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 2]})
left = np.zeros(len(sent_df))
for s in SENT:
    ax[0].barh(sent_df.index, sent_df[s], left=left, color=SENT_COLOR[s], label=s)
    for i, (v, l) in enumerate(zip(sent_df[s], left)):
        if v >= 40:
            ax[0].text(l + v / 2, i, str(v), ha="center", va="center", color="white", fontsize=8)
    left += sent_df[s].values
for i, t in enumerate(left):
    ax[0].text(t + 6, i, f"{int(t)}", va="center", fontsize=9)
ax[0].invert_yaxis()
ax[0].set(title="항목별 감정 분포 (한 리뷰가 여러 항목에 속할 수 있음)", xlabel="건수", xlim=(0, left.max() * 1.1))
ax[0].legend(frameon=False, ncol=5, fontsize=8, loc="lower right")
tot = fin["감정_전체"].value_counts().reindex(SENT).fillna(0)
wedges, _ = ax[1].pie(tot, colors=[SENT_COLOR[s] for s in SENT], startangle=90, counterclock=False,
                      wedgeprops=dict(width=0.42, edgecolor="white"))
ax[1].legend(wedges, [f"{s}  {int(v)}건 ({v / tot.sum():.1%})" for s, v in tot.items()], frameon=False, fontsize=9,
             loc="upper center", bbox_to_anchor=(0.5, 0.02))
ax[1].set_title(f"리뷰 전체 감정 ({len(fin):,}건)")
save(fig, "fig07_classification_sentiment.png")

rate = pd.crosstab(fin["별점"], fin["감정_전체"]).reindex(columns=SENT).fillna(0)
share = rate.div(rate.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(8.5, 3.8))
left = np.zeros(len(share))
for s in SENT:
    ax.barh([f"★{int(i)}" for i in share.index], share[s], left=left, color=SENT_COLOR[s], label=s)
    for i, (v, l) in enumerate(zip(share[s], left)):
        if v >= 7:
            ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
    left += share[s].values
ax.invert_yaxis()
ax.set(title="별점별 리뷰 전체 감정 비율 — 별점과 감정이 항상 일치하지는 않음", xlabel="%", xlim=(0, 100))
ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
save(fig, "fig08_rating_vs_sentiment.png")

rv = fin[fin["검토필요"] == "Y"]
reasons = pd.Series([r for x in rv["검토사유"] for r in split_reasons(x)]).value_counts()
fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
ax[0].barh(reasons.index, reasons.values, color="#b5523b")
for i, v in enumerate(reasons.values):
    ax[0].text(v + 0.5, i, str(v), va="center", fontsize=8.5)
ax[0].invert_yaxis()
ax[0].set(title=f"검토 필요 {len(rv)}건 — 사유 (한 리뷰에 여러 사유 가능)", xlabel="리뷰 수")
rr = rv["별점"].value_counts().sort_index()
bars = ax[1].bar([f"★{int(i)}" for i in rr.index], rr.values, color="#b5523b")
bar_labels(ax[1], bars)
ax[1].set(title="검토 필요 리뷰의 별점 분포")
save(fig, "fig09_review_needed.png")

src = fin["근거출처"].value_counts()
conf = fin["근거신뢰도"].replace({"-": "해당 없음(기타)"}).value_counts().reindex(["높음", "보통", "낮음", "없음", "해당 없음(기타)"]).dropna()
smp = pd.read_csv(REV / "outputs/tables/표본132_직접분류.csv")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].bar(src.index, src.values, color=["#3b82c4", "#edae49"])
for i, v in enumerate(src.values):
    ax[0].text(i, v + 15, f"{v:,}", ha="center", fontsize=9)
ax[0].set(title="근거 문장을 고른 방식\n(규칙 스크립트 / 표본 직접 선택)", ylim=(0, src.max() * 1.15))
ax[1].bar(conf.index, conf.values, color="#8d99ae")
for i, v in enumerate(conf.values):
    ax[1].text(i, v + 12, str(v), ha="center", fontsize=9)
ax[1].set_xticks(range(len(conf)), conf.index, fontsize=8)
ax[1].set(title="근거 신뢰도 (자동 부여)")
m = [(smp["항목일치"] == "O").sum(), (smp["극성일치"] == "O").sum()]
bars = ax[2].bar(["항목 일치", "극성 일치"], [x / len(smp) * 100 for x in m], color=["#2e9e6b", "#2e9e6b"])
for b, x in zip(bars, m):
    ax[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{x}/{len(smp)}\n({x / len(smp):.1%})", ha="center", fontsize=9)
ax[2].set(title=f"표본 {len(smp)}건 재판정 일관성\n(같은 모델의 두 번째 판단 · 정확도 아님)", ylim=(0, 115), ylabel="%")
save(fig, "fig10_classification_quality.png")

# ───────────────────────── 6. 뉴스 제목 분류
rep = pd.read_csv(NEWS / "outputs/tables/dev_classification_report.csv", index_col=0)
cls = rep.index[:7]
cnt = pd.read_csv(NEWS / "outputs/tables/dev_predicted_vs_actual_counts.csv").set_index("주제")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
xx = np.arange(7)
ax[0].bar(xx - 0.27, rep.loc[cls, "precision"], 0.27, label="정밀도", color="#b8c2cc")
ax[0].bar(xx, rep.loc[cls, "recall"], 0.27, label="재현율", color="#8fb8e0")
ax[0].bar(xx + 0.27, rep.loc[cls, "f1-score"], 0.27, label="F1", color="#3b82c4")
ax[0].set_xticks(xx, cls)
ax[0].set(title="주제별 정밀도·재현율·F1 (검증 9,107건)", ylim=(0.6, 1))
ax[0].legend(frameon=False, ncol=3, fontsize=8)
ax[1].bar(xx - 0.2, cnt.loc[cls, "실제 건수"], 0.4, label="실제", color="#b8c2cc")
ax[1].bar(xx + 0.2, cnt.loc[cls, "예측 건수"], 0.4, label="예측", color="#3b82c4")
ax[1].set_xticks(xx, cls)
ax[1].set(title="주제별 실제 건수 vs 예측 건수", ylabel="건수")
ax[1].legend(frameon=False)
save(fig, "fig11_news_per_class.png")

ts = pd.read_csv(NEWS / "outputs/tables/dev_threshold_summary.csv")
ts.columns = ["th", "n", "hold", "cover", "acc_cls", "acc_all"]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(ts["th"], ts["acc_cls"], "o-", color="#2e9e6b", label="분류된 건의 정답률")
ax.plot(ts["th"], ts["cover"], "s--", color="#3b82c4", label="분류된 비율(커버리지)")
for _, r in ts.iterrows():
    ax.annotate(f"{r.acc_cls:.1f}", (r.th, r.acc_cls), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color="#2e9e6b")
    ax.annotate(f"{r.cover:.1f}", (r.th, r.cover), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8, color="#3b82c4")
ax.set(title="확률 임계값을 높이면: 정답률은 오르고 분류되는 건수는 줄어듦", xlabel="임계값 (최고 확률이 이 값 미만이면 '보류')", ylabel="%", ylim=(30, 105))
ax.set_xticks(ts["th"])
ax.legend(frameon=False, loc="center left")
save(fig, "fig12_news_threshold.png")

# ───────────────────────── 7. RAG: 임베딩·벡터DB·유사도
emb = pd.read_csv(RAG / "embedding_compare_results.csv")
emb = emb[~emb["모델"].str.contains("512토큰") | emb["모델"].str.contains("e5")]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
labels = [m.replace(" (", "\n(") for m in emb["모델"]]
xx = np.arange(len(emb))
for k, (col, color) in enumerate([("Recall@1", "#b8c2cc"), ("Recall@4", "#8fb8e0"), ("MRR", "#3b82c4")]):
    b = ax[0].bar(xx + (k - 1) * 0.27, emb[col], 0.27, label=col, color=color)
    for r in b:
        ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}", ha="center", fontsize=7.5)
ax[0].set_xticks(xx, labels, fontsize=8)
ax[0].set(title="임베딩 모델별 검색 정확도 (질문 36개)", ylim=(0, 1.1))
ax[0].legend(frameon=False, ncol=3, fontsize=8)
ax[1].bar(xx - 0.2, emb["관련질문 1위 유사도 최소"], 0.4, label="관련 질문 1위 유사도 최소", color="#2e9e6b")
ax[1].bar(xx + 0.2, emb["없는내용 1위 유사도 최대"], 0.4, label="문서에 없는 질문 1위 유사도 최대", color="#d1495b")
ax[1].set_xticks(xx, labels, fontsize=8)
ax[1].set(title="'있는 질문'과 '없는 질문'을 가르는 간격 (클수록 구분이 쉬움)")
ax[1].legend(frameon=False, fontsize=8)
save(fig, "fig13_embedding_compare.png")

vs = pd.read_csv(RAG / "vectordb_compare_scale.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for dbn, g in vs.groupby("벡터 DB"):
    ax[0].plot(g["문서 수"], g["질문당 검색(ms)"], "o-", label=dbn)
    ax[1].plot(g["문서 수"], g["저장 용량(MB)"], "o-", label=dbn)
for a, t, yl in [(ax[0], "문서 수에 따른 검색 시간", "질문당 검색 (ms)"), (ax[1], "문서 수에 따른 저장 용량", "MB")]:
    a.set(xscale="log", yscale="log", title=t, xlabel="문서 수", ylabel=yl)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a.legend(frameon=False, fontsize=8)
save(fig, "fig14_vectordb_scale.png")

# 위키 모드 유사도 측정(2026-10-20, 질문 19개의 1위 문서 유사도)
sim = {"관련 질문": [0.871, 0.904, 0.878, 0.867, 0.872, 0.847, 0.861, 0.876, 0.882, 0.837],
       "문서에 없는 질문(같은 분야)": [0.840, 0.848, 0.785, 0.866, 0.838],
       "무관한 질문": [0.779, 0.788, 0.781, 0.754]}
pd.DataFrame([(g, v) for g, vs_ in sim.items() for v in vs_], columns=["질문 종류", "1위 유사도"]).to_csv(
    RAG / "wiki_similarity_check.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, (g, v) in enumerate(sim.items()):
    ax.scatter(v, [i] * len(v), s=60, alpha=.8, color=["#2e9e6b", "#edae49", "#d1495b"][i])
ax.axvline(0.81, color="k", ls="--")
ax.text(0.812, 2.35, "기준값 0.81", fontsize=9)
ax.set_yticks(range(3), sim.keys())
ax.set(title="위키 검색 1위 유사도 — 0.81 미만이면 LLM을 부르지 않고 '확인할 수 없음'", xlabel="코사인 유사도", ylim=(-0.5, 2.6))
save(fig, "fig15_wiki_similarity.png")

# ───────────────────────── 8. 위키 구성
pages = {}
for d in ["products", "entities", "topics", "analysis", "sources"]:
    pages[d] = len(list((RAG / "wiki" / d).glob("*.md")))
fig, ax = plt.subplots(figsize=(7, 3.2))
ko = {"products": "제품", "entities": "앱·조직", "topics": "주제", "analysis": "분석(대조·공백)", "sources": "원본 요약"}
bars = ax.bar([ko[k] for k in pages], pages.values(), color="#3b82c4")
bar_labels(ax, bars)
ax.set(title=f"LLM Wiki 페이지 구성 (+ index, log, 개요 = 총 {sum(pages.values()) + 3}개)", ylabel="페이지 수")
save(fig, "fig16_wiki_pages.png")

print("done")
