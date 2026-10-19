"""리뷰 분류 검토 대시보드 (Streamlit)

실행: streamlit run review_dashboard.py
데이터: 같은 폴더의 리뷰분류_최종.csv
"""
import html
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA = Path(__file__).parent / "리뷰분류_최종.csv"
CATS = [  # (표시 이름, 데이터 칼럼 접두)
    ("장착/고정", "장착_고정"),
    ("접힘", "접힘"),
    ("파손", "파손"),
    ("열", "열"),
    ("휴대성", "휴대성"),
    ("건조 결과", "건조_결과"),
]
SENTS = ["긍정", "부정", "중립", "혼합", "판단 어려움"]
COLORS = {"긍정": "#2e9e6b", "부정": "#d1495b", "중립": "#8d99ae", "혼합": "#edae49", "판단 어려움": "#7b6cb5"}
PAGE = 10

st.set_page_config(page_title="리뷰 분류 검토 대시보드", layout="wide")
st.markdown(
    """
<style>
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;color:#fff;margin-right:6px}
.chip{display:inline-block;padding:1px 8px;border-radius:6px;font-size:12px;background:rgba(128,128,128,.18);margin-right:4px}
.body{white-space:pre-wrap;line-height:1.6;margin:6px 0 4px 0}
mark{background:#ffe08a;color:#222;border-radius:3px;padding:0 2px}
.ev{font-size:13px;opacity:.85}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load():
    df = pd.read_csv(DATA, keep_default_na=False, na_values=[None], encoding="utf-8-sig")
    df["제목"] = df["제목"].astype(str)
    df["본문"] = df["본문"].astype(str)
    return df


if not DATA.exists():
    st.error(f"데이터 파일이 없습니다: {DATA.name}")
    st.stop()
df_all = load()


def badge(s):
    return f'<span class="badge" style="background:{COLORS.get(s, "#999")}">{s}</span>' if s else ""


def highlight(body, evidences):
    """본문을 HTML로 이스케이프하고 근거 문장을 <mark>로 강조한다."""
    out = html.escape(body)
    for ev in evidences:
        ev = ev.strip().strip("…").strip()
        if len(ev) < 4:
            continue
        pat = ""
        prev_space = False
        for ch in ev:
            if ch.isspace():
                if not prev_space:
                    pat += r"\s+"
                prev_space = True
                continue
            prev_space = False
            if ch in "’'":
                pat += "(?:’|&#x27;)"
            else:
                pat += re.escape(html.escape(ch))
        out, n = re.subn(pat, lambda m: f"<mark>{m.group(0)}</mark>", out, count=1, flags=re.I)
    return out


# ------------------------------------------------------------------ 사이드바 필터
with st.sidebar:
    st.header("필터")
    stars = st.multiselect("별점", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    only_review = st.checkbox("검토 필요 리뷰만 보기")
    kw = st.text_input("키워드 검색 (제목·본문)", placeholder="예: hotel, collapse")
    st.caption("필터는 모든 탭에 적용됩니다.")
    st.divider()
    st.caption(f"데이터: {DATA.name} · 전체 {len(df_all):,}건")

df = df_all[df_all["별점"].isin(stars)]
if only_review:
    df = df[df["검토필요"] == "Y"]
if kw.strip():
    q = kw.strip().lower()
    df = df[df["제목"].str.lower().str.contains(q, regex=False) | df["본문"].str.lower().str.contains(q, regex=False)]

st.title("리뷰 분류 검토 대시보드")
st.caption("헤어 디퓨저 리뷰 · 6개 항목 + 기타 · 항목별 감정과 근거 문장 확인")

k1, k2, k3, k4 = st.columns(4)
n_items = sum(int((df[f"감정_{c}"] != "").sum()) for _, c in CATS)
k1.metric("리뷰", f"{len(df):,}")
k2.metric("항목 분류 건수", f"{n_items:,}")
k3.metric("검토 필요", f"{int((df['검토필요'] == 'Y').sum()):,}")
k4.metric("평균 별점", f"{df['별점'].mean():.2f}" if len(df) else "-")

if df.empty:
    st.info("조건에 맞는 리뷰가 없습니다. 필터를 조정해 주세요.")
    st.stop()


def pie(counts, title):
    s = counts.reindex(SENTS).fillna(0).astype(int)
    s = s[s > 0]
    fig = px.pie(
        names=s.index, values=s.values, hole=0.45, color=s.index, color_discrete_map=COLORS,
        category_orders={"names": SENTS},
    )
    fig.update_traces(textinfo="label+value+percent", sort=False, direction="clockwise")
    fig.update_layout(title=title, showlegend=False, margin=dict(t=50, b=10, l=10, r=10), height=360)
    return fig


def review_card(r, cat_col=None, sentiment=None, highlight_cols=()):
    with st.container(border=True):
        head = f"**★{int(r['별점'])}** · `{r['review_id']}` · {html.escape(r['제목']) or '(제목 없음)'}"
        st.markdown(head)
        meta = badge(sentiment)
        if r["검토필요"] == "Y":
            meta += f'<span class="badge" style="background:#b5523b">검토 필요</span><span class="ev">{html.escape(r["검토사유"])}</span>'
        st.markdown(meta, unsafe_allow_html=True)
        evs = []
        for hc in highlight_cols:
            evs += [e for e in str(r[hc]).split("‖") if e.strip()]
        body = highlight(r["본문"] if r["본문"].strip() else "(본문 없음)", evs)
        st.markdown(f'<div class="body">{body}</div>', unsafe_allow_html=True)
        foot = f'<span class="chip">전체 분류: {html.escape(r["분류"])}</span><span class="chip">전체 감정: {r["감정_전체"]}</span>'
        if cat_col and r.get(f"근거_{cat_col}"):
            foot += f'<br><span class="ev">근거 · {html.escape(str(r[f"근거_{cat_col}"]))} ({r["근거출처"]}·{r["근거신뢰도"]})</span>'
        st.markdown(foot, unsafe_allow_html=True)


def review_list(frame, key, sent_col, cat_col=None, highlight_cols=(), page_key=None):
    page_key = page_key or key  # 필터가 바뀌면 1쪽부터 다시 시작
    sort = st.selectbox("정렬", ["기본(리뷰 ID)", "별점 낮은순", "별점 높은순", "검토 필요 먼저"], key=f"sort_{key}")
    if sort == "별점 낮은순":
        frame = frame.sort_values(["별점", "review_id"])
    elif sort == "별점 높은순":
        frame = frame.sort_values(["별점", "review_id"], ascending=[False, True])
    elif sort == "검토 필요 먼저":
        frame = frame.sort_values(["검토필요", "review_id"], ascending=[False, True])
    else:
        frame = frame.sort_values("review_id")
    total = len(frame)
    pages = max(1, -(-total // PAGE))
    pk = f"page_{page_key}"
    st.session_state.setdefault(pk, 1)
    st.session_state[pk] = min(st.session_state[pk], pages)
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("◀ 이전", key=f"prev_{page_key}", disabled=st.session_state[pk] <= 1):
        st.session_state[pk] -= 1
        st.rerun()
    c2.markdown(f"<div style='text-align:center'>{total:,}건 · {st.session_state[pk]} / {pages} 쪽</div>", unsafe_allow_html=True)
    if c3.button("다음 ▶", key=f"next_{page_key}", disabled=st.session_state[pk] >= pages):
        st.session_state[pk] += 1
        st.rerun()
    st.download_button("현재 목록 CSV 내려받기", frame.drop(columns=[]).to_csv(index=False, encoding="utf-8-sig"),
                       file_name=f"{key}.csv", mime="text/csv", key=f"dl_{key}")
    start = (st.session_state[pk] - 1) * PAGE
    for _, r in frame.iloc[start:start + PAGE].iterrows():
        review_card(r, cat_col=cat_col, sentiment=r[sent_col], highlight_cols=highlight_cols)


tabs = st.tabs(["개요"] + [n for n, _ in CATS] + ["기타", "검토 필요"])

# ------------------------------------------------------------------ 개요
with tabs[0]:
    rows = []
    for name, c in CATS:
        vc = df[f"감정_{c}"][df[f"감정_{c}"] != ""].value_counts()
        for s in SENTS:
            rows.append((name, s, int(vc.get(s, 0))))
    vc = df.loc[df["기타"] == 1, "감정_전체"].value_counts()
    for s in SENTS:
        rows.append(("기타", s, int(vc.get(s, 0))))
    long = pd.DataFrame(rows, columns=["항목", "감정", "건수"])
    a, b = st.columns([3, 2])
    with a:
        fig = px.bar(long, y="항목", x="건수", color="감정", orientation="h", color_discrete_map=COLORS,
                     category_orders={"감정": SENTS, "항목": [n for n, _ in CATS] + ["기타"]})
        fig.update_layout(title="항목별 감정 분포", height=380, margin=dict(t=50, b=10), legend_title_text="")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.plotly_chart(pie(df["감정_전체"].value_counts(), "리뷰 전체 감정"), use_container_width=True)
    st.subheader("항목 × 감정 표")
    pivot = long.pivot(index="항목", columns="감정", values="건수").reindex(columns=SENTS)
    pivot = pivot.reindex([n for n, _ in CATS] + ["기타"])
    pivot["합계"] = pivot.sum(axis=1)
    st.dataframe(pivot, use_container_width=True)
    st.caption("기타 행은 항목에 해당하지 않은 리뷰의 전체 감정입니다. 한 리뷰가 여러 항목에 속할 수 있어 항목별 합계는 리뷰 수보다 큽니다.")

# ------------------------------------------------------------------ 항목별 탭
for i, (name, c) in enumerate(CATS, start=1):
    with tabs[i]:
        sub = df[df[f"감정_{c}"] != ""]
        if sub.empty:
            st.info("이 항목에 분류된 리뷰가 없습니다.")
            continue
        counts = sub[f"감정_{c}"].value_counts()
        left, right = st.columns([2, 3])
        with left:
            st.plotly_chart(pie(counts, f"{name} · 감정 {len(sub):,}건"), use_container_width=True)
            st.markdown(" ".join(f"{badge(s)} {int(counts.get(s, 0))}" for s in SENTS if counts.get(s, 0)), unsafe_allow_html=True)
            st.caption(f"평균 별점 {sub['별점'].mean():.2f} · 검토 필요 {int((sub['검토필요'] == 'Y').sum())}건")
        with right:
            opts = ["전체"] + [s for s in SENTS if counts.get(s, 0)]
            labels = {o: (f"{o} ({len(sub)})" if o == "전체" else f"{o} ({int(counts[o])})") for o in opts}
            pick = st.radio("감정 필터", opts, format_func=lambda o: labels[o], horizontal=True, key=f"pick_{c}")
            shown = sub if pick == "전체" else sub[sub[f"감정_{c}"] == pick]
            review_list(shown, key=c, sent_col=f"감정_{c}", cat_col=c, highlight_cols=(f"근거_{c}",), page_key=f"{c}_{pick}")

# ------------------------------------------------------------------ 기타
with tabs[len(CATS) + 1]:
    sub = df[df["기타"] == 1]
    if sub.empty:
        st.info("기타로 분류된 리뷰가 없습니다.")
    else:
        counts = sub["감정_전체"].value_counts()
        left, right = st.columns([2, 3])
        with left:
            st.plotly_chart(pie(counts, f"기타 · 전체 감정 {len(sub):,}건"), use_container_width=True)
            st.caption("6개 항목에 해당하지 않은 리뷰입니다. 근거 없는 칭찬·불만, 배송·가격·포장, 본문 없음 등이 포함됩니다.")
        with right:
            opts = ["전체"] + [s for s in SENTS if counts.get(s, 0)]
            labels = {o: (f"{o} ({len(sub)})" if o == "전체" else f"{o} ({int(counts[o])})") for o in opts}
            pick = st.radio("감정 필터", opts, format_func=lambda o: labels[o], horizontal=True, key="pick_other")
            shown = sub if pick == "전체" else sub[sub["감정_전체"] == pick]
            review_list(shown, key="기타", sent_col="감정_전체", page_key=f"기타_{pick}")

# ------------------------------------------------------------------ 검토 필요
with tabs[len(CATS) + 2]:
    sub = df[df["검토필요"] == "Y"]
    if sub.empty:
        st.info("검토 필요 리뷰가 없습니다.")
    else:
        reasons = sub["검토사유"].str.split(", ").explode().str.replace(r"\(.*", "", regex=True).str.split(":").str[0]
        rc = reasons.value_counts()
        left, right = st.columns([2, 3])
        with left:
            fig = px.bar(x=rc.values, y=rc.index, orientation="h", labels={"x": "건수", "y": ""})
            fig.update_layout(title=f"검토 사유 · {len(sub):,}건", height=360, margin=dict(t=50, b=10))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            pick = st.selectbox("사유", ["전체"] + list(rc.index), key="pick_review")
            shown = sub if pick == "전체" else sub[sub["검토사유"].str.contains(pick, regex=False)]
            review_list(shown, key="검토필요", sent_col="감정_전체", page_key=f"검토필요_{pick}")
