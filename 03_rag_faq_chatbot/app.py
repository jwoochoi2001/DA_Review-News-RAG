"""
app.py — 한빛 스마트홈 FAQ 챗봇 (Streamlit)

실행:  streamlit run app.py        (먼저 python ingest.py 와 python ingest_wiki.py 로 색인)
질문 → rag_chain.ask() → 답변 + 답변에 인용된 실제 출처 + 검색된 참고 문서를 함께 보여 줍니다.
문서에서 확인되지 않는 내용은 "확인할 수 없습니다"로 답합니다.
"""
import os

import streamlit as st

import rag_chain as R

st.set_page_config(page_title="한빛 스마트홈 FAQ 챗봇", layout="centered")
st.title("한빛 스마트홈 FAQ 챗봇")
st.caption("정리된 위키(또는 원문 매뉴얼·FAQ)를 검색해 출처와 함께 답합니다. 문서에서 확인되지 않는 내용은 답하지 않습니다.")

EXAMPLES = ["에어컨에서 물이 새요", "공기청정기 필터는 언제 교체하나요?", "Wi-Fi 연결이 안 돼요",
            "스마트 플러그에 전기히터를 써도 되나요?", "환불은 어떻게 하나요?", "A/S는 어떻게 신청하나요?",
            "한빛 스마트 TV 화면 크기가 몇 인치예요?"]
STATUS_STYLE = {"answered": ("success", "근거 확인됨"), "partial": ("warning", "일부만 확인됨"),
                "unanswerable": ("info", "문서에서 확인할 수 없음"), "no_retrieval": ("info", "관련 문서 없음"),
                "ungrounded": ("warning", "근거 번호가 없어 답변을 차단함")}


@st.cache_resource(show_spinner="검색 모델을 준비하는 중...")
def load_chain(mode):
    return R.build_chain(mode=mode)


with st.sidebar:
    st.subheader("설정")
    backend = os.getenv("LLM_BACKEND", "claude_code")
    label = {"claude_code": "Claude Code 구독 (claude -p)", "api": "Anthropic API 키",
             "mock": "테스트용 발췌 (LLM 없음)"}.get(backend, backend)
    st.write(f"**LLM 연결**: {label}")
    SOURCES = {"wiki": "LLM Wiki (정리된 위키)", "raw": "원문 (매뉴얼·FAQ 직접 검색)"}
    mode = st.radio("지식 소스", list(SOURCES), format_func=SOURCES.get,
                    index=list(SOURCES).index(R.KNOWLEDGE_SOURCE) if R.KNOWLEDGE_SOURCE in SOURCES else 0,
                    help="위키: wiki/ 페이지를 검색하며 각 문장에 원문 위치(§·#)가 붙어 있습니다. 원문: data/ 문서를 그대로 검색합니다.")
    st.write(f"**검색**: 로컬 임베딩 + Chroma, 상위 {R.TOP_K}개")
    st.write(f"**관련 문서 기준**: 유사도 {R.min_similarity(mode)} 이상")
    if backend == "mock":
        st.warning("테스트 모드입니다. 답변은 검색된 첫 문서를 그대로 발췌한 것이며 LLM 답변이 아닙니다.")
    if not R.CHROMA_DIR.exists():
        st.error("색인이 없습니다. 터미널에서 `python ingest.py` (원문), `python ingest_wiki.py` (위키)를 먼저 실행하세요.")
    st.divider()
    st.subheader("예시 질문")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex
    if st.button("대화 지우기", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

st.session_state.setdefault("messages", [])


def render_sources(m):
    """답변 아래에 실제 출처(인용된 문서)와 검색된 참고 문서를 보여 줍니다."""
    hits = m.get("hits") or []
    cited = {c["n"] for c in m.get("citations", [])}
    if cited:
        st.markdown("**출처** (답변에 인용된 문서)")
        for n in sorted(cited):
            h = hits[n - 1]
            info = R.source_info(h["doc"])
            with st.expander(f"[{n}] {R.source_label(h['doc'])} · 유사도 {h['similarity']:.2f}", expanded=True):
                if info["title"] and info["file"].endswith(".csv"):
                    st.caption(f"FAQ 질문: {info['title']}")
                st.code(h["doc"].page_content, language=None, wrap_lines=True)
    others = [(i, h) for i, h in enumerate(hits, 1) if i not in cited]
    if others:
        if cited:
            title = f"검색됐지만 인용되지 않은 참고 문서 {len(others)}건"
        else:
            title = f"검색된 가장 가까운 문서 {len(others)}건 (답변의 근거가 아닙니다)"
        with st.expander(title):
            for i, h in others:
                st.markdown(f"**[{i}] {R.source_label(h['doc'])}** · 유사도 {h['similarity']:.2f}"
                            + (" · 기준 미만" if h["similarity"] < R.min_similarity(m.get("mode")) else ""))
                st.code(h["doc"].page_content, language=None, wrap_lines=True)


def render_message(m):
    if m["role"] == "user":
        st.markdown(m["content"])
        return
    kind, text = STATUS_STYLE.get(m.get("status"), ("error", "오류"))
    if m.get("status") == "error":
        st.error(m["content"])
    else:
        getattr(st, kind)(text, icon=None)
        st.markdown(m["content"])
        if m.get("mode"):
            st.caption(f"지식 소스: {SOURCES[m['mode']]}")
        if m.get("uncited"):
            st.warning("다음 문장에는 근거 번호가 없습니다. 문서에서 확인되지 않은 내용일 수 있으니 출처와 대조해 주세요.\n\n"
                       + "\n".join(f"- {u}" for u in m["uncited"]))
    render_sources(m)


for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        render_message(m)

question = st.chat_input("제품에 대해 궁금한 점을 물어보세요") or st.session_state.pop("pending", None)
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("근거를 찾고 답변을 만드는 중..."):
                result = R.ask(question, chain=load_chain(mode))
            msg = {"role": "assistant", "content": result["answer"], "status": result["status"],
                   "hits": result["hits"], "citations": result["citations"], "uncited": result["uncited"], "mode": mode}
        except Exception as e:  # LLM 로그인·API 키·색인 문제를 화면에 안내
            hits = []
            try:  # 답변은 실패해도 검색된 문서는 보여 줍니다
                hits = R.retrieve(question, mode=mode)
            except Exception:
                pass
            msg = {"role": "assistant", "content": str(e), "status": "error", "hits": hits, "citations": []}
        render_message(msg)
    st.session_state["messages"].append(msg)
