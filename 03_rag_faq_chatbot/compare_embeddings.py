"""
compare_embeddings.py — 무료·로컬 임베딩 모델 검색 성능 비교

같은 조각(ingest.load_chunks)과 같은 평가 질문으로 모델별 검색 성능을 측정합니다.
  - 관련 질문 {n_in}개: 정답 문서가 상위 몇 위에 오는가 (Recall@1/3/4, MRR)
  - 무관·없는 내용 질문 {n_oos}개: "문서 없음" 판단을 위한 유사도 기준이 얼마나 잘 갈리는가
실행:  python compare_embeddings.py        결과: embedding_compare_results.csv
"""
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import ingest

# (표시 이름, 모델, 질문 접두어, 문서 접두어, max_seq_length)
MODELS = [
    ("MiniLM-L12 (128토큰)", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", "", 128),
    ("mpnet-base (128토큰, 현재)", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", "", 128),
    ("mpnet-base (512토큰)", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", "", 512),
    ("e5-large (512토큰)", "intfloat/multilingual-e5-large", "query: ", "passage: ", 512),
]

# 관련 질문: (질문, 정답 조각에 반드시 들어 있는 문구). 문서 문장을 그대로 베끼지 않고 구어체·다른 표현으로 작성
IN_SCOPE = [
    ("에어컨에서 물이 새요", "배수 호스"), ("에어컨 실내기에서 물이 뚝뚝 떨어져요", "배수 호스"),
    ("에어컨 표면에 이슬이 맺혀요", "결로"), ("냉방이 안 시원해요", "냉방 효율"),
    ("에어컨 바람이 미지근해요", "냉방 효율"), ("실외기 주변에 짐을 둬도 되나요", "1미터"),
    ("에어컨 필터 세척 방법", "세제"), ("필터를 씻을 때 세제를 써도 돼요", "세제"),
    ("에어컨 온도를 몇 도까지 내릴 수 있어요", "18도에서 30도"), ("에어컨 자동 모드가 뭐예요", "자동 모드"),
    ("공기청정기 필터 언제 갈아요", "6개월에서 12개월"), ("공기청정기 필터가 얼마나 남았는지 어디서 봐요", "잔여 수명"),
    ("공기청정기 필터 교체하는 법", "화살표"), ("필터를 바꾼 다음 앱에서 뭘 눌러야 해요", "필터 교체 완료"),
    ("공기청정기 빨간 불이 계속 켜져요", "빨간색"), ("공기청정기 파란 불은 무슨 뜻이에요", "파란색은 좋음"),
    ("와이파이 연결이 안돼요", "2.4GHz"), ("5GHz 와이파이에서도 쓸 수 있나요", "5GHz"),
    ("페어링 모드로 들어가려면 어떻게 해요", "리셋"), ("앱에서 제품이 오프라인이라고 떠요", "오프라인"),
    ("정전 됐다가 켜졌는데 다시 연결이 안돼요", "정전"), ("스마트 플러그에 전기히터 연결해도 되나요", "16암페어"),
    ("플러그로 전기 사용량을 볼 수 있나요", "실시간 소비 전력"), ("플러그로 정해진 시간에 자동으로 켜게 하고 싶어요", "예약"),
    ("보증 기간이 얼마나 돼요", "구매일로부터 1년"), ("물에 빠뜨려서 고장났는데 무료로 고쳐주나요", "침수"),
    ("환불하고 싶어요", "전액 환불"), ("뜯어서 써봤는데 환불 돼요", "개봉 후"),
    ("A/S 접수는 어떻게 해요", "A/S 신청"), ("수리 기사님은 언제 오세요", "2~3일"),
    ("고객센터는 몇 시까지 해요", "평일 09시"), ("택배 열어보니 구성품이 빠졌어요", "구성품이 누락"),
    ("박스 안에 뭐가 들어 있나요", "빠른 설치 안내서"), ("한빛홈 앱은 어디서 받아요", "앱스토어"),
    ("HB-AC300에서 쓸 수 있는 운전 모드", "냉방, 제습, 송풍"), ("HB-PL10 최대 사용 용량", "16암페어"),
]
# 문서에 없는 내용: 완전히 다른 주제 + 같은 도메인이지만 문서에 없는 질문
OUT_OF_SCOPE = [
    "오늘 점심 뭐 먹지", "파이썬으로 웹크롤링 하는 법", "서울에서 부산까지 KTX 요금", "비트코인 지금 시세 알려줘",
    "한빛 스마트 TV 화면 크기가 몇 인치야", "한빛 로봇청소기 배터리 용량", "공기청정기 소음이 몇 dB이에요",
    "에어컨 설치 기사 출장비는 얼마예요", "한빛 스마트 냉장고 사용법", "스마트 플러그 가격이 얼마예요",
]


def encode(model, texts, prefix):
    return model.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=16, show_progress_bar=False)


def auroc(pos, neg):
    """관련 질문의 1위 유사도가 무관 질문보다 높은 비율(1.0이면 완벽히 갈림)."""
    return float(np.mean([[p > n for n in neg] for p in pos]))


def best_threshold(pos, neg):
    cands = sorted(set(pos) | set(neg))
    best = max(((np.mean([p >= t for p in pos]) + np.mean([n < t for n in neg])) / 2, t) for t in cands)
    return best[1], best[0]


def main():
    chunks = ingest.load_chunks()
    texts = [c.page_content for c in chunks]
    gold = [{i for i, t in enumerate(texts) if key in t} for _, key in IN_SCOPE]
    assert all(gold), "정답 문구가 어떤 조각에도 없는 질문이 있습니다: " + str([q for (q, _), g in zip(IN_SCOPE, gold) if not g])
    rows, misses = [], {}
    for label, name, qp, dp, max_len in MODELS:
        t0 = time.time()
        model = SentenceTransformer(name)
        model.max_seq_length = max_len
        load_s = time.time() - t0
        t0 = time.time()
        C = encode(model, texts, dp)
        Qi = encode(model, [q for q, _ in IN_SCOPE], qp)
        Qo = encode(model, OUT_OF_SCOPE, qp)
        enc_s = time.time() - t0
        S = Qi @ C.T
        ranks = []
        for s, g in zip(S, gold):
            order = np.argsort(-s)
            ranks.append(next(r for r, idx in enumerate(order, 1) if idx in g))
        ranks = np.array(ranks)
        misses[label] = [IN_SCOPE[i][0] for i in np.where(ranks > 4)[0]]
        top_in = S.max(axis=1)
        top_out = (Qo @ C.T).max(axis=1)
        th, acc = best_threshold(top_in, top_out)
        rows.append({
            "모델": label, "Recall@1": round(float(np.mean(ranks <= 1)), 3), "Recall@3": round(float(np.mean(ranks <= 3)), 3),
            "Recall@4": round(float(np.mean(ranks <= 4)), 3), "MRR": round(float(np.mean(1 / ranks)), 3),
            "관련질문 1위 유사도 최소": round(float(top_in.min()), 2), "관련질문 1위 유사도 중앙": round(float(np.median(top_in)), 2),
            "없는내용 1위 유사도 최대": round(float(top_out.max()), 2),
            "관련/없음 구분 AUC": round(auroc(top_in, top_out), 3), "최적 기준값": round(float(th), 2), "기준값 정확도": round(float(acc), 3),
            "모델 로드(초)": round(load_s, 1), "인코딩(초)": round(enc_s, 1), "차원": int(C.shape[1]),
        })
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("embedding_compare_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n조각 {len(texts)}개 · 관련 질문 {len(IN_SCOPE)}개 · 없는 내용 질문 {len(OUT_OF_SCOPE)}개")
    print(df.drop(columns=["모델 로드(초)", "인코딩(초)", "차원"]).to_string(index=False))
    for k, v in misses.items():
        print(f"\n[{k}] 정답이 4위 밖인 질문 {len(v)}개:", v)


if __name__ == "__main__":
    main()
