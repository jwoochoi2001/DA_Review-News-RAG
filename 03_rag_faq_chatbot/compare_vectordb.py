"""
compare_vectordb.py — 벡터 DB 비교 실험: Chroma vs FAISS

A. 실제 데이터(이 프로젝트의 30조각, e5-large 임베딩) : 같은 질문 36개로 검색 결과가 같은지, 속도는 어떤지
B. 기능 점검 : 메타데이터 필터, 문서 삭제·추가, 저장·불러오기를 LangChain 연결로 직접 시험
C. 규모 실험(가상 데이터) : 문서 1천 / 1만 / 5만 개일 때 색인 시간, 검색 속도, 정확도(정확 검색 대비)

실행: python compare_vectordb.py     결과: vectordb_compare_real.csv, vectordb_compare_scale.csv
"""
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import compare_embeddings as CE
import ingest

TMP = Path(tempfile.mkdtemp(prefix="vdb_"))


def dir_mb(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


# ───────────────────────────────────────────── A. 실제 데이터
def part_a():
    import chromadb
    import faiss
    from sentence_transformers import SentenceTransformer

    chunks = ingest.load_chunks()
    texts = [c.page_content for c in chunks]
    model = SentenceTransformer(ingest.EMBEDDING_MODEL)
    model.max_seq_length = 512
    C = CE.encode(model, texts, ingest.DOC_PREFIX).astype("float32")
    Q = CE.encode(model, [q for q, _ in CE.IN_SCOPE], ingest.QUERY_PREFIX).astype("float32")
    gold = [{i for i, t in enumerate(texts) if key in t} for _, key in CE.IN_SCOPE]
    np.save(TMP / "C.npy", C)
    np.save(TMP / "Q.npy", Q)

    def score(search):
        ranks, tops = [], []
        t0 = time.perf_counter()
        for q, g in zip(Q, gold):
            ids = search(q, 4)
            tops.append(ids)
            ranks.append(next((r for r, i in enumerate(ids, 1) if i in g), 99))
        ms = (time.perf_counter() - t0) / len(Q) * 1000
        ranks = np.array(ranks)
        return ranks, tops, ms

    # 기준: 전수 비교(numpy)
    exact = lambda q, k: list(np.argsort(-(C @ q))[:k])
    r0, t0_, ms0 = score(exact)

    client = chromadb.PersistentClient(path=str(TMP / "chroma_real"))
    col = client.create_collection("real", metadata={"hnsw:space": "cosine"})
    col.add(ids=[str(i) for i in range(len(C))], embeddings=C.tolist(), documents=texts)
    chroma = lambda q, k: [int(i) for i in col.query(query_embeddings=[q.tolist()], n_results=k)["ids"][0]]
    r1, t1, ms1 = score(chroma)

    idx = faiss.IndexFlatIP(C.shape[1])
    idx.add(C)
    fl = lambda q, k: [int(i) for i in idx.search(q[None], k)[1][0]]
    r2, t2, ms2 = score(fl)

    rows = []
    for name, r, tops, ms in [("전수 비교(numpy, 기준)", r0, t0_, ms0), ("Chroma (HNSW)", r1, t1, ms1), ("FAISS (Flat, 정확 검색)", r2, t2, ms2)]:
        same = np.mean([set(a) == set(b) for a, b in zip(tops, t0_)])
        rows.append({"벡터 DB": name, "Recall@1": round(float(np.mean(r <= 1)), 3), "Recall@4": round(float(np.mean(r <= 4)), 3),
                     "MRR": round(float(np.mean(1 / r)), 3), "상위4 결과가 기준과 같은 질문 비율": round(float(same), 3),
                     "질문당 검색(ms)": round(ms, 3)})
    df = pd.DataFrame(rows)
    df.to_csv("vectordb_compare_real.csv", index=False, encoding="utf-8-sig")
    print("\n[A] 실제 데이터 (조각 30개, 질문 36개)\n" + df.to_string(index=False), flush=True)
    return chunks, C, Q, texts


# ───────────────────────────────────────────── B. 기능 점검 (LangChain 연결)
def part_b(chunks, C, Q, texts):
    from langchain_chroma import Chroma
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings

    vec = {t: C[i].tolist() for i, t in enumerate(texts)}
    qvec = {}

    class Pre(Embeddings):  # 이미 계산한 임베딩을 돌려줘서 모델을 다시 부르지 않음
        def embed_documents(self, ts):
            return [vec[t] for t in ts]

        def embed_query(self, t):
            return qvec[t]

    emb = Pre()
    question = "에어컨에서 물이 새요"
    qvec[question] = Q[[q for q, _ in CE.IN_SCOPE].index(question)].tolist()
    res = {}

    def trial(name, fn):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            res[name] = f"실패: {type(e).__name__}: {str(e)[:60]}"

    chroma = Chroma.from_documents(chunks, emb, persist_directory=str(TMP / "chroma_b"), collection_name="vdb_b",
                                   collection_metadata={"hnsw:space": "cosine"})
    faiss_vs = FAISS.from_documents(chunks, emb)

    def flt_chroma():
        d = chroma.similarity_search(question, k=4, filter={"source": str(chunks[0].metadata["source"])})
        return f"OK ({len(d)}건, 모두 같은 출처: {len({x.metadata['source'] for x in d}) == 1})"

    def flt_faiss():
        d = faiss_vs.similarity_search(question, k=4, filter={"source": str(chunks[0].metadata["source"])}, fetch_k=30)
        return f"OK ({len(d)}건, 모두 같은 출처: {len({x.metadata['source'] for x in d}) == 1})"

    def del_chroma():
        ids = chroma.get()["ids"][:2]
        chroma.delete(ids=ids)
        return f"OK (남은 {chroma._collection.count()}건)"

    def del_faiss():
        ids = list(faiss_vs.index_to_docstore_id.values())[:2]
        faiss_vs.delete(ids)
        return f"OK (남은 {faiss_vs.index.ntotal}건)"

    def save_chroma():
        t = time.perf_counter()
        c2 = Chroma(persist_directory=str(TMP / "chroma_b"), embedding_function=emb, collection_name="vdb_b")
        return f"자동 저장, 다시 열기 {time.perf_counter() - t:.2f}초, {c2._collection.count()}건, 폴더 {dir_mb(TMP / 'chroma_b'):.2f}MB"

    def save_faiss():
        faiss_vs.save_local(str(TMP / "faiss_b"))
        t = time.perf_counter()
        f2 = FAISS.load_local(str(TMP / "faiss_b"), emb, allow_dangerous_deserialization=True)
        return f"수동 저장(save_local), 불러오기 {time.perf_counter() - t:.2f}초, {f2.index.ntotal}건, 폴더 {dir_mb(TMP / 'faiss_b'):.2f}MB"

    trial(("메타데이터 필터", "Chroma"), flt_chroma)
    trial(("메타데이터 필터", "FAISS"), flt_faiss)
    trial(("문서 삭제", "Chroma"), del_chroma)
    trial(("문서 삭제", "FAISS"), del_faiss)
    trial(("저장·불러오기", "Chroma"), save_chroma)
    trial(("저장·불러오기", "FAISS"), save_faiss)
    import site
    sp = Path(site.getsitepackages()[-1] if "site-packages" in site.getsitepackages()[-1] else site.getsitepackages()[0])
    sizes = {k: dir_mb(sp / v) for k, v in {"Chroma": "chromadb", "FAISS": "faiss"}.items() if (sp / v).exists()}
    rows = [{"항목": k[0], "Chroma": res[(k[0], "Chroma")], "FAISS": res[(k[0], "FAISS")]} for k in res if k[1] == "Chroma"]
    rows.append({"항목": "라이브러리 자체 크기", "Chroma": f"{sizes.get('Chroma', 0):.0f}MB (+의존 패키지 다수)", "FAISS": f"{sizes.get('FAISS', 0):.0f}MB"})
    df = pd.DataFrame(rows)
    print("\n[B] 기능 점검 (LangChain 연결)\n" + df.to_string(index=False), flush=True)


# ───────────────────────────────────────────── C. 규모 실험 (가상 데이터)
def synth(n, d, rng, centers):
    lab = rng.integers(0, len(centers), n)
    x = centers[lab] + 0.35 * rng.standard_normal((n, d)).astype("float32")
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype("float32")


def part_c(sizes=(1000, 10000, 50000), d=1024, nq=100, k=10):
    import chromadb
    import faiss

    rng = np.random.default_rng(0)
    centers = rng.standard_normal((200, d)).astype("float32")
    rows = []
    for n in sizes:
        X, Q = synth(n, d, rng, centers), synth(nq, d, rng, centers)
        flat = faiss.IndexFlatIP(d)
        flat.add(X)
        truth = flat.search(Q, k)[1]

        def recall(ids):
            return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, truth)]))

        # FAISS Flat
        t = time.perf_counter(); f = faiss.IndexFlatIP(d); f.add(X); build = time.perf_counter() - t
        t = time.perf_counter(); ids = f.search(Q, k)[1]; ms = (time.perf_counter() - t) / nq * 1000
        faiss.write_index(f, str(TMP / f"flat_{n}.idx"))
        rows.append({"문서 수": n, "벡터 DB": "FAISS Flat(정확)", "색인(초)": round(build, 2), "질문당 검색(ms)": round(ms, 2),
                     "정확도(Recall@10)": round(recall(ids), 3), "저장 용량(MB)": round(dir_mb(TMP / f"flat_{n}.idx"), 1)})
        # FAISS HNSW
        t = time.perf_counter(); h = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 200; h.add(X); build = time.perf_counter() - t
        h.hnsw.efSearch = 64
        t = time.perf_counter(); ids = h.search(Q, k)[1]; ms = (time.perf_counter() - t) / nq * 1000
        faiss.write_index(h, str(TMP / f"hnsw_{n}.idx"))
        rows.append({"문서 수": n, "벡터 DB": "FAISS HNSW(근사)", "색인(초)": round(build, 2), "질문당 검색(ms)": round(ms, 2),
                     "정확도(Recall@10)": round(recall(ids), 3), "저장 용량(MB)": round(dir_mb(TMP / f"hnsw_{n}.idx"), 1)})
        # Chroma (기본 HNSW)
        path = TMP / f"chroma_{n}"
        client = chromadb.PersistentClient(path=str(path))
        col = client.create_collection(f"c{n}", metadata={"hnsw:space": "cosine"})
        bs = client.get_max_batch_size()
        t = time.perf_counter()
        for s in range(0, n, bs):
            col.add(ids=[str(i) for i in range(s, min(s + bs, n))], embeddings=X[s:s + bs].tolist())
        build = time.perf_counter() - t
        t = time.perf_counter()
        ids = [[int(i) for i in col.query(query_embeddings=[q.tolist()], n_results=k)["ids"][0]] for q in Q]
        ms = (time.perf_counter() - t) / nq * 1000
        rows.append({"문서 수": n, "벡터 DB": "Chroma(기본 HNSW)", "색인(초)": round(build, 2), "질문당 검색(ms)": round(ms, 2),
                     "정확도(Recall@10)": round(recall(ids), 3), "저장 용량(MB)": round(dir_mb(path), 1)})
        print(pd.DataFrame(rows[-3:]).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("vectordb_compare_scale.csv", index=False, encoding="utf-8-sig")
    print("\n[C] 규모 실험 (가상 데이터, 1024차원, 질문 100개, Recall@10 = 정확 검색 결과와 겹치는 비율)\n" + df.to_string(index=False), flush=True)


if __name__ == "__main__":
    try:
        chunks, C, Q, texts = part_a()
        part_b(chunks, C, Q, texts)
        part_c()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

