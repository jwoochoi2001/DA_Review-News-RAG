"""
ingest.py — RAG 준비(색인) 단계

이 파일이 하는 일 (한 문장 요약):
  data/ 폴더의 문서(.md, .csv)를 불러와 → 작은 조각으로 나누고(청킹) →
  무료·로컬 임베딩으로 숫자 좌표(벡터)를 만들어 → Chroma 벡터DB에 저장합니다.

한 번 실행해 두면 app.py(상담봇)가 이 chroma_db 폴더를 읽어서
질문과 비슷한 문서 조각을 빠르게 찾아낼 수 있습니다.

실행 방법:
    python ingest.py
"""

# ── 필요한 도구들을 불러옵니다 ─────────────────────────────────────────
import os  # 폴더/파일 경로를 다루기 위한 기본 도구

# LangChain: 문서 로더 (텍스트 파일과 CSV 파일을 읽어들입니다)
from langchain_community.document_loaders import CSVLoader

# LangChain: 긴 글을 일정한 크기로 잘라주는 도구(청킹)
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 무료·로컬 임베딩 (문장을 숫자 좌표로 바꿔줍니다. 인터넷 API 비용 없음)
from langchain_huggingface import HuggingFaceEmbeddings

# 무료·로컬 벡터DB (좌표들을 저장하고 나중에 빠르게 검색합니다)
from langchain_chroma import Chroma


# ── 설정값(상수): 바꿀 값들은 여기 위쪽에 모아 둡니다 ─────────────────
DATA_DIR = "data"            # 원본 문서가 들어 있는 폴더
CHROMA_DIR = "chroma_db"     # 벡터DB를 저장할 폴더

# 임베딩 모델명 (한국어 포함 다국어를 지원하는 무료·로컬 모델)
# 질문 36개로 비교한 결과(compare_embeddings.py, 정답 문서가 상위 4위 안에 든 비율 / 1위 비율):
#   MiniLM-L12 69% / 42%  →  mpnet-base 83% / 67%  →  multilingual-e5-large 97% / 86%
# 그래서 가장 잘 찾는 multilingual-e5-large 를 씁니다. rag_chain.py 도 같은 설정을 그대로 씁니다.
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
# e5 모델은 문서에는 "passage: ", 질문에는 "query: " 를 앞에 붙여야 성능이 나옵니다.
DOC_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def get_embeddings():
    """색인(ingest.py)과 검색(rag_chain.py)이 같은 임베딩 설정을 쓰도록 한 곳에서 만듭니다."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # 좌표 길이를 1로 맞춰 검색 정확도를 높입니다(정규화).
        encode_kwargs={"normalize_embeddings": True, "prompt": DOC_PREFIX},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_PREFIX},
    )

# 청킹 설정: 500자씩 자르되, 앞뒤 100자는 겹치게 하여 문맥이 끊기지 않게 함
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_chunks():
    """[1/4]~[2/4] 문서를 불러와 조각(청크)으로 나눕니다. (임베딩 모델 비교 평가에서도 같은 조각을 씁니다)"""
    # ── [1/4] 문서 불러오기 ───────────────────────────────────────────
    print("[1/4] data 폴더에서 문서를 불러오는 중...")

    documents = []  # 불러온 문서들을 담아 둘 리스트

    # data 폴더 안의 파일들을 하나씩 살펴봅니다.
    for file_name in sorted(os.listdir(DATA_DIR)):
        file_path = os.path.join(DATA_DIR, file_name)

        if file_name.endswith(".md"):
            # .md(마크다운) 파일은 제목(#, ##, ###) 단위로 나눠 읽습니다. (한글 깨짐 방지: utf-8)
            # 각 조각이 어느 장·절에서 왔는지 metadata 에 남겨 두어, 답변할 때 정확한 출처를 보여 줄 수 있습니다.
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "제목"), ("##", "장"), ("###", "절")],
                strip_headers=False,  # 제목 줄도 본문에 남겨 검색·답변 근거에 함께 쓰입니다
            )
            loaded = []
            for part in header_splitter.split_text(text):
                body = "\n".join(l for l in part.page_content.splitlines() if not l.lstrip().startswith("#")).strip(" -\n")
                if len(body) < 20:  # 제목만 있고 본문이 없는 조각은 검색 잡음이 되므로 제외
                    continue
                part.metadata["source"] = file_path
                part.metadata["section"] = " › ".join(v for v in (part.metadata.get("장"), part.metadata.get("절")) if v)
                loaded.append(part)
            documents.extend(loaded)
            print(f"      - (md) {file_name} : {len(loaded)}개 문서 로드")

        elif file_name.endswith(".csv"):
            # .csv 파일은 CSVLoader 로 읽습니다. (한 줄이 한 문서가 됩니다)
            loader = CSVLoader(file_path, encoding="utf-8")
            loaded = loader.load()
            documents.extend(loaded)
            print(f"      - (csv) {file_name} : {len(loaded)}개 문서 로드")

        else:
            # .md, .csv 가 아닌 파일은 건너뜁니다.
            print(f"      - (건너뜀) {file_name}")

    print(f"      => 총 {len(documents)}개의 문서를 불러왔습니다.")

    # ── [2/4] 문서를 작은 조각으로 나누기(청킹) ─────────────────────────
    print("[2/4] 문서를 작은 조각(청크)으로 나누는 중...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # 한 조각의 최대 글자 수
        chunk_overlap=CHUNK_OVERLAP,  # 조각끼리 겹치는 글자 수
    )
    chunks = splitter.split_documents(documents)

    print(f"      => 총 {len(chunks)}개의 조각으로 나눴습니다.")
    return chunks


def main():
    chunks = load_chunks()

    # ── [3/4] 임베딩 준비 (문장 → 숫자 좌표) ────────────────────────────
    print("[3/4] 무료·로컬 임베딩 모델을 준비하는 중...")
    print("      (처음 실행 시 모델을 내려받아 시간이 걸릴 수 있습니다)")

    embeddings = get_embeddings()

    print("      => 임베딩 모델 준비 완료.")

    # ── [4/4] Chroma 벡터DB에 저장 ─────────────────────────────────────
    print("[4/4] 조각들을 임베딩하여 Chroma 벡터DB에 저장하는 중...")

    # from_documents: 조각들을 임베딩한 뒤 chroma_db 폴더에 저장합니다.
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,  # 이 폴더에 벡터DB가 저장됩니다
    )

    print(f"      => '{CHROMA_DIR}' 폴더에 저장을 마쳤습니다.")
    print("\n✅ 색인 완료! 이제 'streamlit run app.py' 로 상담봇을 실행할 수 있습니다.")


# 이 파일을 직접 실행했을 때만 main() 이 동작하도록 합니다.
if __name__ == "__main__":
    main()
