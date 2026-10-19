"""
ingest_wiki.py — LLM Wiki(wiki/)를 챗봇이 검색할 수 있게 색인

wiki/ 의 내용 페이지(products, entities, topics, 개요)를 제목 단위로 나눠
같은 chroma_db 폴더의 별도 컬렉션("wiki")에 저장합니다. 원문 색인(ingest.py)은 건드리지 않습니다.
  - 제외: index.md·log.md(목록·기록), sources/(원본 지도), analysis/(점검 메모) → 답변 근거가 아니라 관리용 페이지
  - 각 조각 앞에 페이지 제목을 붙여, 조각만 읽어도 어느 제품·주제 얘기인지 알 수 있게 합니다.
  - 위키 문장 끝의 "(출처: 제품_매뉴얼.md §3.2)" 는 조각 안에 그대로 남아 원문 위치가 답변에서도 보입니다.

실행:  python ingest_wiki.py        (위키를 고친 뒤에는 다시 실행)
"""
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ingest import get_embeddings

ROOT = Path(__file__).resolve().parent
WIKI_DIR = ROOT / "wiki"
CHROMA_DIR = ROOT / "chroma_db"
WIKI_COLLECTION = "wiki"
INCLUDE_DIRS = ("products", "entities", "topics")  # + 개요.md
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100  # 표·목록이 중간에 잘리지 않도록 원문(500)보다 조금 크게


def wiki_pages():
    yield WIKI_DIR / "개요.md"
    for d in INCLUDE_DIRS:
        yield from sorted((WIKI_DIR / d).glob("*.md"))


def load_wiki_chunks():
    splitter = MarkdownHeaderTextSplitter([("#", "제목"), ("##", "장"), ("###", "절")], strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []
    for path in wiki_pages():
        text = path.read_text(encoding="utf-8")
        fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        body = text[fm.end():] if fm else text
        title = re.search(r"^title:\s*(.*)$", fm.group(1), re.M).group(1).strip() if fm else path.stem
        ptype = re.search(r"^type:\s*(.*)$", fm.group(1), re.M).group(1).strip() if fm else ""
        for part in splitter.split_text(body):
            plain = "\n".join(l for l in part.page_content.splitlines() if not l.lstrip().startswith("#")).strip(" -\n")
            if len(plain) < 20:  # 제목만 있는 조각 제외
                continue
            section = " › ".join(v for v in (part.metadata.get("장"), part.metadata.get("절")) if v)
            part.page_content = f"[위키 페이지: {title}]\n{part.page_content}"
            part.metadata.update(source=str(path.relative_to(ROOT)).replace("\\", "/"), page=path.stem,
                                 title=title, type=ptype, section=section)
            docs.append(part)
    return size_splitter.split_documents(docs)


def main():
    chunks = load_wiki_chunks()
    print(f"위키 페이지 → {len(chunks)}개 조각")
    # 다시 실행해도 중복되지 않도록 위키 컬렉션만 비우고 새로 만듭니다(원문 컬렉션은 그대로).
    if CHROMA_DIR.exists():
        Chroma(collection_name=WIKI_COLLECTION, persist_directory=str(CHROMA_DIR),
               embedding_function=get_embeddings()).delete_collection()
    Chroma.from_documents(chunks, get_embeddings(), collection_name=WIKI_COLLECTION,
                          persist_directory=str(CHROMA_DIR))
    print(f"완료: '{CHROMA_DIR.name}'의 '{WIKI_COLLECTION}' 컬렉션에 저장했습니다.")


if __name__ == "__main__":
    main()
