"""
wiki_lint.py — LLM Wiki 점검 (lint)

위키(wiki/)가 원본(data/)과 어긋나거나 깨지지 않았는지 기계적으로 확인합니다.
  오류(고쳐야 함)  : frontmatter 누락, 깨진 [[링크]], 존재하지 않는 출처(§절 번호·FAQ #번호), 원문에 없는 숫자
  경고(확인 권장)  : 고아 페이지, 색인(index.md)에 없는 페이지, 위키가 인용하지 않은 원문 절·FAQ 행, 로그 형식
실행:  python wiki_lint.py          (오류가 있으면 종료 코드 1)

의미의 모순(예: 두 페이지가 서로 다른 숫자를 말함)은 LLM이 읽고 판단해야 하므로 이 스크립트가 잡지 못합니다.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIKI, DATA = ROOT / "wiki", ROOT / "data"
MANUAL, FAQ = "제품_매뉴얼.md", "제품_FAQ.csv"
TYPES = {"product", "entity", "topic", "source", "analysis", "overview", "index"}
EXEMPT = {"index", "log"}  # frontmatter·출처 검사에서 제외

errors, warnings = [], []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


# ── 원본에서 유효한 출처 목록 만들기 ─────────────────────────────
manual_text = (DATA / MANUAL).read_text(encoding="utf-8")
sections, body_sections, cur = set(), set(), None
for line in manual_text.splitlines():
    m = re.match(r"^#{2,3}\s+(\d+(?:\.\d+)?)\.?\s", line)
    if m:
        cur = m.group(1)
        sections.add(cur)
        continue
    if cur and line.strip() and not line.startswith("#") and len(line.strip(" -")) >= 5:
        body_sections.add(cur)  # 본문이 있는 절만 '인용해야 할 절'로 본다
with open(DATA / FAQ, encoding="utf-8") as f:
    faq_rows = list(csv.DictReader(f))
faq_ids = set(range(1, len(faq_rows) + 1))
raw_text = re.sub(r"\s+", "", manual_text + "".join(r["question"] + r["answer"] for r in faq_rows))

# ── 페이지 읽기 ──────────────────────────────────────────────
pages = {}
for p in sorted(WIKI.rglob("*.md")):
    if p.stem in pages:
        err(f"파일 이름이 겹칩니다: {p.stem}")
    pages[p.stem] = p
text = {n: p.read_text(encoding="utf-8") for n, p in pages.items()}


def frontmatter(t):
    m = re.match(r"^---\n(.*?)\n---\n", t, re.S)
    if not m:
        return None, t
    fm = {}
    for line in m.group(1).splitlines():
        k = re.match(r"^(\w+):\s*(.*)$", line)
        if k:
            fm[k.group(1)] = k.group(2).strip()
    fm["_sources"] = re.findall(r'^\s+-\s+"(.+)"', m.group(1), re.M)
    return fm, t[m.end():]


links, cited_manual, cited_faq = {}, set(), set()
for n, t in text.items():
    fm, body = frontmatter(t)
    if n not in EXEMPT:
        if fm is None:
            err(f"[{n}] frontmatter가 없습니다")
        else:
            for k in ("title", "type", "updated"):
                if not fm.get(k):
                    err(f"[{n}] frontmatter에 {k}가 없습니다")
            if fm.get("type") and fm["type"] not in TYPES:
                err(f"[{n}] 알 수 없는 type: {fm['type']}")
            if fm.get("type") not in ("index", "overview") and not fm.get("_sources"):
                err(f"[{n}] sources(출처)가 비어 있습니다")
    # 링크
    links[n] = set(re.findall(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]", t))
    for target in links[n]:
        if target not in pages:
            err(f"[{n}] 깨진 링크: [[{target}]]")
    # 출처 인용 (frontmatter 포함 전체 텍스트)
    for s in re.findall(rf"{re.escape(MANUAL)} §(\d+(?:\.\d+)?)", t):
        if s not in sections:
            err(f"[{n}] 존재하지 않는 매뉴얼 절 인용: §{s}")
        elif not n.startswith("제품_") and not (WIKI / "sources" / f"{n}.md").exists():
            cited_manual.add(s)
    for s in re.findall(rf"{re.escape(FAQ)} #(\d+)", t):
        if int(s) not in faq_ids:
            err(f"[{n}] 존재하지 않는 FAQ 번호 인용: #{s}")
        elif not (WIKI / "sources" / f"{n}.md").exists():
            cited_faq.add(int(s))
    # 숫자 근거 확인: 본문의 '숫자+단위'가 원문에 있어야 한다 (날짜·출처 표기는 제외)
    if n not in EXEMPT:
        clean = re.sub(r"---\n.*?\n---\n", "", t, flags=re.S)
        clean = re.sub(r"§\d+(?:\.\d+)?|#\d+|\d{4}-\d{2}-\d{2}|`[^`]*`", "", clean)
        for tok in set(re.findall(r"\d+(?:\.\d+)?(?:~\d+)?\s?(?:암페어|GHz|초|미터|도|개월|퍼센트|주|일|년|시)", clean)):
            if re.sub(r"\s+", "", tok) not in raw_text:
                err(f"[{n}] 원문에 없는 숫자 표현: '{tok}'")

# ── 구조 점검 (경고) ──────────────────────────────────────────
inbound = {n: set() for n in pages}
for n, ts in links.items():
    for t in ts:
        if t in inbound and t != n:
            inbound[t].add(n)
for n in pages:
    if n in EXEMPT:
        continue
    if not (inbound[n] - EXEMPT):
        warn(f"[{n}] 고아 페이지: 색인 말고는 어디에서도 연결되지 않습니다")
    if "index" in pages and n not in links.get("index", set()):
        warn(f"[{n}] 색인(index.md)에 없습니다")

for s in sorted(body_sections - cited_manual, key=lambda x: [int(i) for i in x.split(".")]):
    warn(f"매뉴얼 §{s}을 인용한 내용 페이지가 없습니다 (위키가 이 절을 다루지 않음)")
for i in sorted(faq_ids - cited_faq):
    warn(f"FAQ #{i}을 인용한 내용 페이지가 없습니다")

if "log" in pages:
    entries = re.findall(r"^## (.*)$", text["log"], re.M)
    for e in entries:
        if not re.match(r"^\[\d{4}-\d{2}-\d{2}\] (ingest|query|lint|index|update) \| .+", e):
            warn(f"[log] 형식에 맞지 않는 항목: '## {e}'")
    dates = [re.match(r"^\[(\d{4}-\d{2}-\d{2})\]", e).group(1) for e in entries if re.match(r"^\[\d{4}-\d{2}-\d{2}\]", e)]
    if dates != sorted(dates):
        warn("[log] 날짜 순서가 시간 순이 아닙니다")
else:
    warn("log.md가 없습니다")

print(f"페이지 {len(pages)}개 · 매뉴얼 절 {len(sections)}개(본문 있는 절 {len(body_sections)}개) · FAQ {len(faq_ids)}개")
print(f"인용된 매뉴얼 절 {len(cited_manual)}/{len(body_sections)}, 인용된 FAQ {len(cited_faq)}/{len(faq_ids)} (내용 페이지 기준)")
for m in errors:
    print("오류:", m)
for m in warnings:
    print("경고:", m)
print(f"\n결과: 오류 {len(errors)}건, 경고 {len(warnings)}건")
sys.exit(1 if errors else 0)
