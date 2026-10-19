"""
capture_screens.py — 대시보드·챗봇 화면을 Playwright로 캡처해 docs/img 에 저장

먼저 두 앱을 실행해 둡니다(터미널 2개).
  streamlit run 01_review_analysis/dashboard/review_dashboard.py --server.port 8501
  streamlit run 03_rag_faq_chatbot/app.py --server.port 8502
그다음:  python tools/capture_screens.py
필요: pip install playwright pillow  (Chrome이 설치되어 있으면 channel="chrome" 으로 사용)
챗봇 캡처는 실제 LLM(.env 의 LLM_BACKEND)을 호출하므로 질문당 수십 초가 걸립니다.
"""
import time
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"


def crop_blank(path):
    """답변 아래의 빈 여백을 잘라 냅니다(사이드바 제외, 입력창 높이만큼은 무시)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    region = im.crop((300, 0, w, h - 200))
    box = ImageChops.difference(region, Image.new("RGB", region.size, (255, 255, 255))).getbbox()
    im.crop((0, 0, w, min(h, box[3] + 50))).save(path)


def dashboard(b):
    pg = b.new_page(viewport={"width": 1500, "height": 1450})
    pg.goto("http://localhost:8501")
    pg.wait_for_selector("text=리뷰 분류 검토 대시보드", timeout=60000)
    time.sleep(4)
    pg.screenshot(path=OUT / "dash_01_overview.png")

    def tab(name):
        pg.get_by_role("tab", name=name, exact=True).click()
        time.sleep(3)

    tab("장착/고정")
    pg.screenshot(path=OUT / "dash_02_fit.png")
    tab("접힘")
    pg.get_by_text("부정 (330)").first.click()
    time.sleep(3)
    pg.screenshot(path=OUT / "dash_03_fold_negative.png")
    tab("검토 필요")
    pg.screenshot(path=OUT / "dash_04_review_needed.png")


def chatbot(b):
    pg = b.new_page(viewport={"width": 1300, "height": 1900})
    pg.goto("http://localhost:8502")
    pg.wait_for_selector("text=한빛 스마트홈 FAQ 챗봇", timeout=90000)
    time.sleep(8)
    cases = [("에어컨에서 물이 새요", "출처 (답변에 인용된 문서)", "bot_02_answered.png"),
             ("공기청정기 필터 교체 시기와 가격 알려줘", "일부만 확인됨", "bot_03_partial.png"),
             ("고객센터 전화번호가 뭐예요?", "문서에서 확인할 수 없음", "bot_04_unknown.png")]
    for q, done, name in cases:
        pg.get_by_role("button", name="대화 지우기").click()
        time.sleep(2)
        box = pg.get_by_test_id("stChatInputTextArea")
        box.click()
        box.fill(q)
        box.press("Enter")
        pg.wait_for_selector(f"text={done}", timeout=150000)
        time.sleep(3)
        pg.screenshot(path=OUT / name)
        crop_blank(OUT / name)
    pg.get_by_role("button", name="대화 지우기").click()
    time.sleep(2)
    pg.screenshot(path=OUT / "bot_01_home.png", clip={"x": 0, "y": 0, "width": 1300, "height": 1050})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        dashboard(browser)
        chatbot(browser)
        browser.close()
    print("saved to", OUT)
