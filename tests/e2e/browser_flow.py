"""Drives the demo app in a real browser (Playwright): uploads through the actual file widget.

    streamlit run tests/e2e/demo_app.py --server.port 8766 --server.headless true &
    python tests/e2e/browser_flow.py http://localhost:8766 [screenshot_dir]
"""
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.conftest import make_pdf  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8766"
SHOTS = Path(sys.argv[2]) if len(sys.argv) > 2 else None
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(("PASS " if condition else "FAIL ") + name + (f"  [{detail}]" if detail and not condition else ""))


def main():
    work = Path(tempfile.mkdtemp())
    pdfs = []
    for n in range(1, 8):
        p = work / f"pdf{n}.pdf"
        p.write_bytes(make_pdf(f"Document {n} says the Gizmo{n} device weighs {n}00 grams and uses alpha power."))
        pdfs.append(str(p))
    (work / "notes.txt").write_text("Photosynthesis converts sunlight into chemical energy in plants.")
    (work / "guide.md").write_text("# Volcanoes\n\nA volcano erupts when magma reaches the surface.")
    (work / "eiffel_a.pdf").write_bytes(make_pdf("The Eiffel Tower is 300 metres tall."))
    (work / "eiffel_b.pdf").write_bytes(make_pdf("The Eiffel Tower is 450 metres tall."))
    (work / "scan.pdf").write_bytes(b"%PDF-1.4 broken")
    (work / "sheet.xlsx").write_bytes(b"PK fake")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=1)
        page.goto(URL)
        page.wait_for_selector("text=DocuMind AI", timeout=30000)
        sidebar = page.locator('[data-testid="stSidebar"]')
        uploader = sidebar.locator('input[type="file"]').first

        def upload(paths):
            uploader.set_input_files(paths)
            page.wait_for_timeout(2500)

        def shot(name):
            if SHOTS:
                SHOTS.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(SHOTS / f"{name}.png"))

        def remove_nth(index):
            sidebar.locator('[class*="st-key-remove_"]').nth(index).locator("button").first.click()

        def cards():
            return sidebar.locator(".dm-card .name").all_inner_texts()

        check("limit notice shown", sidebar.get_by_text("Maximum 5 PDFs can be uploaded.").count() == 1)
        check("chat input disabled before any source", page.locator("textarea[data-testid='stChatInputTextArea']").is_disabled())
        shot("1_empty")

        upload([pdfs[0]])
        check("1 PDF uploaded", cards() == ["pdf1.pdf"], cards())
        check("count shows 1/5", sidebar.get_by_text("1/5").count() >= 1)

        upload([pdfs[1], pdfs[2]])
        check("3 PDFs after adding 2 more (nothing replaced)", cards() == ["pdf1.pdf", "pdf2.pdf", "pdf3.pdf"], cards())

        upload([pdfs[3], pdfs[4]])
        check("5 PDFs uploaded", len(cards()) == 5 and sidebar.get_by_text("5/5").count() >= 1, cards())
        shot("2_five")

        upload([pdfs[5]])
        check("sixth PDF refused with the exact message",
              sidebar.get_by_text("Maximum upload limit reached. You can upload only 5 PDF files.").count() == 1)
        check("still exactly 5 cards", len(cards()) == 5, cards())

        remove_nth(2)          # remove PDF 3
        page.wait_for_timeout(1500)
        check("removing PDF 3 keeps the other four",
              cards() == ["pdf1.pdf", "pdf2.pdf", "pdf4.pdf", "pdf5.pdf"], cards())

        upload([pdfs[6]])
        check("replacement PDF added after removal",
              len(cards()) == 5 and "pdf7.pdf" in cards() and "pdf1.pdf" in cards(), cards())

        upload([str(work / "scan.pdf")])
        check("seventh PDF (also invalid) is still refused cleanly", len(cards()) == 5)

        # ask across the five PDFs
        box = page.locator("textarea[data-testid='stChatInputTextArea']")
        box.fill("How much does each Gizmo device weigh and what power does it use?")
        box.press("Enter")
        page.wait_for_selector("text=Sources (", timeout=20000)
        check("answer shows numbered source list", page.get_by_text("Sources (").count() >= 1)
        page.get_by_text("Sources (").first.click()
        page.wait_for_timeout(500)
        body = page.inner_text("body")
        names_in_answer = sum(f"pdf{n}.pdf" in body for n in (1, 2, 4, 5, 7))
        check("answer draws on several different PDFs", names_in_answer >= 4, names_in_answer)
        check("removed pdf3 is not cited", "pdf3.pdf" not in body.split("Sources (")[-1])
        shot("3_answer")

        # clear everything with X until empty, then txt / md / conflict
        for _ in range(5):
            remove_nth(0)
            page.wait_for_timeout(900)
        check("all sources removable", cards() == [], cards())

        upload([str(work / "notes.txt"), str(work / "guide.md")])
        check("TXT and MD uploads accepted", cards() == ["notes.txt", "guide.md"], cards())

        upload([str(work / "sheet.xlsx")])
        check("unsupported .xlsx gives a clear message",
              sidebar.get_by_text("is not supported").count() >= 1)

        upload([str(work / "eiffel_a.pdf"), str(work / "eiffel_b.pdf")])
        box.fill("How tall is the Eiffel Tower?")
        box.press("Enter")
        page.wait_for_selector("text=Conflicting information detected in the uploaded documents.", timeout=20000)
        check("conflict banner shown", True)
        check("source comparison present", page.get_by_text("Source comparison").count() >= 1)
        shot("4_conflict")

        # YouTube (mocked transcript)
        sidebar.get_by_placeholder("Paste a YouTube link").fill("https://youtu.be/dQw4w9WgXcQ")
        sidebar.get_by_role("button", name="Submit").click()
        page.wait_for_timeout(2500)
        check("YouTube card added", any("RAG explained" in c for c in cards()), cards())
        check("green 'Submitted' box stays visible", sidebar.get_by_text("Submitted!").count() == 1)
        page.mouse.move(700, 400)                       # move off the button so it is not in its hover colour
        page.wait_for_timeout(300)
        check("Submit button is orange with black text",
              sidebar.get_by_role("button", name="Submit").evaluate(
                  "b => { const s = getComputedStyle(b); return s.backgroundColor === 'rgb(255, 159, 28)' && s.color === 'rgb(0, 0, 0)'; }"))
        check("'Press Enter to submit' hint is hidden",
              not sidebar.get_by_text("Press Enter to submit form").is_visible())
        box.fill("What does RAG mean according to the video?")
        box.press("Enter")
        page.wait_for_timeout(4000)
        links = page.locator("a[href*='youtube.com/watch']").evaluate_all("els => els.map(e => e.href)")
        check("timestamp link opens video at 01:25",
              any("v=dQw4w9WgXcQ" in l and "t=85s" in l for l in links), links)
        shot("5_video")

        sidebar.get_by_placeholder("Paste a YouTube link").fill("https://example.com/nope")
        sidebar.get_by_role("button", name="Submit").click()
        page.wait_for_timeout(1500)
        check("invalid YouTube link gives a clear message",
              sidebar.get_by_text("valid YouTube video link").count() >= 1)

        # ---- slider: one source with many pages; the number of retrieved passages must follow the slider
        for _ in range(len(cards())):
            remove_nth(0)
            page.wait_for_timeout(700)
        (work / "lab.pdf").write_bytes(make_pdf(*[f"Quantum entanglement experiment number {n} results." for n in range(14)]))
        upload([str(work / "lab.pdf")])
        slider = sidebar.locator('[data-testid="stSlider"] input[type="range"]').first
        slider.scroll_into_view_if_needed()
        check("slider starts at 6", int(slider.input_value()) == 6, slider.input_value())
        for presses, expected in ((-3, 3), (+5, 8), (-6, 2), (+8, 10)):
            slider.focus()
            key = "ArrowRight" if presses > 0 else "ArrowLeft"
            for _ in range(abs(presses)):
                page.keyboard.press(key)
                page.wait_for_timeout(350)
            page.wait_for_timeout(600)
            check(f"slider moved to {expected} and shows it",
                  int(slider.input_value()) == expected
                  and sidebar.get_by_text("passages.", exact=False).first.inner_text().count(str(expected)) == 1,
                  slider.input_value())
            box.fill("quantum entanglement experiment results")
            box.press("Enter")
            page.wait_for_selector(f"text=Sources ({expected})", timeout=20000)
            check(f"question uses {expected} passages", page.get_by_text(f"Sources ({expected})").count() >= 1)
            slider.scroll_into_view_if_needed()
        shot("7_slider")

        # ---- answer modes, single "General knowledge" label, a document that contradicts itself
        for _ in range(len(cards())):
            remove_nth(0)
            page.wait_for_timeout(700)
        (work / "ML.pdf").write_bytes(make_pdf(
            "The capital of Andhra Pradesh is Amaravati. The capital of Andhra Pradesh is Hyderabad."))
        upload([str(work / "ML.pdf")])
        box.fill("What is the capital of Andhra Pradesh?")
        box.press("Enter")
        page.wait_for_selector("text=Conflicting information detected in the uploaded documents.", timeout=20000)
        check("conflict inside ONE document shows the banner", True)
        page.wait_for_selector("text=Conflicting statements", timeout=10000)
        check("conflicting statements are quoted from the document",
              page.get_by_text("The capital of Andhra Pradesh is Amaravati").count() >= 1
              and page.get_by_text("The capital of Andhra Pradesh is Hyderabad").count() >= 1)
        shot("8_self_conflict")

        box.fill("What is the capital of Japan?")
        box.press("Enter")
        page.wait_for_selector("text=general-knowledge demo answer", timeout=20000)
        badges = page.get_by_text("General knowledge · not found in your uploaded sources").count()
        check("general-knowledge label appears exactly once", badges == 1, badges)
        check("no 'Source comparison is not available' note on a general answer",
              page.get_by_text("Source comparison is not available").count() == 0)

        sidebar.get_by_text("Document only", exact=True).click()
        page.wait_for_timeout(1200)
        box.fill("What is the capital of Peru?")
        box.press("Enter")
        page.wait_for_selector("text=I could not find sufficient information", timeout=20000)
        check("Document only mode refuses instead of answering from general knowledge", True)
        shot("9_modes")

        page.set_viewport_size({"width": 390, "height": 800})
        page.wait_for_timeout(800)
        shot("6_mobile")
        check("no horizontal scroll on mobile",
              page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))
        browser.close()

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} browser checks passed")
    sys.exit(1 if failed else 0)


main()
