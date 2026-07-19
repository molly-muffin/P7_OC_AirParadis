"""Capture a real Streamlit demo screenshot: prediction + feedback buttons visible."""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots" / "streamlit_demo.png"
TWEET = "I love flying with Air Paradis, the crew was amazing!"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto("http://localhost:8502", wait_until="networkidle")
        page.wait_for_timeout(3000)

        page.get_by_role("textbox").last.fill(TWEET)
        page.get_by_role("button", name="Analyser le sentiment").click()

        # Wait for the prediction result and feedback section to render
        page.wait_for_selector("text=Sentiment prédit", timeout=60000)
        page.wait_for_selector("text=La prédiction est-elle correcte", timeout=30000)
        page.wait_for_timeout(1500)

        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
        print(f"saved {OUT}")


if __name__ == "__main__":
    main()
