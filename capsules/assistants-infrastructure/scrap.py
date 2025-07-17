import asyncio
from bs4 import BeautifulSoup
import httpx
import os
import shutil
from datetime import datetime

URLS = {
    "Overall": "https://lmarena.ai/leaderboard/text/overall",
    "Math":    "https://lmarena.ai/leaderboard/text/math",
    "English": "https://lmarena.ai/leaderboard/text/english",
    "Spanish": "https://lmarena.ai/leaderboard/text/spanish",
    "Coding":  "https://lmarena.ai/leaderboard/text/coding",
}

LEADERBOARDS_FOLDER = "leaderboards"


def recreate_leaderboards_folder():
    """Remove and recreate the leaderboards folder to ensure clean state."""
    if os.path.exists(LEADERBOARDS_FOLDER):
        shutil.rmtree(LEADERBOARDS_FOLDER)
    os.makedirs(LEADERBOARDS_FOLDER)


async def fetch_table(page, url):
    await page.goto(url, wait_until="networkidle")
    try:
        await page.wait_for_selector("table tbody tr", timeout=60000)
    except Exception as e:
        print(f"Error waiting for table at {url}: {e}")
        return []

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("table tbody tr")
    data = []
    for tr in rows:
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) < 7:  # ensure we have enough columns
            continue
        # cols layout: [Rank, Model, Score, 95% CI, Votes, Organization, License]
        license_type = cols[-1]  # last column
        if license_type.lower() == "proprietary":
            continue  # skip proprietary models
        rank = cols[0]
        model = cols[1]
        score = cols[2]
        data.append((rank, model, score))
    return data


async def scrape_and_create_markdown(name, url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if table is None:
        raise RuntimeError(
            f"Could not find any <table> tag on {name} leaderboard page."
        )

    header_cells = table.find("thead").find_all("th")
    headers = [th.get_text(strip=True) for th in header_cells]

    body_rows = table.find("tbody").find_all("tr")
    rows: list[list[str]] = []
    
    for tr in body_rows:
        cell_texts = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cell_texts and len(cell_texts) >= 7:
            license_type = cell_texts[-1]
            if license_type.lower() != "proprietary":
                rows.append(cell_texts)

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = []
    md.append(f"# {name} Leaderboard\n")
    md.append(f"*Data scraped on: {current_date}*\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join("---" for _ in headers) + " |")

    for row in rows:
        md.append("| " + " | ".join(row) + " |")

    markdown_content = "\n".join(md)
    
    filename = f"{name.lower()}_leaderboard.md"
    filepath = os.path.join(LEADERBOARDS_FOLDER, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"Created {filepath}")
    return markdown_content


async def main():
    recreate_leaderboards_folder()
    print(f"Recreated {LEADERBOARDS_FOLDER} folder")
    
    for name, url in URLS.items():
        try:
            await scrape_and_create_markdown(name, url)
        except Exception as e:
            print(f"Error processing {name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
