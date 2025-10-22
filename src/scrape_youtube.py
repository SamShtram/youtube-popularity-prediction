import os
import re
import json
import time
import random
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# === Setup ===
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

KEYWORDS = [
    "music video", "podcast", "sports highlights", "tech review",
    "news update", "tutorial", "education", "vlog", "movie trailer",
    "finance tips", "travel vlog", "gaming", "reaction video",
    "motivational speech", "fashion haul", "documentary", "interview",
    "live performance", "cooking recipe", "product review"
]

SAVE_PATH = "../data/youtube_scraped_raw.csv"
MAX_VIDEOS = 3000

os.makedirs(os.path.join("..", "data"), exist_ok=True)


# === Extract video data ===
def extract_videos_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")

    json_data = None
    for script in scripts:
        if "ytInitialData" in script.text:
            match = re.search(r"ytInitialData\s*=\s*(\{.*?\});", script.text, re.S)
            if match:
                try:
                    json_data = json.loads(match.group(1))
                    break
                except Exception:
                    continue

    if not json_data:
        print("No ytInitialData found.")
        return []

    videos = []
    try:
        sections = json_data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
        for section in sections:
            contents = section.get("itemSectionRenderer", {}).get("contents", [])
            for item in contents:
                video = item.get("videoRenderer")
                if not video:
                    continue

                vid = video.get("videoId")
                title = video.get("title", {}).get("runs", [{}])[0].get("text", "")
                channel = video.get("ownerText", {}).get("runs", [{}])[0].get("text", "")
                views = video.get("viewCountText", {}).get("simpleText", "N/A")
                duration = video.get("lengthText", {}).get("simpleText", "N/A")

                videos.append({
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": title,
                    "channel": channel,
                    "views": views,
                    "duration": duration
                })

    except Exception as e:
        print("Error traversing JSON:", e)

    print(f"Extracted {len(videos)} videos from this keyword.")
    return videos


# === Main scraper ===
def scrape_youtube_data():
    all_videos = []
    for keyword in tqdm(KEYWORDS, desc="Scraping YouTube search results"):
        search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}"
        try:
            response = requests.get(search_url, headers=HEADERS, timeout=10)
            response.encoding = "utf-8"
        except Exception as e:
            print(f"Request failed for {keyword}: {e}")
            continue

        videos = extract_videos_from_html(response.text)
        all_videos.extend(videos)

        if len(all_videos) >= MAX_VIDEOS:
            break

        time.sleep(random.uniform(2, 4))

    df = pd.DataFrame(all_videos)
    df.drop_duplicates(subset="url", inplace=True)
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(df)} videos to {SAVE_PATH}")


if __name__ == "__main__":
    scrape_youtube_data()
