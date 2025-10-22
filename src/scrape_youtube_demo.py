"""
scrape_youtube_demo.py
Lightweight demonstration version of the full YouTube video scraper.
Runs in under ~90 seconds and collects ~300 samples for grading/demo.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random

def scrape_youtube_demo(output_path="../data/youtube_scraped_demo.csv"):
    print("🎬 DEMO MODE: Collecting YouTube video data (limited sample)...")

    START_TIME = time.time()
    MAX_RUNTIME = 90   # seconds
    MAX_SAMPLES = 300  # demo limit
    BASE_URL = "https://www.youtube.com/feed/trending"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(BASE_URL, headers=headers)
    if response.status_code != 200:
        print(f" Request failed with status code {response.status_code}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    videos = []

    # Attempt to extract video metadata from trending section
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/watch?v=" in href:
            title = a_tag.text.strip()
            video_url = f"https://www.youtube.com{href}"
            videos.append({"title": title, "url": video_url})

            # Stop conditions
            if len(videos) >= MAX_SAMPLES:
                print(f" Max sample limit reached ({MAX_SAMPLES}) — stopping early.")
                break
            if time.time() - START_TIME > MAX_RUNTIME:
                print(f" Max runtime reached ({MAX_RUNTIME}s) — stopping early.")
                break

            time.sleep(random.uniform(0.05, 0.2))  # simulate request delay

    # Convert to DataFrame and save
    df = pd.DataFrame(videos)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = round(time.time() - START_TIME, 2)
    print(f" DEMO SCRAPER completed in {elapsed}s — collected {len(df)} videos.")
    print(f" Saved demo data to → {output_path}")

if __name__ == "__main__":
    scrape_youtube_demo()
