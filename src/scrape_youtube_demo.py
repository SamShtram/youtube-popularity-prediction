import requests, json, re, time, random, pandas as pd
from tqdm import tqdm

os.makedirs("data", exist_ok=True)
SAVE_PATH = os.path.join("data", "youtube_scraped_raw.csv")
MAX_VIDEOS = 3000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0 Safari/537.36"
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


def extract_videos_from_json(html):
    """Extract video metadata from YouTube search HTML."""
    import json, re

    match = re.search(r"var ytInitialData\s*=\s*(\{.*?\});</script>", html, re.S)
    if not match:
        match = re.search(r"ytInitialData\"[:=]\s*(\{.*?\})\s*;</script>", html, re.S)
    if not match:
        print("No ytInitialData found.")
        return []

    try:
        json_text = match.group(1)
        data = json.loads(json_text)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

    videos = []
    try:
        sections = (
            data.get("contents", {})
                .get("twoColumnSearchResultsRenderer", {})
                .get("primaryContents", {})
                .get("sectionListRenderer", {})
                .get("contents", [])
        )

        for section in sections:
            contents = section.get("itemSectionRenderer", {}).get("contents", [])
            for item in contents:
                video = item.get("videoRenderer")
                if not video:
                    continue

                vid = video.get("videoId")
                if not vid:
                    continue

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
        print(f"JSON traversal error: {e}")

    print(f"Extracted {len(videos)} videos from page.")
    return videos


def scrape_youtube_data():
    all_videos = []
    for keyword in tqdm(KEYWORDS, desc=" Scraping search results"):
        search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}"
        resp = requests.get(search_url, headers=HEADERS)
        videos = extract_videos_from_json(resp.text)

        for video in videos:
            all_videos.append(video)
            if len(all_videos) >= MAX_VIDEOS:
                break
        if len(all_videos) >= MAX_VIDEOS:
            break
        time.sleep(random.uniform(2, 4))

    df = pd.DataFrame(all_videos)
    df.drop_duplicates(subset="url", inplace=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f" Saved {len(df)} videos to {SAVE_PATH}")


if __name__ == "__main__":
    scrape_youtube_data()
