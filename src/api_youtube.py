import os
import pathlib
from dotenv import load_dotenv

# Load environment variables
env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

API_KEY = os.getenv("YOUTUBE_API_KEY") or os.getenv("api_key")

# If API key not found, switch to demo mode
if not API_KEY:
    print("⚠️ No API key found. Running in demo mode using pre-collected sample data.")
    DEMO_FILE = pathlib.Path(__file__).resolve().parent.parent / "data" / "youtube_api_sample.csv"

    if DEMO_FILE.exists():
        import pandas as pd
        df = pd.read_csv(DEMO_FILE)
        print(f"Loaded demo dataset with {len(df)} rows from {DEMO_FILE}")
        exit(0)
    else:
        raise ValueError("No API key or demo file found. Please create .env or include sample CSV.")

# === Output file ===
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
SAVE_PATH = DATA_DIR / "youtube_api_raw.csv"

# === YouTube regions (to reach 3000 total videos) ===
REGIONS = ["US", "IN", "GB", "BR", "JP", "KR", "FR", "DE", "CA", "MX", "RU", "IT", "AU", "ES", "ID"]

def get_trending_videos(region="US", max_results=300):
    """Collect trending videos from a single region using YouTube Data API."""
    base_url = "https://www.googleapis.com/youtube/v3/videos"
    videos = []
    next_page_token = None

    while len(videos) < max_results:
        params = {
            "part": "snippet,contentDetails,statistics",
            "chart": "mostPopular",
            "regionCode": region,
            "maxResults": 50,
            "pageToken": next_page_token,
            "key": API_KEY
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code != 200:
            print(f" API error {response.status_code} for region {region}: {data}")
            break

        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            videos.append({
                "region": region,
                "video_id": item.get("id"),
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "category_id": snippet.get("categoryId"),
                "views": stats.get("viewCount"),
                "likes": stats.get("likeCount"),
                "comments": stats.get("commentCount"),
                "upload_date": snippet.get("publishedAt"),
                "duration": content.get("duration"),
                "tags": ", ".join(snippet.get("tags", [])) if "tags" in snippet else "",
                "description": snippet.get("description", "")
            })

            if len(videos) >= max_results:
                break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.25)  # polite delay

    return videos


if __name__ == "__main__":
    print("🎥 Collecting YouTube trending data via API...\n")
    all_videos = []

    for region in tqdm(REGIONS, desc=" Fetching by region"):
        region_videos = get_trending_videos(region, max_results=300)
        all_videos.extend(region_videos)
        print(f" {region}: Collected {len(region_videos)} videos (Total: {len(all_videos)})")

    df = pd.DataFrame(all_videos)
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"\n Saved {len(df)} total videos → {SAVE_PATH}")
