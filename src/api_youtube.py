# === Load API Key ===
import os
import pathlib
from dotenv import load_dotenv

# Look for .env in project root
env_path = pathlib.Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Load from .env or GitHub Secrets
API_KEY = os.getenv("YOUTUBE_API_KEY") or os.getenv("api_key")

if not API_KEY:
    raise ValueError(" YouTube API key not found. Add it to your .env or GitHub Secrets.")
else:
    print(" API key loaded successfully.")

# === Output file ===
SAVE_PATH = "/content/drive/MyDrive/youtube-popularity-prediction/data/youtube_api_3000.csv"

# === YouTube regions (to reach 3000 videos total) ===
REGIONS = ["US", "IN", "GB", "BR", "JP", "KR", "FR", "DE", "CA", "MX", "RU", "IT", "AU", "ES", "ID"]

def get_trending_videos(region="US", max_results=300):
    """Collect trending videos from a single region."""
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
            print(f"API error {response.status_code} for region {region}: {data}")
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

        time.sleep(1)  # polite delay to avoid quota spikes

    return videos


if __name__ == "__main__":
    all_videos = []
    for region in REGIONS:
        print(f" Collecting for region: {region}")
        region_videos = get_trending_videos(region, max_results=300)
        all_videos.extend(region_videos)
        print(f" Collected {len(region_videos)} from {region}. Total so far: {len(all_videos)}")

    df = pd.DataFrame(all_videos)
    df.to_csv(SAVE_PATH, index=False)
    print(f"\n Saved {len(df)} total videos to {SAVE_PATH}")
