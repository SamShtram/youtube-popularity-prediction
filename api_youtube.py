from googleapiclient.discovery import build
import pandas as pd
from dotenv import load_dotenv
import os, pathlib
print("🔑 Using API key prefix:", API_KEY[:8])
print("📁 Saving to:", SAVE_PATH)
# Load API key
env_path = pathlib.Path("/content/drive/MyDrive/youtube-popularity-prediction/.env")
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("api_key")

# Initialize YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_trending_videos(region="US", max_results=3000):
    videos = []
    next_page_token = None

    print(f"🌎 Collecting trending videos for region: {region}")

    while len(videos) < max_results:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            chart="mostPopular",
            regionCode=region,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            videos.append({
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

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    print(f"✅ Collected {len(videos)} videos total.")
    return pd.DataFrame(videos)
