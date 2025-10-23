import os
import re
import json
import time
import random
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

# -------------------------------
# Configuration
# -------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

KEYWORDS = [
    # Music & Entertainment
    "music video", "official music video", "song release", "acoustic cover", "pop hits",
    "rap freestyle", "hip hop mix", "country music", "EDM festival", "live concert",
    "guitar solo", "drum cover", "piano performance", "instrumental music", "top music charts",

    # Movies & Trailers
    "movie trailer", "film teaser", "behind the scenes movie", "short film", "animation short",
    "fan edit", "movie recap", "movie explanation", "movie review", "film analysis",

    # Gaming
    "gaming highlights", "game walkthrough", "let's play", "speedrun", "esports match",
    "minecraft survival", "fortnite gameplay", "valorant highlights", "call of duty montage", "roblox tycoon",
    "gta roleplay", "fifa gameplay", "apex legends clips", "league of legends match", "retro gaming",

    # Technology & Reviews
    "tech review", "unboxing video", "gadget review", "smartphone comparison", "camera test",
    "laptop review", "PC build tutorial", "best budget phones", "apple event", "android tips",
    "AI tools 2025", "coding tutorial", "python tutorial", "web development course", "programming for beginners",

    # Sports
    "sports highlights", "football match", "basketball highlights", "soccer skills", "baseball game",
    "tennis rally", "UFC fight", "boxing knockout", "cricket highlights", "F1 race",
    "golf tips", "olympics recap", "NBA game", "NFL highlights", "hockey goals",

    # News & Politics
    "breaking news", "daily news update", "political debate", "global economy", "finance news",
    "business report", "tech news", "science news", "climate change news", "weather forecast",

    # Education & Learning
    "math tutorial", "science experiment", "physics explained", "chemistry reaction", "biology lecture",
    "history documentary", "world war history", "language learning", "english grammar tips", "study motivation",
    "AI lecture", "data science course", "machine learning basics", "college vlog", "study techniques",

    # Lifestyle & Vlogs
    "daily vlog", "travel vlog", "morning routine", "night routine", "lifestyle vlog",
    "week in my life", "day in the life", "home organization", "minimalist lifestyle", "college dorm tour",
    "family vlog", "pet vlog", "couple vlog", "moving vlog", "travel diary",

    # Food & Cooking
    "cooking recipe", "easy dinner ideas", "street food tour", "baking tutorial", "meal prep ideas",
    "healthy breakfast", "vegan recipe", "dessert recipe", "grilling tips", "restaurant review",
    "food challenge", "cake decorating", "instant pot recipes", "homemade pizza", "coffee brewing",

    # Fashion & Beauty
    "makeup tutorial", "fashion haul", "outfit ideas", "skincare routine", "hair styling tips",
    "nail art tutorial", "wardrobe organization", "perfume review", "clothing try on", "summer fashion",
    "winter outfits", "beauty hacks", "get ready with me", "celebrity fashion", "makeup review",

    # Finance & Business
    "investing for beginners", "stock market analysis", "cryptocurrency news", "passive income ideas", "real estate investing",
    "business motivation", "entrepreneurship tips", "startup pitch", "finance tips", "budgeting hacks",
    "saving money tips", "credit score advice", "debt payoff strategy", "financial freedom", "side hustle ideas",

    # Motivation & Self-Improvement
    "motivational speech", "self improvement tips", "productivity hacks", "morning motivation", "goal setting",
    "discipline habits", "mental health awareness", "how to stay focused", "study motivation", "career advice",
    "success stories", "confidence building", "public speaking tips", "personal growth", "habit tracking",

    # Comedy & Reactions
    "funny moments", "comedy skit", "stand up comedy", "funny fails", "reaction video",
    "prank video", "meme compilation", "tiktok compilation", "try not to laugh", "parody video",
    "best vines", "funny short", "spoof video", "viral trends", "humor sketches",

    # DIY & Crafts
    "DIY home decor", "craft tutorial", "room makeover", "art tutorial", "painting tutorial",
    "drawing for beginners", "woodworking project", "resin art", "upcycling ideas", "knitting tutorial",

    # Health & Fitness
    "home workout", "gym motivation", "fitness routine", "yoga session", "pilates class",
    "weight loss tips", "nutrition advice", "healthy lifestyle", "mental wellness", "stretching routine",
    "bodybuilding motivation", "running tips", "cycling workout", "posture correction", "healthy habits",

    # Miscellaneous & Trending
    "reaction to viral video", "AI trends", "luxury lifestyle", "celebrity interview", "car review",
    "travel destination", "ASMR sounds", "space documentary", "nature sounds", "wildlife documentary",
    "top 10 list", "life hacks", "satisfying video", "optical illusion", "weird facts"
]



# Save to /data/youtube_scraped_raw.csv (relative to project root)
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "youtube_scraped_raw.csv")
MAX_VIDEOS = 3000

# Ensure /data directory exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)


# -------------------------------
# Helper: Extract videos from HTML
# -------------------------------
def extract_videos_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")

    json_data = None
    for script in scripts:
        if "ytInitialData" in script.text:
            match = re.search(r"ytInitialData\s*=\s*(\{.*\})\s*;", script.text, re.DOTALL)
            if match:
                try:
                    json_data = json.loads(match.group(1))
                    break
                except Exception:
                    continue

    if not json_data:
        print("No ytInitialData found in page.")
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
        print("Error parsing YouTube data:", e)

    return videos


# -------------------------------
# Main Scraper Function
# -------------------------------
def scrape_youtube_data():
    all_videos = []
    print("Starting YouTube scraping process...")

    for keyword in tqdm(KEYWORDS, desc="Scraping YouTube search results"):
        attempt = 1
        while attempt <= 2:
            search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}"
            try:
                response = requests.get(search_url, headers=HEADERS, timeout=10)
                response.encoding = "utf-8"
            except Exception as e:
                print(f"Request failed for keyword '{keyword}': {e}")
                break

            videos = extract_videos_from_html(response.text)
            print(f"{keyword}: {len(videos)} videos found (attempt {attempt})")

            if len(videos) >= 10 or attempt == 2:
                all_videos.extend(videos)
                break

            attempt += 1
            time.sleep(random.uniform(3, 6))

        if len(all_videos) >= MAX_VIDEOS:
            break

        time.sleep(random.uniform(2, 4))

    # Save to CSV
    df = pd.DataFrame(all_videos)
    df.drop_duplicates(subset="url", inplace=True)
    df.to_csv(SAVE_PATH, index=False, encoding="utf-8")

    print(f"\nScraping completed.")
    print(f"Saved {len(df)} videos to {SAVE_PATH}")


# -------------------------------
# Run script
# -------------------------------
if __name__ == "__main__":
    scrape_youtube_data()
