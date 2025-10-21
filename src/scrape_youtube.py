import requests, re, time, random, pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

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

SAVE_PATH = "../data/youtube_scraped_3000.csv"
MAX_VIDEOS = 3000


def extract_video_ids_from_search(html):
    return list(set(re.findall(r'"videoId":"(.*?)"', html)))


def scrape_video_metadata(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    resp = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(resp.text, "html.parser")

    data = {
        "url": url,
        "title": None,
        "channel": None,
        "views": None,
        "upload_date": None,
        "category": None,
        "duration": None,
        "tags": None,
    }

    try:
        data["title"] = soup.find("meta", {"name": "title"})["content"]
    except:
        pass

    try:
        data["channel"] = soup.find("link", {"itemprop": "name"})["content"]
    except:
        pass

    try:
        data["views"] = re.search(r'"viewCount":"(\d+)"', resp.text).group(1)
    except:
        pass

    try:
        data["upload_date"] = soup.find("meta", {"itemprop": "uploadDate"})["content"]
    except:
        pass

    try:
        data["duration"] = soup.find("meta", {"itemprop": "duration"})["content"]
    except:
        pass

    try:
        data["category"] = soup.find("meta", {"itemprop": "genre"})["content"]
    except:
        pass

    try:
        tags = soup.find_all("meta", {"property": "og:video:tag"})
        data["tags"] = ", ".join([t["content"] for t in tags])
    except:
        pass

    return data


def scrape_youtube_data():
    all_videos, seen_ids = [], set()

    for keyword in tqdm(KEYWORDS, desc="🔍 Scraping search results"):
        search_url = f"https://www.youtube.com/results?search_query={keyword.replace(' ', '+')}"
        response = requests.get(search_url, headers=HEADERS)
        video_ids = extract_video_ids_from_search(response.text)

        for vid in video_ids:
            if vid not in seen_ids:
                try:
                    info = scrape_video_metadata(vid)
                    all_videos.append(info)
                    seen_ids.add(vid)
                except Exception as e:
                    print(f"⚠️ Error scraping video {vid}: {e}")
                time.sleep(random.uniform(1.5, 3.0))

            if len(all_videos) >= MAX_VIDEOS:
                break
        if len(all_videos) >= MAX_VIDEOS:
            break

    df = pd.DataFrame(all_videos)
    df.drop_duplicates(subset="url", inplace=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"✅ Saved {len(df)} videos to {SAVE_PATH}")


if __name__ == "__main__":
    scrape_youtube_data()

