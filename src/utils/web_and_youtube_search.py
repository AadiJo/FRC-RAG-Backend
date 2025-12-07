import os
import requests

REQUEST_TIMEOUT = 8

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Web search using SerpAPI
def web_search(query, num_results=3, api_key=None):
    key_to_use = api_key if api_key else SERPAPI_KEY
    if not key_to_use:
        print("[DEBUG] web_search skipped: missing SERPAPI_KEY")
        return []
    params = {
        "q": query,
        "api_key": key_to_use,
        "num": num_results,
        "engine": "google",
    }
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        print(f"[ERROR] web_search request failed: {e}")
        return []
    print(f"[DEBUG] web_search status={resp.status_code}")
    results = []
    if resp.status_code == 200:
        data = resp.json()
        for r in data.get("organic_results", []):
            results.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet", "")
            })
    else:
        print(f"[ERROR] web_search non-200: {resp.text[:200]}")
    return results

# YouTube search using YouTube Data API
def youtube_search(query, num_results=3, api_key=None):
    key_to_use = api_key if api_key else YOUTUBE_API_KEY
    if not key_to_use:
        print("[DEBUG] youtube_search skipped: missing YOUTUBE_API_KEY")
        return []
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": num_results,
        "key": key_to_use,
    }
    try:
        resp = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        print(f"[ERROR] youtube_search request failed: {e}")
        return []
    print(f"[DEBUG] youtube_search status={resp.status_code}")
    results = []
    if resp.status_code == 200:
        data = resp.json()
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            results.append({
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "link": f"https://www.youtube.com/watch?v={video_id}",
                "description": snippet.get("description", "")
            })
    else:
        print(f"[ERROR] youtube_search non-200: {resp.text[:200]}")
    return results
