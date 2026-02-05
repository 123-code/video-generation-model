"""
Download videos from Pexels API for video generation training.
Requires PEXELS_API_KEY environment variable.
"""
import os
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Categories to download (matching your 11 classes)
CATEGORIES = [
    "bird",
    "cat",
    "clouds",
    "dog",
    "dolphin",
    "fish",
    "horse",
    "rabbit",
    "snake",
    "tiger",
    "wolf",
]

VIDEOS_PER_CATEGORY = 1500  # Target videos per category
OUTPUT_DIR = "videos"
PEXELS_API_URL = "https://api.pexels.com/videos/search"


def get_api_key():
    """Get Pexels API key from environment."""
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise ValueError(
            "PEXELS_API_KEY environment variable not set. "
            "Get your free API key at https://www.pexels.com/api/"
        )
    return api_key


def search_videos(query: str, api_key: str, per_page: int = 80, page: int = 1):
    """Search for videos on Pexels."""
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "per_page": per_page,
        "page": page,
        "orientation": "landscape",  # Better for video generation
    }

    response = requests.get(PEXELS_API_URL, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        print(f"Rate limited, waiting 60s...")
        time.sleep(60)
        return search_videos(query, api_key, per_page, page)
    else:
        print(f"API error {response.status_code}: {response.text}")
        return None


def get_best_video_file(video_data: dict, target_height: int = 720):
    """Get the best quality video file URL (prefer 720p or closest)."""
    video_files = video_data.get("video_files", [])

    # Sort by height, prefer closest to target
    valid_files = [f for f in video_files if f.get("height") and f.get("link")]
    if not valid_files:
        return None

    # Prefer HD quality (720p) for good quality without huge files
    valid_files.sort(key=lambda x: abs(x["height"] - target_height))
    return valid_files[0]["link"]


def download_video(url: str, output_path: Path) -> bool:
    """Download a single video."""
    try:
        if output_path.exists():
            return True

        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Download failed for {output_path.name}: {e}")
    return False


def download_category(category: str, api_key: str, output_dir: Path, max_videos: int):
    """Download videos for a single category."""
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Check existing videos
    existing = list(category_dir.glob("*.mp4"))
    if len(existing) >= max_videos:
        print(f"Category '{category}' already has {len(existing)} videos, skipping.")
        return len(existing)

    downloaded = len(existing)
    page = 1
    video_urls = []

    print(f"\nSearching for '{category}' videos...")

    # Collect video URLs
    while len(video_urls) < max_videos:
        result = search_videos(category, api_key, per_page=80, page=page)
        if not result or not result.get("videos"):
            break

        for video in result["videos"]:
            url = get_best_video_file(video)
            if url:
                video_id = video["id"]
                video_urls.append((url, category_dir / f"{video_id}.mp4"))

        if len(result["videos"]) < 80:  # No more pages
            break
        page += 1
        time.sleep(0.5)  # Rate limiting

    print(f"Found {len(video_urls)} videos for '{category}'")

    # Download videos
    to_download = [(url, path) for url, path in video_urls if not path.exists()]
    to_download = to_download[:max_videos - downloaded]

    if not to_download:
        return downloaded

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_video, url, path): path
            for url, path in to_download
        }

        pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {category}")
        for future in pbar:
            if future.result():
                downloaded += 1
            pbar.set_postfix(downloaded=downloaded)

    return downloaded


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download videos from Pexels")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--videos_per_category", type=int, default=VIDEOS_PER_CATEGORY)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = parser.parse_args()

    api_key = get_api_key()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading videos to: {output_dir}")
    print(f"Categories: {args.categories}")
    print(f"Target videos per category: {args.videos_per_category}")

    total_downloaded = 0
    for category in args.categories:
        count = download_category(category, api_key, output_dir, args.videos_per_category)
        total_downloaded += count
        print(f"Category '{category}': {count} videos")

    print(f"\nTotal videos downloaded: {total_downloaded}")
    print(f"Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
