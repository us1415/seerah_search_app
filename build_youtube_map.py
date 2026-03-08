import csv
import json
import os
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLAEA99D24CA2F9A8F"
TRANSCRIPTS_DIR = Path("./transcripts")
OUTPUT_CSV = Path("./youtube_map.csv")


def normalize(text: str) -> str:
    text = text.lower()

    # remove extension
    text = re.sub(r"\.txt$", "", text)

    # unify common spellings
    replacements = {
        "muhammed": "muhammad",
        "dr.": "dr",
        "yasir qadhi": "yasir qadhi",
        "prophet ﷺ": "prophet",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # remove punctuation and extra spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def load_playlist_entries() -> list[dict]:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        PLAYLIST_URL,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    entries = []
    for entry in data.get("entries", []):
        title = entry.get("title", "").strip()
        url = f"https://www.youtube.com/watch?v={entry['id']}"
        entries.append({
            "title": title,
            "url": url,
            "norm_title": normalize(title),
        })
    return entries


def load_transcripts() -> list[dict]:
    transcripts = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        transcripts.append({
            "file": path.name,
            "norm_file": normalize(path.name),
        })
    return transcripts


def main():
    if not TRANSCRIPTS_DIR.exists():
        raise FileNotFoundError(f"Transcript folder not found: {TRANSCRIPTS_DIR.resolve()}")

    playlist_entries = load_playlist_entries()
    transcripts = load_transcripts()

    rows = []
    unmatched = []

    for transcript in transcripts:
        best_match = None
        best_score = 0.0

        for video in playlist_entries:
            score = similarity(transcript["norm_file"], video["norm_title"])
            if score > best_score:
                best_score = score
                best_match = video

        if best_match and best_score >= 0.55:
            rows.append({
                "file": transcript["file"],
                "youtube_url": best_match["url"],
                "matched_title": best_match["title"],
                "score": round(best_score, 4),
            })
        else:
            unmatched.append(transcript["file"])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "youtube_url", "matched_title", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {OUTPUT_CSV} with {len(rows)} matched rows.")
    if unmatched:
        print("\nUnmatched transcripts:")
        for item in unmatched:
            print(f"  {item}")


if __name__ == "__main__":
    main()