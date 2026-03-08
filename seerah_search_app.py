import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
APP_TITLE = "Seerah AI Search"
DEFAULT_TRANSCRIPTS_DIR = "./transcripts"
INDEX_FILE = "./seerah_search_index.npz"
META_FILE = "./seerah_search_meta.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8
WINDOW_SIZE = 3
GROUP_GAP_SECONDS = 30
YOUTUBE_MAP_FILE = "./youtube_map.csv"

# ---------- Helpers ----------
def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_transcript_files(folder: str):
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([f for f in p.glob("*.txt")], key=lambda x: natural_sort_key(x.name))


def extract_timestamped_lines(text: str):
    items = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        m = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.+)", line)
        if m:
            items.append({"timestamp": m.group(1), "text": m.group(2).strip()})
    return items


def build_windows(items, window_size: int = WINDOW_SIZE):
    windows = []
    if not items:
        return windows

    for i in range(len(items)):
        group = items[i:i + window_size]
        if not group:
            continue
        combined = " ".join(x["text"] for x in group).strip()
        if combined:
            windows.append({
                "timestamp": group[0]["timestamp"],
                "text": combined,
            })
    return windows


def hms_to_seconds(ts: str):
    try:
        h, m, s = map(int, ts.split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return 0


def normalize_for_matching(text: str):
    text = text.lower()
    text = text.replace("ﷺ", " ").replace("ﷻ", " ")
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    replacements = {
        r"\btafseer\b": "tafsir",
        r"\btefsir\b": "tafsir",
        r"\btafsir\b": "tafsir",
        r"\bmaududi\b": "mawdudi",
        r"\bmodudi\b": "mawdudi",
        r"\bmawdoodi\b": "mawdudi",
        r"\bbadrul\b": "badru",
        r"\bbadr\b": "badru",
        r"\balaina\b": "alayna",
        r"\balayna\b": "alayna",
        r"\btalaa\b": "tala",
        r"\btal a\b": "tala",
        r"\bawliyaa\b": "awliya",
        r"\bawliya\b": "awliya",
        r"\bkoran\b": "quran",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def expand_query(query: str):
    normalized = normalize_for_matching(query)
    expansions = {normalized}

    variant_map = {
        "tafsir": ["tafsir", "tafseer", "tefsir"],
        "mawdudi": ["mawdudi", "maududi", "modudi", "mawdoodi"],
        "badru": ["badru", "badr", "badrul"],
        "alayna": ["alayna", "alaina"],
        "tala": ["tala", "talaa"],
        "awliya": ["awliya", "awliyaa"],
    }

    words = normalized.split()
    phrase_variants = [words]

    for idx, word in enumerate(words):
        if word in variant_map:
            new_phrase_variants = []
            for phrase in phrase_variants:
                for variant in variant_map[word]:
                    new_phrase = phrase.copy()
                    new_phrase[idx] = normalize_for_matching(variant)
                    new_phrase_variants.append(new_phrase)
            phrase_variants = new_phrase_variants

    for phrase in phrase_variants:
        expansions.add(" ".join(phrase))

    return sorted(expansions)


def pretty_lecture_name(file_name: str):
    lecture_name = Path(file_name).stem
    lecture_name = re.sub(r"^seerah of prophet muhammad\s*", "Seerah ", lecture_name, flags=re.IGNORECASE)
    lecture_name = re.sub(r"^seerah of prophet muhammed\s*", "Seerah ", lecture_name, flags=re.IGNORECASE)
    lecture_name = re.sub(r"\s*~\s*dr\.?\s*yasir qadhi.*$", "", lecture_name, flags=re.IGNORECASE)
    lecture_name = re.sub(r"\s*[｜|]\s*.*$", "", lecture_name).strip()
    lecture_name = re.sub(r"\s+", " ", lecture_name)
    return lecture_name


def build_index(transcripts_dir: str, model_name: str = MODEL_NAME):
    files = load_transcript_files(transcripts_dir)
    if not files:
        raise FileNotFoundError(f"No .txt transcripts found in: {transcripts_dir}")

    corpus = []
    meta = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        items = extract_timestamped_lines(text)
        windows = build_windows(items)

        for i, window in enumerate(windows):
            raw_text = window["text"]
            corpus.append(raw_text)
            meta.append({
                "file": file_path.name,
                "chunk_id": i,
                "timestamp": window["timestamp"],
                "timestamp_seconds": hms_to_seconds(window["timestamp"]),
                "text": raw_text,
                "normalized_text": normalize_for_matching(raw_text),
            })

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        corpus,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    np.savez_compressed(INDEX_FILE, embeddings=embeddings)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return len(files), len(corpus)


def load_index():
    if not Path(INDEX_FILE).exists() or not Path(META_FILE).exists():
        return None, None
    data = np.load(INDEX_FILE)
    embeddings = data["embeddings"]
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, meta


def hybrid_search(query: str, embeddings: np.ndarray, meta: list, model_name: str = MODEL_NAME, top_k: int = TOP_K):
    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    semantic_scores = np.dot(embeddings, q[0])

    normalized_query = normalize_for_matching(query)
    query_variants = expand_query(query)

    combined_scores = []
    for idx, item in enumerate(meta):
        score = float(semantic_scores[idx])
        norm_text = item.get("normalized_text", "")

        if normalized_query and normalized_query in norm_text:
            score += 0.35

        if any(variant and variant in norm_text for variant in query_variants):
            score += 0.25

        query_tokens = [t for t in normalized_query.split() if len(t) > 2]
        if query_tokens:
            overlap = sum(1 for t in query_tokens if t in norm_text)
            score += min(0.2, overlap * 0.05)

        combined_scores.append(score)

    best_idx = np.argsort(np.array(combined_scores))[::-1][: max(top_k * 3, top_k)]

    results = []
    for idx in best_idx:
        item = dict(meta[idx])
        item["score"] = float(combined_scores[idx])
        results.append(item)
    return results


def group_results(results, group_gap_seconds: int = GROUP_GAP_SECONDS, max_groups: int = TOP_K):
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: (r["file"], r.get("timestamp_seconds", 0)))
    groups = []
    current = None

    for r in sorted_results:
        ts = r.get("timestamp_seconds", 0)
        if current is None:
            current = {
                "file": r["file"],
                "start_seconds": ts,
                "end_seconds": ts,
                "start_timestamp": r["timestamp"],
                "end_timestamp": r["timestamp"],
                "score": r["score"],
                "hits": [r],
            }
            continue

        same_file = r["file"] == current["file"]
        close_enough = ts - current["end_seconds"] <= group_gap_seconds

        if same_file and close_enough:
            current["end_seconds"] = ts
            current["end_timestamp"] = r["timestamp"]
            current["score"] = max(current["score"], r["score"])
            current["hits"].append(r)
        else:
            groups.append(current)
            current = {
                "file": r["file"],
                "start_seconds": ts,
                "end_seconds": ts,
                "start_timestamp": r["timestamp"],
                "end_timestamp": r["timestamp"],
                "score": r["score"],
                "hits": [r],
            }

    if current is not None:
        groups.append(current)

    groups = sorted(groups, key=lambda g: (g["score"], len(g["hits"])), reverse=True)
    return groups[:max_groups]


def load_youtube_map(csv_path=YOUTUBE_MAP_FILE):
    p = Path(csv_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        required = {"file", "youtube_url"}
        if not required.issubset(set(df.columns)):
            return {}
        return dict(zip(df["file"], df["youtube_url"]))
    except Exception:
        return {}


def build_youtube_timestamp_url(base_url: str, seconds: int):
    if not base_url:
        return ""
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}t={seconds}s"


# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.caption("Private semantic + transliteration-aware search over your Seerah transcripts")

youtube_map = load_youtube_map()

with st.sidebar:
    st.header("Index")
    transcripts_dir = st.text_input("Transcript folder", value=DEFAULT_TRANSCRIPTS_DIR)
    st.caption("Tip: after changing indexing logic, click Build / Rebuild Index once.")
    if st.button("Build / Rebuild Index"):
        try:
            file_count, chunk_count = build_index(transcripts_dir)
            st.success(f"Indexed {file_count} files into {chunk_count} timestamp windows.")
        except Exception as e:
            st.error(str(e))
    st.divider()
    st.caption("Optional YouTube mapping: create youtube_map.csv with columns file,youtube_url")

embeddings, meta = load_index()

if embeddings is None or meta is None:
    st.info("No index found yet. Put your .txt transcripts in a folder, then click 'Build / Rebuild Index' in the sidebar.")
else:
    query = st.text_input(
        "Ask a question or search a topic",
        placeholder="Where does Shaykh talk about mistranslation of awliya?"
    )
    top_k = st.slider("Results", min_value=3, max_value=20, value=TOP_K)

    if query:
        raw_results = hybrid_search(query, embeddings, meta, top_k=top_k)
        grouped = group_results(raw_results, max_groups=top_k)

        st.subheader("Top matches")

        for i, g in enumerate(grouped, start=1):
            youtube_url = youtube_map.get(g["file"], "")
            watch_link = build_youtube_timestamp_url(youtube_url, g["start_seconds"]) if youtube_url else ""

            lecture = pretty_lecture_name(g["file"])
            range_label = f"{g['start_timestamp']}–{g['end_timestamp']}" if g['start_timestamp'] != g['end_timestamp'] else g['start_timestamp']
            preview = " ".join(hit["text"] for hit in g["hits"][:2])[:300]

            reference_text = f"{lecture} | {g['start_timestamp']}"
            share_text = f"{reference_text}\n{watch_link}" if watch_link else reference_text

            st.markdown(f"### {i}. {lecture}")
            st.caption(f"{range_label} • {len(g['hits'])} hits • score={g['score']:.4f}")
            st.write(preview)

            c1, c2 = st.columns(2)
            with c1:
                if watch_link:
                    st.link_button("▶ Watch", watch_link, use_container_width=True)
                else:
                    st.button("▶ Watch", disabled=True, use_container_width=True, key=f"disabled_watch_{i}")
            with c2:
                st.code(share_text, language=None)

            with st.expander("Show discussion window"):
                for hit in g["hits"]:
                    st.write(f"**[{hit['timestamp']}]** {hit['text']}")

            st.divider()

        with st.expander("Query variants used"):
            st.write(expand_query(query))