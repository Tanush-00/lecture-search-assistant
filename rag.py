"""The building blocks of the Lecture Search Assistant.

Every script (step0, step1, step2, ask, app, evaluate) imports its functions from this file.
Reading order: time helpers -> lecture catalog -> transcripts -> chunks -> Ollama -> index -> search -> answer.
"""
import csv
import functools
import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import config

MEDIA_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".m4a", ".mp3", ".wav")
AUDIO_EXTENSIONS = (".m4a", ".mp3", ".wav")
METHODS = ("embeddings", "keywords")

# The search index = the chunks (a table), their embeddings (a matrix) and the settings used to build them.
Index = namedtuple("Index", ["chunks", "matrix", "settings"])


class OllamaError(RuntimeError):
    """Ollama is not running, a model is missing, or it sent an unexpected reply."""


class OllamaNaNError(OllamaError):
    """Ollama calculated 'NaN' (not a number) for a text and so could not send the vector back."""


def setup_console():
    """Stop print() from crashing on Windows consoles that cannot show every character."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


# ----------------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------------
def format_time(seconds):
    """83.4 -> '1:23'  (and 3725 -> '1:02:05')."""
    total = int(max(0, seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_time(text):
    """'1:25' -> 85.0,  '85' -> 85.0,  '1:02:05' -> 3725.0."""
    original = str(text).strip()
    try:
        numbers = [float(part) for part in original.split(":")]
    except ValueError:
        numbers = []
    ok = (1 <= len(numbers) <= 3
          and all(np.isfinite(n) and n >= 0 for n in numbers)
          and all(n < 60 for n in numbers[1:]))
    if not ok:
        raise ValueError(f"Cannot read the time '{original}'. "
                         "Write it like 1:25 (minutes:seconds) or 85 (seconds).")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


# ----------------------------------------------------------------------------
# The lecture catalog (catalog.csv): which videos exist, their titles and where to download them
# ----------------------------------------------------------------------------
CATALOG_REQUIRED = ("video_id", "title", "archive_file")


def read_catalog(path):
    """catalog.csv -> {video_id: {column: text}} in file order. A missing file gives an empty catalog."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = [c for c in CATALOG_REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} needs the columns {list(CATALOG_REQUIRED)}. Missing: {missing}")
        rows = list(reader)
    catalog = {}
    for number, raw in enumerate(rows, start=2):                         # row 1 of the file is the header
        row = {key.strip(): (value or "").strip() for key, value in raw.items() if key}
        if not row["video_id"]:
            continue
        if row["video_id"] in catalog:
            raise ValueError(f"{path.name}, row {number}: the id '{row['video_id']}' appears twice.")
        row["selected"] = row.get("selected", "").lower() in ("1", "true", "yes", "y")
        catalog[row["video_id"]] = row
    return catalog


@functools.lru_cache(maxsize=8)
def _cached_catalog(path, modified):
    return read_catalog(path)


def catalog():
    """The catalog from config.CATALOG_FILE. It is read again whenever the file changes."""
    path = Path(config.CATALOG_FILE)
    return _cached_catalog(str(path), path.stat().st_mtime if path.exists() else None)


def catalog_videos(only_selected=True):
    """The catalog rows (in file order). By default only the videos marked selected = 1."""
    return [row for row in catalog().values() if row["selected"] or not only_selected]


def video_title(video_id):
    """'3.2-V5' -> 'Thresholding' (from catalog.csv).
    A name that is not in the catalog is tidied up: '02_overfitting_cross_validation' -> 'Overfitting cross validation'."""
    entry = catalog().get(video_id)
    if entry and entry.get("title"):
        return entry["title"]
    name = video_id.lstrip("0123456789").lstrip("_- ").replace("_", " ").replace("-", " ").strip()
    return name[:1].upper() + name[1:] if name else video_id


def video_context(video_id):
    """Where a lecture sits in the course, e.g. '3.2 Modeling the Expert: ... - video 5' ('' if it is not in the catalog)."""
    entry = catalog().get(video_id)
    if not entry:
        return ""
    parts = [entry.get("sequence", ""), f"video {entry['video_no']}" if entry.get("video_no") else ""]
    return " - ".join(part for part in parts if part)


# ----------------------------------------------------------------------------
# Transcripts
# ----------------------------------------------------------------------------
def find_media_files(folder):
    """All video / audio files in a folder, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)


def find_media_for(video_id, folder):
    """The recording that belongs to a video id (its file name without extension), or None."""
    for path in find_media_files(folder):
        if path.stem == video_id:
            return path
    return None


def clean_segments(raw_segments):
    """Keep only start, end and text of Whisper's segments and drop the empty ones."""
    segments = []
    for raw in raw_segments:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        start = round(float(raw["start"]), 2)
        end = max(round(float(raw["end"]), 2), start)
        segments.append({"start": start, "end": end, "text": text})
    segments.sort(key=lambda s: s["start"])
    return segments


def save_transcript(folder, video_id, segments, meta=None):
    """Write transcripts/<video>.json (for the code) and <video>.txt (for you to read)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = dict(meta or {})
    data["video"] = video_id
    data["segments"] = segments
    (folder / f"{video_id}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"[{format_time(s['start'])}] {s['text']}" for s in segments]
    (folder / f"{video_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_transcripts(folder):
    """Read every transcripts/*.json and return {video_id: [segments]}."""
    transcripts = {}
    for path in sorted(Path(folder).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        transcripts[data.get("video", path.stem)] = clean_segments(data["segments"])
    return transcripts


# ----------------------------------------------------------------------------
# Chunks: Whisper gives 2-5 second segments, which are too short to search. Join them.
# ----------------------------------------------------------------------------
def chunk_ranges(segments, chunk_seconds, overlap_seconds):
    """Decide which segments go into which chunk. Returns a list of (first, last + 1) positions."""
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be bigger than 0")
    if not 0 <= overlap_seconds < chunk_seconds:
        raise ValueError("overlap_seconds must be 0 or more, and smaller than chunk_seconds")
    ranges = []
    covered = 0                 # how many segments already sit inside an earlier chunk
    i = 0
    while i < len(segments):
        j = i + 1               # a chunk always holds at least one segment
        while j < len(segments) and segments[j]["end"] - segments[i]["start"] <= chunk_seconds:
            j += 1
        if j > covered:         # skip a window that would only repeat what is already covered
            ranges.append((i, j))
            covered = j
        if j >= len(segments):
            break
        # the next window starts where the last `overlap_seconds` of this one begin
        restart_time = segments[j - 1]["end"] - overlap_seconds
        k = i + 1
        while k < j and segments[k]["start"] < restart_time:
            k += 1
        i = k
    return ranges


def make_chunks(segments, chunk_seconds=None, overlap_seconds=None):
    """Join neighbouring segments into chunks of about `chunk_seconds`, overlapping a little."""
    chunk_seconds = config.CHUNK_SECONDS if chunk_seconds is None else chunk_seconds
    overlap_seconds = config.OVERLAP_SECONDS if overlap_seconds is None else overlap_seconds
    segments = [s for s in segments if s["text"].strip()]
    chunks = []
    for first, last in chunk_ranges(segments, chunk_seconds, overlap_seconds):
        piece = segments[first:last]
        chunks.append({
            "start": piece[0]["start"],
            "end": piece[-1]["end"],
            "text": " ".join(s["text"].strip() for s in piece),
        })
    return chunks


# ----------------------------------------------------------------------------
# Ollama (runs the embedding model and the LLM on your own laptop)
# ----------------------------------------------------------------------------
def _ollama_post(path, payload, url=None, timeout=300):
    url = (url or config.OLLAMA_URL).rstrip("/")
    try:
        reply = requests.post(f"{url}{path}", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"Cannot reach Ollama at {url}. "
                          "Start the Ollama app (or run `ollama serve`) and try again.") from None
    except requests.exceptions.Timeout:
        raise OllamaError("Ollama took too long to answer. Try again, or use a smaller model.") from None
    if reply.status_code == 404:
        model = payload.get("model")
        raise OllamaError(f"Ollama does not have the model '{model}'. Download it with:  ollama pull {model}")
    if reply.status_code != 200:
        message = f"Ollama returned error {reply.status_code}: {reply.text[:200]}"
        if reply.status_code == 500 and "NaN" in reply.text:       # "json: unsupported value: NaN", see embed_texts()
            raise OllamaNaNError(message)
        raise OllamaError(message)
    try:
        return reply.json()
    except ValueError:
        raise OllamaError("Ollama sent a reply that is not JSON.") from None


def progress_printer(what="chunks", steps=10, out=print):
    """A progress function for embed_texts(): it prints about `steps` lines such as 'embedded 130/430 chunks (30%)'."""
    last = {"bucket": -1}

    def show(done, total):
        bucket = done * steps // total if total else steps
        if bucket != last["bucket"]:
            last["bucket"] = bucket
            out(f"  embedded {done}/{total} {what} ({done * 100 // total if total else 100}%)")

    return show


NAN_HELP = ("This is a known problem of Ollama with the bge-m3 model (its 'flash attention' can calculate NaN for some "
            "texts). Update Ollama, or set the environment variable OLLAMA_FLASH_ATTENTION=false, restart Ollama "
            "and run this step again.")


def _count_words(number):
    return f"{number} word" + ("" if number == 1 else "s")


def _text_variants(text):
    """Slightly changed versions of a text. They are tried one after the other when Ollama cannot embed the text as it
    is. Returns a list of (what was changed, the changed text). Every version keeps almost all words of the text, so
    its meaning (and so its vector) stays nearly the same."""
    words = text.split()
    variants = [("added a full stop at the end", text.rstrip() + " .")]
    for fraction in (0.10, 0.25):
        cut = max(1, round(len(words) * fraction))
        if len(words) - cut >= 3:                                   # never leave fewer than 3 words
            variants.append((f"dropped the last {_count_words(cut)}", " ".join(words[:-cut])))
            variants.append((f"dropped the first {_count_words(cut)}", " ".join(words[cut:])))
    if len(words) >= 8:
        half = len(words) // 2
        variants.append((f"kept only the first {_count_words(half)}", " ".join(words[:half])))
        variants.append((f"kept only the last {_count_words(len(words) - half)}", " ".join(words[half:])))
    seen, unique = {text}, []
    for what, version in variants:
        if version not in seen:                                     # tiny texts can give the same version twice
            seen.add(version)
            unique.append((what, version))
    return unique


def _vectors_in(data):
    """The embeddings of an Ollama reply as a float32 table. A NaN or a missing number counts as the NaN problem."""
    if "embeddings" not in data:
        raise OllamaError("Ollama's reply has no 'embeddings'. Is your Ollama up to date?")
    try:
        table = np.array(data["embeddings"], dtype=np.float32)
    except TypeError:                                               # a null where a number should be
        raise OllamaNaNError("Ollama sent embeddings with missing numbers.") from None
    except ValueError:
        raise OllamaError("Ollama sent embeddings of different lengths.") from None
    if table.size and table.ndim != 2:
        raise OllamaError("Ollama's embeddings have an unexpected shape.")
    if table.size and not np.isfinite(table).all():
        raise OllamaNaNError("Ollama sent embeddings that contain NaN.")
    return table


def _embed_alone(text, position, total, model, url, problems, log):
    """Embed ONE text (used after a batch failed with NaN). If Ollama still answers NaN, slightly changed versions of
    the text are tried. Returns the vector. A text that had to be changed is announced with log() and added to problems."""
    for what, version in [(None, text)] + _text_variants(text):
        try:
            table = _vectors_in(_ollama_post("/api/embed", {"model": model, "input": [version]}, url=url))
        except OllamaNaNError:
            continue
        if len(table) != 1:
            raise OllamaError(f"Asked for 1 embedding but got {len(table)}.")
        if what:
            log(f"  Text {position + 1} of {total}: Ollama answered NaN for it. I {what} and then it worked.")
            if problems is not None:
                problems.append({"number": position, "change": what})
        return table[0]
    snippet = " ".join(text.split())[:80]
    raise OllamaError(f"Ollama cannot embed text {position + 1} of {total}: it answers NaN, even after small changes "
                      f"to the text. The text starts: \"{snippet}\". {NAN_HELP}")


def embed_texts(texts, model=None, url=None, batch_size=None, progress=None, problems=None, log=print):
    """Turn texts into vectors (numbers) with Ollama. Returns a (number of texts, dimensions) array.

    Every vector is scaled to length 1, so cosine similarity becomes a simple dot product.
    progress(done, total), if given, is called after every batch: embedding hundreds of chunks on a CPU takes minutes.

    Ollama sometimes cannot embed a text: it calculates NaN (not a number) and answers the WHOLE batch with error 500.
    Then the texts of that batch are sent one by one. If one text still fails, slightly changed versions of it are
    tried (_text_variants). A text that had to be changed is announced with log() and, if `problems` is a list,
    added to it as {"number": position in texts, "change": what was changed}. If nothing works, an OllamaError
    names the text, so nothing is skipped silently.
    """
    model = model or config.EMBED_MODEL
    batch_size = batch_size or config.EMBED_BATCH_SIZE
    texts = list(texts)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            found = list(_vectors_in(_ollama_post("/api/embed", {"model": model, "input": batch}, url=url)))
        except OllamaNaNError:
            log(f"  Ollama answered NaN for texts {start + 1}-{start + len(batch)}. Sending them one by one ...")
            found = [_embed_alone(text, start + offset, len(texts), model, url, problems, log)
                     for offset, text in enumerate(batch)]
        vectors.extend(found)
        if progress:
            progress(min(start + batch_size, len(texts)), len(texts))
    if len(vectors) != len(texts):
        raise OllamaError(f"Asked for {len(texts)} embeddings but got {len(vectors)}.")
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.array(vectors, dtype=np.float32)
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(lengths, 1e-12, None)


def ask_llm(prompt, model=None, url=None):
    """Send a prompt to the LLM (default llama3.2) and return its answer as text."""
    payload = {"model": model or config.LLM_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0}}      # temperature 0 = same question, same answer
    data = _ollama_post("/api/generate", payload, url=url)
    return str(data.get("response", "")).strip()


def ollama_status(url=None):
    """Return (is_running, names of the models you have downloaded)."""
    url = (url or config.OLLAMA_URL).rstrip("/")
    try:
        reply = requests.get(f"{url}/api/tags", timeout=3)
        reply.raise_for_status()
        return True, [m.get("name", "") for m in reply.json().get("models", [])]
    except (requests.exceptions.RequestException, ValueError):
        return False, []


def has_model(model_names, wanted):
    """True if Ollama has the model. 'bge-m3' matches 'bge-m3:latest'."""
    for name in model_names:
        if name == wanted or (":" not in wanted and name.split(":")[0] == wanted):
            return True
    return False


# ----------------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------------
def build_index(transcripts, chunk_seconds=None, overlap_seconds=None, embed_fn=None, settings=None, progress=None):
    """transcripts = {video: segments}  ->  Index(chunks table, embeddings matrix, settings).
    progress(done, total) is passed on to the embedding function, which should call it after every batch.
    If Ollama could only embed some chunks in a slightly changed form (see embed_texts), they are listed in
    settings["embed_changed"], so the change is never hidden."""
    rows = []
    for video_id in sorted(transcripts):
        for chunk in make_chunks(transcripts[video_id], chunk_seconds, overlap_seconds):
            rows.append({"video": video_id, **chunk})
    chunks = pd.DataFrame(rows, columns=["video", "start", "end", "text"])
    chunks.insert(0, "chunk_id", range(len(chunks)))
    settings = dict(settings or {})
    if len(chunks) == 0:
        matrix = np.zeros((0, 0), dtype=np.float32)
    else:
        embed = embed_fn or embed_texts
        options = {"progress": progress} if progress else {}
        problems = []
        if embed_fn is None:
            options["problems"] = problems                 # only the real embedder (embed_texts) reports these
        matrix = embed(chunks["text"].tolist(), **options)
        if problems:                                       # chunks that Ollama could only embed in a slightly changed form
            settings["embed_changed"] = [
                {"chunk_id": int(p["number"]), "video": str(chunks.at[p["number"], "video"]),
                 "start": format_time(chunks.at[p["number"], "start"]), "change": p["change"]} for p in problems]
    return Index(chunks, matrix, settings)


def save_index(folder, index):
    """Write index/chunks.csv, index/embeddings.npy and index/settings.json."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index.chunks.to_csv(folder / "chunks.csv", index=False, encoding="utf-8")
    np.save(folder / "embeddings.npy", np.asarray(index.matrix, dtype=np.float32))
    (folder / "settings.json").write_text(json.dumps(index.settings, indent=2), encoding="utf-8")


def load_index(folder=None):
    """Read an index written by save_index()."""
    folder = Path(folder or config.INDEX_DIR)
    chunk_file, matrix_file, settings_file = folder / "chunks.csv", folder / "embeddings.npy", folder / "settings.json"
    if not (chunk_file.exists() and matrix_file.exists()):
        raise FileNotFoundError(f"No index found in '{folder}'. "
                                "Run step1_transcribe.py and then step2_build_index.py first.")
    chunks = pd.read_csv(chunk_file, encoding="utf-8", dtype={"video": str, "text": str}, keep_default_na=False)
    matrix = np.load(matrix_file)
    settings = json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.exists() else {}
    if len(chunks) != len(matrix):
        raise ValueError("chunks.csv and embeddings.npy do not match. Run step2_build_index.py again.")
    return Index(chunks, matrix, settings)


def check_index_matches_config(index):
    """Questions must be embedded with the same model that embedded the chunks."""
    built_with = index.settings.get("embed_model")
    if built_with and built_with != config.EMBED_MODEL:
        raise ValueError(f"The index was built with '{built_with}' but config.py says '{config.EMBED_MODEL}'. "
                         "Change config.py back, or run step2_build_index.py again.")


# ----------------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------------
def keyword_scores(questions, chunk_texts):
    """TF-IDF keyword matching (the simple baseline): one score per question and chunk."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        chunk_vectors = vectorizer.fit_transform(chunk_texts)
    except ValueError:                      # the chunks contain no usable words at all
        return np.zeros((len(questions), len(chunk_texts)))
    return linear_kernel(vectorizer.transform(questions), chunk_vectors)


def score_matrix(questions, index, method="embeddings", embed_fn=None):
    """Score every question against every chunk. Result shape: (questions, chunks)."""
    questions = list(questions)
    if method == "embeddings":
        question_vectors = (embed_fn or embed_texts)(questions)
        if question_vectors.shape[1] != index.matrix.shape[1]:
            raise ValueError("The question and the chunks have different embedding sizes - the index was built "
                             "with another model. Run step2_build_index.py again.")
        return question_vectors @ index.matrix.T        # unit-length vectors: dot product = cosine similarity
    if method == "keywords":
        return keyword_scores(questions, index.chunks["text"].tolist())
    raise ValueError(f"Unknown search method '{method}'. Use one of: {', '.join(METHODS)}.")


def top_hits(chunks, scores, top_k):
    """The `top_k` chunks with the highest scores, best first."""
    order = np.argsort(-scores, kind="stable")[:top_k]
    hits = chunks.iloc[order].copy()
    hits["score"] = scores[order]
    return hits.reset_index(drop=True)


def search(question, index, method="embeddings", top_k=None, embed_fn=None):
    """Find the chunks that best match a question."""
    scores = score_matrix([question], index, method, embed_fn)[0]
    return top_hits(index.chunks, scores, top_k or config.TOP_K)


def minimum_score(method, override=None):
    """The 'not found' cut-off: below this score the assistant refuses to answer."""
    return config.MIN_SCORE[method] if override is None else override


# ----------------------------------------------------------------------------
# Answer
# ----------------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a study assistant. Answer the question using ONLY the video passages below.

Rules:
- If the passages do not contain the answer, reply exactly: The videos do not answer this.
- Otherwise answer in 2 to 4 simple sentences. Do not add facts that are not in the passages.
- The passages are speech-to-text, so a few words may be wrong. Use the meaning.

Passages:
{passages}

Question: {question}
Answer:"""


def make_prompt(question, hits):
    """Put the question and the found passages into one prompt for the LLM."""
    passages = "\n\n".join(
        f"[{row.video}, {format_time(row.start)}-{format_time(row.end)}]\n{row.text}"
        for row in hits.itertuples()
    )
    return PROMPT_TEMPLATE.format(passages=passages, question=question)


def answer_question(question, index, method="embeddings", top_k=None, min_score=None,
                    use_llm=True, embed_fn=None, llm_fn=None):
    """Search, decide 'found / not found', and (if found) let the LLM write a short answer.

    Returns {"found": bool, "hits": table of passages, "answer": text or None, "cutoff": number}.
    """
    hits = search(question, index, method, top_k, embed_fn)
    cutoff = minimum_score(method, min_score)
    found = len(hits) > 0 and float(hits["score"].iloc[0]) >= cutoff
    answer = None
    if found and use_llm:
        answer = (llm_fn or ask_llm)(make_prompt(question, hits))
    return {"found": found, "hits": hits, "answer": answer, "cutoff": cutoff}
