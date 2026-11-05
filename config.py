"""All settings in one place. Change a value here instead of editing the other files."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Folders ----
# The lecture videos are big (about 400 MB). If you do not want them inside a OneDrive folder, set the
# environment variable LECTURE_VIDEO_DIR to another folder (for example D:\lecture_videos) before running step 0.
VIDEO_DIR = Path(os.environ.get("LECTURE_VIDEO_DIR") or ROOT / "videos")   # step 0 downloads here, never uploaded
TRANSCRIPT_DIR = ROOT / "transcripts"    # step 1 writes one .json and one .txt per video
INDEX_DIR = ROOT / "index"               # step 2 writes chunks.csv, embeddings.npy, settings.json
EVAL_DIR = ROOT / "eval"                 # the labelled questions + evaluation results
IMAGE_DIR = ROOT / "images"              # charts made by evaluate.py
CATALOG_FILE = ROOT / "catalog.csv"      # the list of lecture videos: id, title, file name on archive.org

# ---- Step 0: where the lectures come from ----
# MIT OpenCourseWare 15.071 "The Analytics Edge" (Spring 2017), CC BY-NC-SA 4.0, kept by the Internet Archive.
ARCHIVE_URL = "https://archive.org"
ARCHIVE_ITEM = "MIT15.071S17"

# ---- Step 1: speech to text (Whisper) ----
WHISPER_MODEL = "small.en"   # tiny.en, base.en, small.en, medium.en (English only). Bigger = more accurate but slower.
WHISPER_LANGUAGE = "en"      # the language spoken in the videos
WHISPER_TASK = "transcribe"  # "transcribe" = write down what is said, "translate" = speech -> English text
WHISPER_PROMPT = ""          # optional: a few topic words (e.g. "logistic regression, ROC curve") to help spelling

# ---- Step 2: chunks and embeddings ----
CHUNK_SECONDS = 30           # about how long one searchable piece (chunk) is
OVERLAP_SECONDS = 5          # neighbouring chunks share this much, so an answer is not cut in half
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
EMBED_BATCH_SIZE = 32        # how many chunks are sent to Ollama in one request

# ---- Step 3: search and answer ----
TOP_K = 4                    # how many passages are shown / given to the LLM
LLM_MODEL = "llama3.2"
# If the best passage scores below this number, the assistant says "not found in the videos".
# These are only starting values: run `python evaluate.py` to find the right ones for your videos.
MIN_SCORE = {"embeddings": 0.40, "keywords": 0.10}
