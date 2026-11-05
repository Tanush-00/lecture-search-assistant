"""Step 1: turn every lecture video in videos/ into a transcript with Whisper.

    python step1_transcribe.py --only 3.2-V5            # try ONE video first, then read transcripts/3.2-V5.txt
    python step1_transcribe.py                          # every video that has no transcript yet
    python step1_transcribe.py --engine faster-whisper  # up to 4x faster on a laptop without a GPU
    python step1_transcribe.py --model base.en          # smaller and faster, a little less accurate
    python step1_transcribe.py --force                  # do everything again

A video is skipped only if its transcript was made with the same Whisper model, language and task as now,
so a quick test with --model base.en never ends up mixed with the real run. You can stop with Ctrl+C at any
time: finished transcripts are kept, and running the same command again continues where it stopped.

Whisper reads the audio from the video by itself, so no separate mp3 step is needed.
Result: transcripts/<video>.json (used by the code) and transcripts/<video>.txt (for you to read).
"""
import argparse
import importlib
import json
import os
import re
import shutil
import tempfile
import time
import wave
from pathlib import Path

import config
import rag

# On Windows, Hugging Face (where the Whisper model is downloaded from) often cannot create the "symlinks" it would
# like to use, so it stores plain copies instead and prints a long warning at every start. Nothing is wrong, so the
# warning is switched off. This must happen before faster-whisper is imported, which is why it sits up here.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ENGINES = ("auto", "openai-whisper", "faster-whisper")

# faster-whisper 1.2.1 opens the audio with the "av" package and hands it an option (metadata_errors) that av 19
# removed. With av 19 or newer, every video fails with "open() got an unexpected keyword argument 'metadata_errors'".
# av 18.1 and older work. This is the command that fixes it (see faster_whisper_audio_error below).
AV_FIX = 'pip install "av<19"'


def show_progress(done, total):
    """Rewrite one line with how far Whisper has come in this video, so that a long video does not look stuck."""
    line = f"    transcribing ... {rag.format_time(done)}" + (f" of {rag.format_time(total)}" if total else "")
    print("\r" + line, end="", flush=True)


def clear_progress():
    """Wipe the progress line, so that the line printed after it starts on a clean row."""
    print("\r" + " " * 50 + "\r", end="", flush=True)


class FasterWhisperModel:
    """Lets faster-whisper be used exactly like openai-whisper: model.transcribe(path, ...) -> {"segments": [...]}."""

    def __init__(self, model_name):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(self, path, language=None, task="transcribe", initial_prompt=None, **_ignored):
        segments, info = self.model.transcribe(path, language=language, task=task, initial_prompt=initial_prompt)
        total = getattr(info, "duration", 0) or 0       # length of the video, known before the first segment
        result = []
        for segment in segments:                        # the speech is turned into text while this loop runs
            result.append({"start": segment.start, "end": segment.end, "text": segment.text})
            show_progress(segment.end, total)
        clear_progress()
        return {"segments": result}


def transcribe_one(model, media_path, language, task, use_gpu, prompt=""):
    """Run Whisper on one file and return its cleaned segments: [{start, end, text}, ...]."""
    result = model.transcribe(
        str(media_path),
        language=language,
        task=task,
        word_timestamps=False,
        fp16=use_gpu,                      # half precision only works on a GPU
        initial_prompt=prompt or None,
    )
    return rag.clean_segments(result["segments"])


def pick_files(files, only=None):
    """One file per video name (if 3.2-V5.mp4 and 3.2-V5.mp3 both exist, the first wins)."""
    chosen = {}
    for path in files:
        chosen.setdefault(path.stem, path)
    if only:
        by_lower = {name.lower(): name for name in chosen}                  # 3.2-v5 and 3.2-V5 are the same video
        missing = [name for name in only if name.lower() not in by_lower]
        if missing:
            raise ValueError(f"No video called {missing} in the videos folder. "
                             f"Available: {sorted(chosen)}")
        chosen = {by_lower[name.lower()]: chosen[by_lower[name.lower()]] for name in only}
    return list(chosen.values())


def needs_transcript(video_id, model_name, force=False):
    """True if the video has no transcript yet, or its transcript was made with other settings."""
    path = config.TRANSCRIPT_DIR / f"{video_id}.json"
    if force or not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    made_with = (data.get("whisper_model"), data.get("language"), data.get("task"))
    return made_with != (model_name, config.WHISPER_LANGUAGE, config.WHISPER_TASK)


def can_import(module):
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def has_gpu():
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def faster_whisper_audio_error():
    """None if faster-whisper can read an audio file on this computer, otherwise the error text.

    It decodes 0.1 s of silence. That takes a moment and finds a broken install (for example av 19, see AV_FIX)
    at the start, not after the 500 MB model is loaded and the first video has been running for minutes."""
    try:
        from faster_whisper.audio import decode_audio
    except ImportError:
        return None                          # not installed (load_model says so), so there is nothing to test
    folder = tempfile.mkdtemp()
    try:
        path = Path(folder) / "silence.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x00\x00" * 1600)
        try:
            decode_audio(str(path), sampling_rate=16000)
        except Exception as error:           # whatever the reason, Whisper would fail on the first video too
            return f"{type(error).__name__}: {error}"
        return None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def audio_error_advice(error):
    """One line that says what to do about an error text from faster_whisper_audio_error()."""
    if "metadata_errors" in error:
        return f'The "av" package is too new for faster-whisper. Run:  {AV_FIX}'
    return "Try:  pip install --upgrade faster-whisper   (or use  --engine openai-whisper, which needs ffmpeg)"


def choose_engine(requested):
    """'auto' = openai-whisper when there is a GPU; on a CPU, faster-whisper if it is installed (about 4x faster)."""
    if requested != "auto":
        return requested
    if has_gpu() or not can_import("faster_whisper"):
        return "openai-whisper"
    return "faster-whisper"


def load_model(engine, model_name):
    """Load Whisper. Returns (model, use_gpu). Raises ImportError with advice if the engine is not installed
    or cannot read audio files here."""
    if engine == "faster-whisper":
        if not can_import("faster_whisper"):
            raise ImportError('faster-whisper is not installed. Run:  pip install faster-whisper "av<19"')
        error = faster_whisper_audio_error()           # before the big model is loaded
        if error:
            raise ImportError(f"faster-whisper cannot read audio files on this computer ({error}).\n"
                              + audio_error_advice(error))
        print(f"Loading faster-whisper '{model_name}' on the CPU ...")
        return FasterWhisperModel(model_name), False
    try:
        import torch
        import whisper
    except ImportError:
        raise ImportError("Whisper is not installed. Run:  pip install openai-whisper\n"
                          "(It also needs ffmpeg. Run `python check_setup.py` to see what is missing.)") from None
    use_gpu = torch.cuda.is_available()
    print(f"Loading Whisper '{model_name}' on {'the GPU' if use_gpu else 'the CPU (slow - be patient)'} ...")
    if not use_gpu and not can_import("faster_whisper"):
        print('Tip: without a GPU, `pip install faster-whisper "av<19"` and then `--engine faster-whisper` is up to 4x faster.')
    return whisper.load_model(model_name), use_gpu


def title_warning(video_id, segments):
    """A warning text if the transcript has none of the words that catalog.csv expects for this video, else None.
    It catches a file that does not belong to its title. It is only a hint, never an error."""
    entry = rag.catalog().get(video_id)
    pattern = entry.get("check_regex", "") if entry else ""
    if not pattern:
        return None
    try:
        if re.search(pattern, " ".join(s["text"] for s in segments), re.IGNORECASE):
            return None
    except re.error:
        return None
    return (f"WARNING: the transcript of {video_id} ('{entry['title']}') has none of the words: {pattern}. "
            "Open transcripts/%s.txt and check that it really is that lecture." % video_id)


def main(argv=None):
    rag.setup_console()
    parser = argparse.ArgumentParser(description="Turn the lecture videos into transcripts with Whisper.")
    parser.add_argument("--only", nargs="+", metavar="ID", help="video ids (like 3.2-V5, without .mp4) to process")
    parser.add_argument("--model", default=config.WHISPER_MODEL, help="Whisper model size (default from config.py)")
    parser.add_argument("--engine", choices=ENGINES, default="auto",
                        help="auto = openai-whisper on a GPU, faster-whisper on a CPU if installed")
    parser.add_argument("--force", action="store_true", help="redo videos that already have a transcript")
    args = parser.parse_args(argv)

    files = rag.find_media_files(config.VIDEO_DIR)
    if not files:
        print(f"No videos found in {config.VIDEO_DIR}")
        print("Download them first:  python step0_download_videos.py")
        return 1
    try:
        files = pick_files(files, args.only)
    except ValueError as error:
        print(error)
        return 1
    todo = [p for p in files if needs_transcript(p.stem, args.model, args.force)]
    if not todo:
        print("Every video already has a transcript made with these settings. Use --force to redo them.")
        return 0

    engine = choose_engine(args.engine)
    if engine == "openai-whisper" and shutil.which("ffmpeg") is None:
        print("ffmpeg was not found. openai-whisper needs it to read the videos.")
        print("Install ffmpeg (see README), restart the terminal, or use:  --engine faster-whisper")
        return 1
    try:
        model, use_gpu = load_model(engine, args.model)
    except ImportError as error:
        print(error)
        return 1

    started_all = time.time()
    try:
        for number, path in enumerate(todo, start=1):
            started = time.time()
            print(f"[{number}/{len(todo)}] {path.name}  {rag.video_title(path.stem)} ...", flush=True)
            segments = transcribe_one(model, path, config.WHISPER_LANGUAGE, config.WHISPER_TASK,
                                      use_gpu, config.WHISPER_PROMPT)
            duration = segments[-1]["end"] if segments else 0
            meta = {
                "source_file": path.name,
                "whisper_model": args.model,
                "engine": engine,
                "language": config.WHISPER_LANGUAGE,
                "task": config.WHISPER_TASK,
                "duration_seconds": duration,
            }
            rag.save_transcript(config.TRANSCRIPT_DIR, path.stem, segments, meta)
            took = max(time.time() - started, 0.001)
            print(f"    done in {took:.0f} s: {len(segments)} segments, {rag.format_time(duration)} of speech "
                  f"({duration / took:.1f}x real time)")
            warning = title_warning(path.stem, segments)
            if warning:
                print("    " + warning)
    except KeyboardInterrupt:
        print("\nStopped. The finished transcripts are saved. Run the same command again to continue.")
        return 1
    print(f"\nAll done in {rag.format_time(time.time() - started_all)}.")
    print("Now open some .txt files in the transcripts folder and read them.")
    print("Do not fix them by hand yet - the evaluation should measure the real pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
