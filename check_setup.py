"""Check that everything is installed and tell you the next step.   Run:  python check_setup.py"""
import importlib
import shutil
import sys

import config
import rag
import step1_transcribe


def report(ok, name, hint=""):
    print(f"[{'OK' if ok else 'MISSING'}] {name}" + ("" if ok or not hint else f"   -> {hint}"))
    return ok


def has_package(module):
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def main():
    rag.setup_console()
    print("Python", sys.version.split()[0], "-", sys.executable)        # which Python runs this (VS Code can pick another)
    report(sys.version_info >= (3, 9), "Python 3.9 or newer", "install a newer Python")

    print("\nPackages (pip install -r requirements.txt):")
    for module in ("numpy", "pandas", "sklearn", "requests", "matplotlib", "streamlit"):
        report(has_package(module), module, "pip install -r requirements.txt")

    print("\nOnly for step 1 (turning videos into text) - one Whisper engine is enough:")
    openai_ok, faster_ok = has_package("whisper"), has_package("faster_whisper")
    report(openai_ok or faster_ok, "a Whisper engine",
           'pip install faster-whisper "av<19"   (best without a GPU; or: pip install openai-whisper)')
    print(f"      faster-whisper: {'installed' if faster_ok else 'not installed'}   |   "
          f"openai-whisper: {'installed' if openai_ok else 'not installed'}")
    if faster_ok:                                   # it can be installed and still fail on every video (see AV_FIX)
        audio_error = step1_transcribe.faster_whisper_audio_error()
        hint = f"{step1_transcribe.audio_error_advice(audio_error)}   [{audio_error}]" if audio_error else ""
        report(audio_error is None, "faster-whisper can read audio files", hint)
    engine = step1_transcribe.choose_engine("auto")
    if engine == "openai-whisper":                  # only this engine needs ffmpeg to read the videos
        report(shutil.which("ffmpeg") is not None, "ffmpeg on PATH (openai-whisper needs it)",
               "install ffmpeg and restart the terminal, or use faster-whisper instead")
    if step1_transcribe.has_gpu():
        print("      GPU found: 'small.en' or 'medium.en' will be fast.")
    elif openai_ok or faster_ok:
        print("      No GPU found: keep 'small.en' and use faster-whisper (python step1_transcribe.py --engine faster-whisper).\n"
              "      Try ONE video first: the trial run shows how long the other 35 will take.")

    print("\nOllama (runs the embedding model and the LLM on your laptop):")
    running, models = rag.ollama_status()
    report(running, f"Ollama running at {config.OLLAMA_URL}", "start the Ollama app, or run: ollama serve")
    if running:
        report(rag.has_model(models, config.EMBED_MODEL), f"embedding model '{config.EMBED_MODEL}'",
               f"ollama pull {config.EMBED_MODEL}")
        report(rag.has_model(models, config.LLM_MODEL), f"LLM '{config.LLM_MODEL}'",
               f"ollama pull {config.LLM_MODEL}")

    catalog = rag.catalog()
    n_media = len(rag.find_media_files(config.VIDEO_DIR))
    n_transcripts = len(list(config.TRANSCRIPT_DIR.glob("*.json"))) if config.TRANSCRIPT_DIR.is_dir() else 0
    has_index = (config.INDEX_DIR / "embeddings.npy").exists()
    questions_file = config.EVAL_DIR / "questions.csv"
    report(bool(catalog), f"{config.CATALOG_FILE.name} ({len(catalog)} lectures, "
           f"{len(rag.catalog_videos())} chosen)", "it must be next to the scripts")
    print(f"\nYour files: {n_media} videos, {n_transcripts} transcripts, "
          f"index {'built' if has_index else 'not built'}.")

    print("\nNext step:")
    if n_media == 0 and n_transcripts == 0:
        print("  python step0_download_videos.py     (downloads the chosen lecture videos, about 400 MB)")
    elif n_transcripts < n_media:
        print("  python step1_transcribe.py --only 3.2-V5     (a quick trial on one video first)\n"
              "  python step1_transcribe.py                   (then all the others)")
    elif not has_index:
        print("  python step2_build_index.py")
    elif not questions_file.exists():
        print("  Write your questions in eval/questions.csv (see eval/HOW_TO_WRITE_QUESTIONS.md).")
    else:
        print("  python ask.py \"what is an ROC curve?\"   |   streamlit run app.py   |   python evaluate.py")


if __name__ == "__main__":
    main()
