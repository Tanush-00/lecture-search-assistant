"""Step 2: turn the transcripts into a searchable index (chunks + embeddings).

    python step2_build_index.py

Needs Ollama running with the embedding model:  ollama pull bge-m3
Result: index/chunks.csv, index/embeddings.npy, index/settings.json
"""
import argparse
from datetime import date

import config
import rag


def main(argv=None):
    parser = argparse.ArgumentParser(description="Turn the transcripts into a searchable index (chunks + embeddings). "
                                                 "Needs Ollama running with the embedding model.")
    parser.parse_args(argv)                  # there are no options: this gives --help and rejects typing mistakes
    rag.setup_console()
    transcripts = rag.load_transcripts(config.TRANSCRIPT_DIR)
    if not transcripts:
        print(f"No transcripts found in {config.TRANSCRIPT_DIR}. Run step1_transcribe.py first.")
        return 1

    settings = {
        "embed_model": config.EMBED_MODEL,
        "chunk_seconds": config.CHUNK_SECONDS,
        "overlap_seconds": config.OVERLAP_SECONDS,
        "n_videos": len(transcripts),
        "created": date.today().isoformat(),
    }
    print(f"Building the index from {len(transcripts)} videos "
          f"(chunks of ~{config.CHUNK_SECONDS} s, overlap {config.OVERLAP_SECONDS} s) ...")
    try:
        index = rag.build_index(transcripts, config.CHUNK_SECONDS, config.OVERLAP_SECONDS, settings=settings,
                                progress=rag.progress_printer())
    except rag.OllamaError as error:
        print(f"\nStopped: {error}")
        return 1
    if len(index.chunks) == 0:
        print("The transcripts are empty - nothing to index. Check the .txt files in the transcripts folder.")
        return 1

    index.settings["n_chunks"] = int(len(index.chunks))
    index.settings["dimensions"] = int(index.matrix.shape[1])
    rag.save_index(config.INDEX_DIR, index)

    lengths = index.chunks["end"] - index.chunks["start"]
    print(f"{len(index.chunks)} chunks, {lengths.mean():.0f} s on average "
          f"(shortest {lengths.min():.0f} s, longest {lengths.max():.0f} s), "
          f"{index.matrix.shape[1]} numbers per chunk.")
    print(f"Saved to {config.INDEX_DIR}")
    changed = index.settings.get("embed_changed", [])
    if changed:                              # never hide it: these chunks were embedded in a slightly changed form
        print(f"\nNOTE: Ollama answered NaN (not a number) for {len(changed)} chunk{'' if len(changed) == 1 else 's'}. "
              "Each of them was embedded in a slightly changed form; all other chunks are exactly as written:")
        for item in changed:
            print(f"    chunk {item['chunk_id']}: video {item['video']} at {item['start']}: {item['change']}")
        print("  (the list is also saved in index/settings.json under \"embed_changed\")")
    print("Next:  python ask.py \"what is an ROC curve?\"   or   streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
