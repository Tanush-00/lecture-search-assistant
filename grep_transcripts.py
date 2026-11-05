"""Look into the transcripts. Useful when you write or check eval/questions.csv.

    python grep_transcripts.py "k-means|clustering"             # where do the lectures say these words?
    python grep_transcripts.py threshold --video 3.2-V5         # only in one video
    python grep_transcripts.py --show 3.2-V5 0:30 1:10          # what is said between 0:30 and 1:10?

A question that the lectures cannot answer must not find its key words here. For a question with a marked answer,
`--show` lets you check that the time range really contains the answer, and nothing else in the lectures answers it too.
Upper and lower case do not matter. The text is a regular expression, so `a|b` means "a or b".
"""
import argparse
import re

import config
import rag


def segment_matches(regex, segments, i):
    """True if segment i contains the pattern, also when the words continue in the next segment."""
    text = segments[i]["text"]
    if regex.search(text):
        return True
    if i + 1 < len(segments):
        joined = text + " " + segments[i + 1]["text"]
        return any(m.start() < len(text) < m.end() for m in regex.finditer(joined))
    return False


def find_lines(transcripts, pattern, video=None):
    """Every transcript line that contains the pattern: [(video, start seconds, text), ...]."""
    regex = re.compile(pattern, re.IGNORECASE)
    found = []
    for name in sorted(transcripts):
        if video is not None and name != video:
            continue
        segments = transcripts[name]
        for i, segment in enumerate(segments):
            if segment_matches(regex, segments, i):
                found.append((name, segment["start"], segment["text"]))
    return found


def lines_between(segments, start, end):
    """The segments that share at least a moment with start..end seconds."""
    return [s for s in segments if s["start"] < end and s["end"] > start]


def resolve_video(transcripts, name):
    """3.2-v5 -> 3.2-V5. Raises ValueError with the known names if there is no such transcript."""
    by_lower = {known.lower(): known for known in transcripts}
    if name.lower() not in by_lower:
        raise ValueError(f"No transcript called '{name}'. Known: {sorted(transcripts)}")
    return by_lower[name.lower()]


def main(argv=None):
    rag.setup_console()
    parser = argparse.ArgumentParser(description="Look into the transcripts.")
    parser.add_argument("pattern", nargs="?", help="the word or pattern to look for")
    parser.add_argument("--video", metavar="ID", help="only look in this video")
    parser.add_argument("--show", nargs=3, metavar=("ID", "START", "END"),
                        help="print what is said in this video between START and END (like 0:30 1:10)")
    args = parser.parse_args(argv)
    if not args.pattern and not args.show:
        parser.print_usage()
        print("Give a word to look for, or --show ID START END.")
        return 1

    transcripts = rag.load_transcripts(config.TRANSCRIPT_DIR)
    if not transcripts:
        print(f"No transcripts found in {config.TRANSCRIPT_DIR}. Run step1_transcribe.py first.")
        return 1
    try:
        if args.show:
            name = resolve_video(transcripts, args.show[0])
            start, end = rag.parse_time(args.show[1]), rag.parse_time(args.show[2])
            print(f"{name}  {rag.video_title(name)}   {rag.format_time(start)} - {rag.format_time(end)}")
            lines = lines_between(transcripts[name], start, end)
            for segment in lines:
                print(f"  [{rag.format_time(segment['start'])}] {segment['text']}")
            if not lines:
                print("  (nothing is said in this range)")
            return 0
        video = resolve_video(transcripts, args.video) if args.video else None
        found = find_lines(transcripts, args.pattern, video)
    except (ValueError, re.error) as error:
        print(error)
        return 1

    for name, start, text in found:
        print(f"{name:<8} [{rag.format_time(start)}] {text}")
    videos = len({name for name, _, _ in found})
    print(f"\n{len(found)} line(s) in {videos} video(s) say '{args.pattern}'." if found
          else f"No line says '{args.pattern}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
