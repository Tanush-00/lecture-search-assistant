"""Ask a question in the terminal.

    python ask.py "what is an ROC curve?"
    python ask.py "what is an ROC curve?" --no-llm       # only show where it is explained
    python ask.py "what is an ROC curve?" --method keywords
    python ask.py                                        # type questions one after another
"""
import argparse
import textwrap

import config
import rag


def show_hit(position, hit):
    title = rag.video_title(hit.video)
    print(f"  {position}. {title}   {rag.format_time(hit.start)}-{rag.format_time(hit.end)}   score {hit.score:.2f}")
    context = rag.video_context(hit.video)
    if context:
        print(f"     ({context})")
    print(textwrap.fill(hit.text, width=88, initial_indent="     ", subsequent_indent="     ",
                        max_lines=3, placeholder=" ..."))


def ask_once(question, index, args):
    result = rag.answer_question(question, index, method=args.method, top_k=args.top,
                                 min_score=args.min_score, use_llm=not args.no_llm)
    hits = result["hits"]
    best = float(hits["score"].iloc[0])
    print()
    if not result["found"]:
        print(f"Not found in the videos (best score {best:.2f} is below the cut-off {result['cutoff']:.2f}).")
        print("Closest passage, probably not relevant:")
        show_hit(1, next(hits.head(1).itertuples()))
        return
    print(f"Found in the videos (best score {best:.2f}, cut-off {result['cutoff']:.2f}).")
    if result["answer"]:
        print(f"\nAnswer ({config.LLM_MODEL}):")
        print(textwrap.fill(result["answer"], width=88, initial_indent="  ", subsequent_indent="  "))
    print("\nWhere it is explained:")
    for position, hit in enumerate(hits.itertuples(), start=1):
        show_hit(position, hit)


def main(argv=None):
    rag.setup_console()
    parser = argparse.ArgumentParser(description="Ask a question about the lecture videos.")
    parser.add_argument("question", nargs="*", help="your question (leave empty to type several)")
    parser.add_argument("--method", choices=rag.METHODS, default="embeddings",
                        help="embeddings = search by meaning (needs Ollama), keywords = TF-IDF")
    parser.add_argument("--top", type=int, default=config.TOP_K, help="how many passages to show")
    parser.add_argument("--min-score", type=float, default=None, help="override the 'not found' cut-off")
    parser.add_argument("--no-llm", action="store_true", help="do not write an answer, only show passages")
    args = parser.parse_args(argv)

    try:
        index = rag.load_index(config.INDEX_DIR)
        if args.method == "embeddings":
            rag.check_index_matches_config(index)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    questions = [" ".join(args.question)] if args.question else []
    try:
        if questions:
            ask_once(questions[0], index, args)
        else:
            print("Type a question (empty line to quit).")
            while True:
                question = input("\nQuestion: ").strip()
                if not question:
                    break
                ask_once(question, index, args)
    except rag.OllamaError as error:
        print(f"\n{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
