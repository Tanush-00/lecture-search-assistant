"""Measure how good the search is.

    python evaluate.py                   # prints the results, saves charts and eval/results.md
    python evaluate.py --write-readme    # also pastes the results tables into README.md

It needs: transcripts (step 1), your questions in eval/questions.csv, and Ollama running (for embeddings).
A retrieved chunk counts as correct if it comes from the right video AND its time range overlaps the
time range where the answer is said (answer_start .. answer_end in questions.csv).

Every number comes with a yardstick, so that nobody has to take it on trust:
  - random guessing: the exact chance that k randomly picked chunks contain the answer (calculated, not simulated);
  - a 95% confidence interval (Wilson): with few questions, one lucky or unlucky question moves a percentage a lot.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # draw charts into files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

import config
import rag

KS = (1, 3, 5)                                   # Recall@1, Recall@3, Recall@5
SHOWN_ORDER = ("keywords", "embeddings")         # the simple method first, then the one that should beat it
SWEEP = [(15, 3), (30, 5), (60, 10), (90, 15)]   # (chunk seconds, overlap seconds) to compare
QUESTION_COLUMNS = ["question", "video", "answer_start", "answer_end"]
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

BLUE, ORANGE, GREY, LIGHT = "#2a78d6", "#eb6834", "#b5b3ad", "#d9d7cf"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def method_name(method):
    if method == "embeddings":
        return f"Meaning ({config.EMBED_MODEL} embeddings)"
    return "Keywords (TF-IDF)"


# ----------------------------------------------------------------------------
# Questions
# ----------------------------------------------------------------------------
def read_csv_file(path):
    """Read a CSV as text. Excel may save it with a BOM or an old Windows encoding."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp1252")


def load_questions(path, durations=None):
    """Read eval/questions.csv.

    A row without a video is a question the videos cannot answer (it tests the 'not found' cut-off).
    durations = {video: length in seconds}, used to catch typos in the video names and times.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    table = read_csv_file(path)
    table.columns = [c.strip().lower() for c in table.columns]
    missing = [c for c in QUESTION_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} needs the columns {QUESTION_COLUMNS}. Missing: {missing}")

    known = {name.lower(): name for name in durations} if durations is not None else {}
    rows = []
    for number, (_, row) in enumerate(table.iterrows(), start=2):       # row 1 of the file is the header
        question = row["question"].strip()
        video = row["video"].strip()
        start_text, end_text = row["answer_start"].strip(), row["answer_end"].strip()
        if not question:
            continue
        where = f"Row {number} ('{question}')"
        if Path(video).suffix.lower() in rag.MEDIA_EXTENSIONS:          # allow "3.2-V5.mp4" as well as "3.2-V5"
            video = Path(video).stem
        if not video:
            if start_text or end_text:
                raise ValueError(f"{where}: times are given but the video name is empty.")
            rows.append({"question": question, "video": "", "start": np.nan, "end": np.nan, "in_scope": False})
            continue
        video = known.get(video.lower(), video)                         # 3.2-v5 means 3.2-V5
        if durations is not None and video not in durations:
            raise ValueError(f"{where}: there is no video called '{video}'. Known videos: {sorted(durations)}")
        if not start_text or not end_text:
            raise ValueError(f"{where}: fill in answer_start and answer_end, "
                             "or leave the video empty if the videos cannot answer it.")
        try:
            start, end = rag.parse_time(start_text), rag.parse_time(end_text)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        if end <= start:
            raise ValueError(f"{where}: answer_end must come after answer_start.")
        if durations is not None and start >= durations[video] + 2:
            raise ValueError(f"{where}: answer_start {rag.format_time(start)} is after the end of '{video}' "
                             f"({rag.format_time(durations[video])} long). "
                             "If you edited the file in Excel, it may have changed your times.")
        rows.append({"question": question, "video": video, "start": start, "end": end, "in_scope": True})
    return pd.DataFrame(rows, columns=["question", "video", "start", "end", "in_scope"])


# ----------------------------------------------------------------------------
# Retrieval quality: does the right passage show up in the top results?
# ----------------------------------------------------------------------------
def overlaps(start_a, end_a, start_b, end_b):
    """True if two time ranges share at least a moment (touching at one point does not count).
    Works for single numbers and for whole numpy arrays."""
    return (start_a < end_b) & (end_a > start_b)


def evaluate_method(questions, index, method, embed_fn=None, max_k=max(KS)):
    """Search every question. Returns, per question: the rank of the first correct chunk (None = not among the
    top max_k), the score of the best chunk, and how many chunks of the index are correct at all (n_correct)."""
    scores = rag.score_matrix(questions["question"].tolist(), index, method, embed_fn)
    chunk_video = index.chunks["video"].to_numpy()
    chunk_start = index.chunks["start"].to_numpy(dtype=float)
    chunk_end = index.chunks["end"].to_numpy(dtype=float)
    ranks, top_scores, orders, n_correct = [], [], [], []
    for row, q in enumerate(questions.itertuples(index=False)):
        order = np.argsort(-scores[row], kind="stable")
        top_scores.append(float(scores[row][order[0]]))
        rank, correct_chunks = None, 0
        if q.in_scope:
            is_correct = (chunk_video == q.video) & overlaps(chunk_start, chunk_end, q.start, q.end)
            correct_chunks = int(is_correct.sum())
            hits = np.flatnonzero(is_correct[order[:max_k]])
            if len(hits):
                rank = int(hits[0]) + 1
        ranks.append(rank)
        n_correct.append(correct_chunks)
        orders.append(order)
    return {"ranks": ranks, "top_scores": np.array(top_scores), "scores": scores, "orders": orders,
            "n_correct": n_correct}


def recall_counts(ranks, answerable, k):
    """(answerable questions whose right passage is in the top k, number of answerable questions)."""
    wanted = [rank for rank, flag in zip(ranks, answerable) if flag]
    return sum(rank is not None and rank <= k for rank in wanted), len(wanted)


def recall_at_k(ranks, answerable, k):
    """Share of the answerable questions whose right passage is among the top k results."""
    hits, total = recall_counts(ranks, answerable, k)
    return hits / total if total else float("nan")


def wilson_interval(successes, total, z=1.96):
    """95% confidence interval for a share (`successes` out of `total`). It stays sensible for small totals,
    where the textbook 'p +/- 1.96 * sqrt(p * (1 - p) / n)' can go below 0% or above 100%."""
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def random_recall(n_chunks, n_correct, k):
    """The chance that k chunks picked at random (all different) include at least one of the n_correct right chunks.
    = 1 - (chance that all k picks are wrong) = 1 - (N-c)/N * (N-c-1)/(N-1) * ... for k factors."""
    if n_chunks <= 0 or n_correct <= 0:
        return 0.0
    k = min(k, n_chunks)
    all_wrong = 1.0
    for i in range(k):
        all_wrong *= max(n_chunks - n_correct - i, 0) / (n_chunks - i)
    return 1.0 - all_wrong


def baseline_recall(n_correct, answerable, n_chunks, k):
    """Random guessing, averaged over the answerable questions (each has its own number of right chunks)."""
    chances = [random_recall(n_chunks, count, k) for count, flag in zip(n_correct, answerable) if flag]
    return float(np.mean(chances)) if chances else float("nan")


def failure_table(questions, index, run, k=3):
    """Answerable questions whose right passage is NOT in the top k - the ones to study."""
    rows = []
    for row, q in enumerate(questions.itertuples(index=False)):
        rank = run["ranks"][row]
        if not q.in_scope or (rank is not None and rank <= k):
            continue
        found = []
        for chunk in run["orders"][row][:k]:
            c = index.chunks.iloc[chunk]
            found.append(f"{c['video']} {rag.video_title(c['video'])} {rag.format_time(c['start'])}-"
                         f"{rag.format_time(c['end'])} ({run['scores'][row][chunk]:.2f})")
        rows.append({
            "question": q.question,
            "expected": f"{q.video} {rag.video_title(q.video)} {rag.format_time(q.start)}-{rag.format_time(q.end)}",
            "first_correct_rank": rank if rank is not None else f"not in top {max(KS)}",
            "top_results": " | ".join(found),
        })
    return pd.DataFrame(rows, columns=["question", "expected", "first_correct_rank", "top_results"])


# ----------------------------------------------------------------------------
# The "not found" cut-off
# ----------------------------------------------------------------------------
def split_rates(top_scores, answerable, threshold):
    """At this cut-off: share of answerable questions that get an answer, share of the others that are refused."""
    answered = top_scores >= threshold
    return float(answered[answerable].mean()), float((~answered)[~answerable].mean())


def best_threshold(top_scores, answerable):
    """The cut-off that best balances 'answer what can be answered' and 'refuse the rest'."""
    unique = np.unique(top_scores)
    candidates = np.concatenate(([unique[0] - 1e-6], (unique[:-1] + unique[1:]) / 2, [unique[-1] + 1e-6]))
    balanced = np.array([sum(split_rates(top_scores, answerable, t)) / 2 for t in candidates])
    best = np.flatnonzero(np.isclose(balanced, balanced.max()))
    runs = np.split(best, np.flatnonzero(np.diff(best) != 1) + 1)       # groups of neighbouring candidates
    longest = max(runs, key=len)
    return float(candidates[longest[len(longest) // 2]])               # the middle of the widest plateau


def cross_validated_balance(top_scores, answerable, seed=42, n_repeats=10):
    """An honest estimate: pick the cut-off on some questions and test it on questions it has not seen.
    Returns (mean, standard deviation) of the balanced accuracy, or None if there are too few questions."""
    labels = answerable.astype(int)
    n_splits = int(min(5, labels.sum(), (1 - labels).sum()))
    if n_splits < 2:
        return None
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    results = []
    for train, test in splitter.split(top_scores.reshape(-1, 1), labels):
        threshold = best_threshold(top_scores[train], answerable[train])
        results.append(sum(split_rates(top_scores[test], answerable[test], threshold)) / 2)
    return float(np.mean(results)), float(np.std(results))


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def style_charts():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })


def label_bars(ax, bars, values, tops=None):
    """Write the value above each bar (above the error bar if there is one)."""
    tops = values if tops is None else tops
    for bar, value, top in zip(bars, values, tops):
        ax.text(bar.get_x() + bar.get_width() / 2, top + 1.5, f"{value:.0f}%",
                ha="center", color=INK, fontsize=9)


def plot_methods(methods, intervals, baseline, path, subtitle=""):
    """Bars: random guessing, keyword search, search by meaning - for Recall@1/3/5. Black whiskers = 95% interval."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x, width = np.arange(len(KS)), 0.26
    values = [baseline[k] * 100 for k in KS]
    bars = ax.bar(x - width, values, width, color=LIGHT, hatch="///", edgecolor="white", label="Random guessing")
    label_bars(ax, bars, values)
    for offset, method, color in ((0, "keywords", GREY), (width, "embeddings", BLUE)):
        values = np.array([methods[method][k] * 100 for k in KS])
        low = np.array([intervals[method][k][0] * 100 for k in KS])
        high = np.array([intervals[method][k][1] * 100 for k in KS])
        bars = ax.bar(x + offset, values, width, color=color, label=method_name(method))
        ax.errorbar(x + offset, values, yerr=[values - low, high - values], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=3, zorder=4)
        label_bars(ax, bars, values, tops=high)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top {k}" for k in KS])
    ax.set_ylim(0, 115)
    ax.set_ylabel("Questions with a correct passage (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Is the right passage among the top results?", loc="left", fontsize=13, pad=34)
    fig.text(0.012, 0.01, f"{subtitle} Black lines: 95% confidence interval.".strip(), color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_chunk_sizes(sweep, path, subtitle=""):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x, width = np.arange(len(sweep)), 0.26
    values = [row["random@3"] * 100 for row in sweep]
    bars = ax.bar(x - width, values, width, color=LIGHT, hatch="///", edgecolor="white", label="Random guessing, top 3")
    label_bars(ax, bars, values)
    for offset, k, color in ((0, 1, ORANGE), (width, 3, BLUE)):
        values = [row[f"recall@{k}"] * 100 for row in sweep]
        bars = ax.bar(x + offset, values, width, color=color, label=f"Right passage in top {k}")
        label_bars(ax, bars, values)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['chunk_seconds']} s\n({row['n_chunks']} chunks)" for row in sweep])
    ax.set_xlabel("Chunk length")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Questions with a correct passage (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Does the chunk length change the results?", loc="left", fontsize=13, pad=34)
    fig.text(0.012, 0.01, f"Search by meaning. {subtitle}".strip(), color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(top_scores, answerable, threshold, method, path):
    rng = np.random.default_rng(0)              # fixed, so the chart looks the same every time
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for height, flag, color in ((1, True, BLUE), (0, False, ORANGE)):
        values = top_scores[answerable == flag]
        ax.scatter(values, height + rng.uniform(-0.12, 0.12, len(values)), s=42, color=color,
                   edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(threshold, 1.42, f" cut-off {threshold:.2f}", color=INK, fontsize=10, va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not answerable\n(not in the lectures)", "Answerable\n(in the lectures)"])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("Score of the best matching passage")
    ax.grid(axis="y", visible=False)
    ax.set_title("Can a score tell the two kinds of question apart?", loc="left", fontsize=13)
    fig.text(0.012, 0.01, f"{method_name(method)}. One dot = one question.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# The full evaluation
# ----------------------------------------------------------------------------
def percent(value):
    return "n/a" if value != value else f"{value * 100:.0f}%"


def percent_with_range(value, interval):
    """0.78 and (0.65, 0.88) -> '78% (65-88)'."""
    if value != value:
        return "n/a"
    return f"{value * 100:.0f}% ({interval[0] * 100:.0f}-{interval[1] * 100:.0f})"


def results_markdown(summary):
    """The Results section for the README, with the real numbers filled in."""
    lines = [
        f"**Test questions:** {summary['n_answerable']} questions that the {summary['n_videos']} lecture videos "
        f"answer ({summary['minutes']:.0f} minutes of speech; `eval/questions.csv` says where each answer is said) "
        f"and {summary['n_unanswerable']} questions that they cannot answer (unrelated, or about topics the "
        f"selected lectures do not cover). Chunks are about {summary['chunk_seconds']} s long with "
        f"{summary['overlap_seconds']} s overlap ({summary['n_chunks']} chunks). A result is correct if it comes "
        "from the right video and overlaps the time where the answer is said.",
        "",
        "#### Keyword search vs search by meaning",
        "",
        "| Search method | Recall@1 | Recall@3 | Recall@5 |",
        "|---|---|---|---|",
        "| Random guessing (calculated) | " + " | ".join(percent(summary["baseline"][k]) for k in KS) + " |",
    ]
    for method in SHOWN_ORDER:
        values, ranges = summary["methods"][method], summary["intervals"][method]
        lines.append(f"| {method_name(method)} | "
                     + " | ".join(percent_with_range(values[k], ranges[k]) for k in KS) + " |")
    lines += [
        "",
        "Recall@k = share of the answerable questions whose right passage is among the top k results. "
        "The range in brackets is the 95% confidence interval (Wilson). "
        "'Random guessing' is the exact chance that k chunks picked at random contain the answer, "
        "so it shows how much of a result could be luck.",
    ]
    if summary["ci_points"] == summary["ci_points"]:
        lines.append(f"With {summary['n_answerable']} questions the interval is about +/-{summary['ci_points']:.0f} "
                     "points around each percentage, so a gap smaller than that between two rows is not a reliable "
                     "difference.")
    lines += [
        "",
        "![Recall of random guessing, keyword search and search by meaning](images/retrieval_methods.png)",
        "",
        "#### Chunk length (search by meaning)",
        "",
        "| Chunk length | Chunks | Recall@1 | Recall@3 | Recall@5 | Random guessing, Recall@3 |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["sweep"]:
        lines.append(f"| {row['chunk_seconds']} s (overlap {row['overlap_seconds']} s) | {row['n_chunks']} | "
                     + " | ".join(percent(row[f"recall@{k}"]) for k in KS) + f" | {percent(row['random@3'])} |")
    lines += [
        "",
        "A result counts as correct when it overlaps the answer, so longer chunks have an easier job: "
        "look at how the random column grows. They also point to a coarser spot in the video.",
        "",
        "![Recall for different chunk lengths](images/chunk_size.png)",
        "",
        "#### The 'not found' cut-off",
        "",
    ]
    if summary["cutoffs"]:
        lines += [
            "| Search method | Cut-off | Answerable questions answered | Other questions refused | Balance on unseen questions |",
            "|---|---|---|---|---|",
        ]
        for method, row in summary["cutoffs"].items():
            cv = row["cv"]
            unseen = "too few questions" if cv is None else f"{cv[0] * 100:.0f}% (+/- {cv[1] * 100:.0f})"
            lines.append(f"| {method_name(method)} | {row['value']:.2f} | {percent(row['answered'])} | "
                         f"{percent(row['refused'])} | {unseen} |")
        lines += [
            "",
            "The cut-off is the lowest score at which the assistant still answers. I chose it on these same "
            "questions, so the middle two columns are a little optimistic. The last column picks the cut-off "
            "on some questions and tests it on others (5-fold, repeated 10 times), which is the fairer number.",
            "",
            "![Scores of answerable and not answerable questions](images/not_found_scores.png)",
        ]
    else:
        lines.append("Add questions that the videos cannot answer (leave `video` empty in `eval/questions.csv`) "
                     "to measure and tune the cut-off.")
    return "\n".join(lines) + "\n"


def run_evaluation(transcripts, questions, embed_fn=None, image_dir=None, eval_dir=None, log=None):
    """Run every experiment, save the charts and files, and return a summary dictionary.
    log (for example `print`) is called with progress messages: embedding all the chunks takes a few minutes."""
    image_dir, eval_dir = Path(image_dir or config.IMAGE_DIR), Path(eval_dir or config.EVAL_DIR)
    progress = rag.progress_printer(out=log) if log else None
    log = log or (lambda message: None)
    image_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    style_charts()

    answerable = questions["in_scope"].to_numpy()
    chunk_seconds, overlap_seconds = config.CHUNK_SECONDS, config.OVERLAP_SECONDS
    log(f"Embedding the chunks of the main search ({chunk_seconds} s) ...")
    main_index = rag.build_index(transcripts, chunk_seconds, overlap_seconds, embed_fn, progress=progress)
    n_chunks = int(len(main_index.chunks))

    # 1. random guessing vs keyword search vs search by meaning
    runs = {method: evaluate_method(questions, main_index, method, embed_fn) for method in rag.METHODS}
    methods = {m: {k: recall_at_k(runs[m]["ranks"], answerable, k) for k in KS} for m in rag.METHODS}
    intervals = {m: {k: wilson_interval(*recall_counts(runs[m]["ranks"], answerable, k)) for k in KS}
                 for m in rag.METHODS}
    correct_chunks = runs["embeddings"]["n_correct"]          # the same for every method: it depends on the index only
    baseline = {k: baseline_recall(correct_chunks, answerable, n_chunks, k) for k in KS}
    low, high = intervals["embeddings"][3]
    ci_points = (high - low) / 2 * 100

    # 2. chunk length
    sweep = []
    for size, overlap in sorted(set(SWEEP + [(chunk_seconds, overlap_seconds)])):
        if (size, overlap) == (chunk_seconds, overlap_seconds):
            index, run = main_index, runs["embeddings"]
        else:
            log(f"Embedding the chunks for the chunk length {size} s ...")
            index = rag.build_index(transcripts, size, overlap, embed_fn, progress=progress)
            run = evaluate_method(questions, index, "embeddings", embed_fn)
        lengths = index.chunks["end"] - index.chunks["start"]
        row = {"chunk_seconds": size, "overlap_seconds": overlap, "n_chunks": int(len(index.chunks)),
               "avg_seconds": float(lengths.mean())}
        row.update({f"recall@{k}": recall_at_k(run["ranks"], answerable, k) for k in KS})
        row.update({f"random@{k}": baseline_recall(run["n_correct"], answerable, len(index.chunks), k) for k in KS})
        sweep.append(row)

    # 3. the 'not found' cut-off
    cutoffs = {}
    if answerable.any() and (~answerable).any():
        for method in rag.METHODS:
            top = runs[method]["top_scores"]
            value = best_threshold(top, answerable)
            answered, refused = split_rates(top, answerable, value)
            cutoffs[method] = {"value": value, "answered": answered, "refused": refused,
                               "cv": cross_validated_balance(top, answerable)}

    # 4. the mistakes
    failures = failure_table(questions, main_index, runs["embeddings"])
    failures.to_csv(eval_dir / "failures.csv", index=False, encoding="utf-8")

    summary = {
        "n_answerable": int(answerable.sum()), "n_unanswerable": int((~answerable).sum()),
        "n_videos": len(transcripts),
        "minutes": sum(max((s["end"] for s in segs), default=0.0) for segs in transcripts.values()) / 60,
        "chunk_seconds": chunk_seconds, "overlap_seconds": overlap_seconds, "n_chunks": n_chunks,
        "methods": methods, "intervals": intervals, "baseline": baseline, "ci_points": ci_points,
        "sweep": sweep, "cutoffs": cutoffs, "failures": failures,
    }

    subtitle = f"{summary['n_answerable']} answerable questions about {summary['n_videos']} lecture videos."
    plot_methods(methods, intervals, baseline, image_dir / "retrieval_methods.png", subtitle)
    plot_chunk_sizes(sweep, image_dir / "chunk_size.png", subtitle)
    if cutoffs:
        plot_scores(runs["embeddings"]["top_scores"], answerable, cutoffs["embeddings"]["value"],
                    "embeddings", image_dir / "not_found_scores.png")
    summary["markdown"] = results_markdown(summary)
    (eval_dir / "results.md").write_text(summary["markdown"], encoding="utf-8")
    return summary


def print_summary(summary):
    print(f"\n{summary['n_answerable']} answerable + {summary['n_unanswerable']} not answerable questions, "
          f"{summary['n_videos']} videos, {summary['n_chunks']} chunks "
          f"(~{summary['chunk_seconds']} s, overlap {summary['overlap_seconds']} s)")
    rows = {"Random guessing": {f"Recall@{k}": percent(summary["baseline"][k]) for k in KS}}
    for method in SHOWN_ORDER:
        values = summary["methods"][method]
        rows[method_name(method)] = {f"Recall@{k}": percent_with_range(values[k], summary["intervals"][method][k])
                                     for k in KS}
    print("\nIs the right passage among the top results?  (in brackets: 95% confidence interval)\n"
          + pd.DataFrame(rows).T.to_string())
    sweep = pd.DataFrame([{"chunk": f"{r['chunk_seconds']} s", "chunks": r["n_chunks"],
                           **{f"Recall@{k}": percent(r[f"recall@{k}"]) for k in KS},
                           "random@3": percent(r["random@3"])} for r in summary["sweep"]])
    print("\nChunk length (search by meaning)\n" + sweep.to_string(index=False))
    for method, row in summary["cutoffs"].items():
        cv = row["cv"]
        unseen = "too few questions" if cv is None else f"{cv[0] * 100:.0f}% +/- {cv[1] * 100:.0f}"
        print(f"\n'Not found' cut-off for {method_name(method)}: {row['value']:.2f} -> answers "
              f"{percent(row['answered'])} of answerable questions, refuses {percent(row['refused'])} of the others "
              f"(balance on unseen questions: {unseen}).")
    if not summary["cutoffs"]:
        print("\nNo not-answerable questions yet, so no cut-off was tuned.")
    failures = summary["failures"]
    print(f"\n{len(failures)} answerable question(s) where the right passage is not in the top 3 "
          "(saved in eval/failures.csv):")
    for row in failures.itertuples(index=False):
        print(f"  - {row.question}\n      expected {row.expected} | best results: {row.top_results}")


def write_into_readme(readme_path, block):
    """Replace the text between the RESULTS markers in README.md."""
    readme_path = Path(readme_path)
    text = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise ValueError(f"{readme_path.name} has no {README_START} ... {README_END} markers.")
    new_block = f"{README_START}\n\n{block.strip()}\n\n{README_END}"
    readme_path.write_text(pattern.sub(lambda match: new_block, text, count=1), encoding="utf-8")


def main(argv=None, embed_fn=None):
    rag.setup_console()
    parser = argparse.ArgumentParser(description="Measure how good the search is.")
    parser.add_argument("--write-readme", action="store_true", help="paste the results tables into README.md")
    args = parser.parse_args(argv)

    transcripts = rag.load_transcripts(config.TRANSCRIPT_DIR)
    if not transcripts:
        print(f"No transcripts found in {config.TRANSCRIPT_DIR}. Run step1_transcribe.py first.")
        return 1
    durations = {video: max((s["end"] for s in segs), default=0.0) for video, segs in transcripts.items()}
    try:
        questions = load_questions(config.EVAL_DIR / "questions.csv", durations)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1
    if not questions["in_scope"].any():
        print("eval/questions.csv has no answerable questions yet. Add rows with a video name and the "
              "time range of the answer (see eval/HOW_TO_WRITE_QUESTIONS.md).")
        return 1
    try:
        summary = run_evaluation(transcripts, questions, embed_fn, log=print)
    except rag.OllamaError as error:
        print(f"Stopped: {error}")
        return 1
    print_summary(summary)
    print(f"\nSaved: {config.EVAL_DIR / 'results.md'}, {config.EVAL_DIR / 'failures.csv'} and charts in {config.IMAGE_DIR}")
    if args.write_readme:
        write_into_readme(config.ROOT / "README.md", summary["markdown"])
        print("README.md updated.")
    else:
        print("To paste the results into README.md run:  python evaluate.py --write-readme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
