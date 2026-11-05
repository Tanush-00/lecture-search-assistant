"""Tests for the building blocks in rag.py. Run from the project folder:  python -m pytest"""
import random

import numpy as np
import pandas as pd
import pytest

import config
import rag
from conftest import DIM, fake_embed, make_segments


def seg(start, end, text="word"):
    return {"start": start, "end": end, "text": text}


def uniform(n, length=2.0):
    return [seg(i * length, (i + 1) * length, f"s{i}") for i in range(n)]


# ---------------------------------------------------------------- time helpers
def test_format_time():
    assert rag.format_time(0) == "0:00"
    assert rag.format_time(83.4) == "1:23"
    assert rag.format_time(59.99) == "0:59"
    assert rag.format_time(3725) == "1:02:05"
    assert rag.format_time(-3) == "0:00"


def test_parse_time():
    assert rag.parse_time("1:25") == 85
    assert rag.parse_time("85") == 85
    assert rag.parse_time(" 0:35.5 ") == 35.5
    assert rag.parse_time("1:02:05") == 3725
    assert rag.parse_time(40) == 40


@pytest.mark.parametrize("bad", ["", "abc", "1:75", "-5", "nan", "inf", "1:2:3:4", "1:", ":30"])
def test_parse_time_rejects_nonsense(bad):
    with pytest.raises(ValueError, match="Cannot read the time"):
        rag.parse_time(bad)


def test_parse_and_format_agree():
    for seconds in (0, 7, 59, 60, 61, 599, 3600, 3725):
        assert rag.parse_time(rag.format_time(seconds)) == seconds


def test_video_title():
    assert rag.video_title("02_overfitting_cross_validation") == "Overfitting cross validation"
    assert rag.video_title("08-what-is-rag") == "What is rag"
    assert rag.video_title("7") == "7"                      # nothing left after the number: keep the id


# ---------------------------------------------------------------- the lecture catalog
CATALOG_TEXT = (
    "video_id,unit,sequence,video_no,title,archive_file,size_mb,selected,check_regex\n"
    "3.2-V5,Unit 3,3.2 Modeling the Expert,5,Thresholding,f5.mp4,11.2,1,threshold|sensitivity\n"
    "3.2-V6, Unit 3 ,3.2 Modeling the Expert,6, ROC Curves ,f6.mp4,9.0,TRUE,\n"
    "3.2-V7,Unit 3,3.2 Modeling the Expert,7,Not chosen,f7.mp4,3.0,0,\n"
    ",,,,,,,,\n"
)


@pytest.fixture
def catalog_file(tmp_path, monkeypatch):
    path = tmp_path / "catalog.csv"
    path.write_text(CATALOG_TEXT, encoding="utf-8")
    monkeypatch.setattr(config, "CATALOG_FILE", path)
    return path


def test_read_catalog_cleans_the_rows_and_keeps_their_order(catalog_file):
    catalog = rag.read_catalog(catalog_file)
    assert list(catalog) == ["3.2-V5", "3.2-V6", "3.2-V7"]                # the empty row is skipped
    assert catalog["3.2-V6"]["title"] == "ROC Curves" and catalog["3.2-V6"]["unit"] == "Unit 3"   # spaces removed
    assert [row["selected"] for row in catalog.values()] == [True, True, False]    # "1" and "TRUE" both mean yes


def test_read_catalog_accepts_a_file_saved_by_excel(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_bytes(CATALOG_TEXT.encode("utf-8-sig"))                    # Excel puts an invisible mark at the start
    assert list(rag.read_catalog(path)) == ["3.2-V5", "3.2-V6", "3.2-V7"]


def test_read_catalog_without_a_file_is_empty(tmp_path):
    assert rag.read_catalog(tmp_path / "nope.csv") == {}


def test_read_catalog_explains_a_broken_file(tmp_path):
    path = tmp_path / "catalog.csv"
    path.write_text("video_id,title\n3.2-V5,Thresholding\n", encoding="utf-8")
    with pytest.raises(ValueError, match="archive_file"):
        rag.read_catalog(path)
    path.write_text("video_id,title,archive_file\na,A,a.mp4\nb,B,b.mp4\na,C,c.mp4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 4.*'a' appears twice"):
        rag.read_catalog(path)


def test_the_catalog_gives_titles_and_places_in_the_course(catalog_file):
    assert rag.video_title("3.2-V5") == "Thresholding"
    assert rag.video_context("3.2-V5") == "3.2 Modeling the Expert - video 5"
    assert rag.video_title("02_overfitting") == "Overfitting"            # not in the catalog: tidy up the name
    assert rag.video_context("02_overfitting") == ""
    assert [row["video_id"] for row in rag.catalog_videos()] == ["3.2-V5", "3.2-V6"]
    assert [row["video_id"] for row in rag.catalog_videos(only_selected=False)] == ["3.2-V5", "3.2-V6", "3.2-V7"]


def test_the_catalog_is_read_again_when_the_file_changes(catalog_file):
    import os
    assert rag.video_title("3.2-V5") == "Thresholding"
    catalog_file.write_text(CATALOG_TEXT.replace("Thresholding", "A new title"), encoding="utf-8")
    stamp = catalog_file.stat().st_mtime + 5
    os.utime(catalog_file, (stamp, stamp))
    assert rag.video_title("3.2-V5") == "A new title"


def test_without_a_catalog_file_titles_come_from_the_names(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CATALOG_FILE", tmp_path / "missing.csv")
    assert rag.catalog() == {} and rag.catalog_videos() == []
    assert rag.video_title("02_overfitting") == "Overfitting"


def test_the_shipped_catalog_is_complete_and_consistent():
    import re
    import config as project_config
    rows = list(rag.read_catalog(project_config.ROOT / "catalog.csv").values())
    assert len(rows) == 82 and sum(row["selected"] for row in rows) == 36
    assert len({row["archive_file"] for row in rows}) == 82                      # no file is listed twice
    sequences = {}
    for row in rows:
        assert re.fullmatch(r"\d\.\d-V\d+", row["video_id"]), row["video_id"]
        assert re.fullmatch(r"MIT15_071S17_Session_\d\.\d\.\d\d_300k\.mp4", row["archive_file"]), row["archive_file"]
        assert row["video_id"].split("-")[0] == row["sequence"].split()[0]       # 3.2-V5 belongs to sequence 3.2
        assert row["archive_file"].split("Session_")[1].startswith(row["video_id"].split("-")[0] + ".")
        assert row["title"] and float(row["size_mb"]) > 0
        sequences.setdefault(row["sequence"], []).append(int(row["video_no"]))
        if row["selected"]:
            assert row["check_regex"], f"{row['video_id']} is chosen but has no check_regex"
        re.compile(row["check_regex"] or "x")                                    # every pattern must be valid
    for numbers in sequences.values():
        assert numbers == list(range(1, len(numbers) + 1))                       # video 1, 2, 3 ... without gaps
    chosen_mb = sum(float(row["size_mb"]) for row in rows if row["selected"])
    assert 380 < chosen_mb < 430                                                 # "about 400 MB" in the README


# ---------------------------------------------------------------- transcripts
def test_clean_segments_keeps_only_what_is_needed():
    raw = [
        {"start": 1.234, "end": 2.999, "text": "  hello  ", "tokens": [1, 2], "avg_logprob": -0.2},
        {"start": 0.0, "end": 1.0, "text": "   "},                         # empty -> dropped
        {"start": 0.5, "end": 0.2, "text": "backwards end"},                # end before start -> fixed
    ]
    clean = rag.clean_segments(raw)
    assert clean == [{"start": 0.5, "end": 0.5, "text": "backwards end"},
                     {"start": 1.23, "end": 3.0, "text": "hello"}]


def test_transcript_round_trip_keeps_unicode(tmp_path):
    segments = [seg(0, 2, "Hello ’world’"), seg(2, 4, "नमस्ते")]
    rag.save_transcript(tmp_path, "01_demo", segments, {"whisper_model": "tiny"})
    assert rag.load_transcripts(tmp_path) == {"01_demo": segments}
    text = (tmp_path / "01_demo.txt").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "[0:00] Hello ’world’"
    assert text.splitlines()[1].startswith("[0:02] ")


def test_load_transcripts_of_missing_folder_is_empty(tmp_path):
    assert rag.load_transcripts(tmp_path / "nope") == {}


def test_find_media_files(tmp_path):
    for name in ("02_b.mp4", "01_a.MOV", "notes.txt", "03_c.m4a"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in rag.find_media_files(tmp_path)] == ["01_a.MOV", "02_b.mp4", "03_c.m4a"]
    assert rag.find_media_for("02_b", tmp_path).name == "02_b.mp4"
    assert rag.find_media_for("zzz", tmp_path) is None
    assert rag.find_media_files(tmp_path / "missing") == []


# ---------------------------------------------------------------- chunks
def test_chunks_of_nothing_and_of_one_segment():
    assert rag.make_chunks([], 30, 5) == []
    assert rag.make_chunks([seg(0, 3, "hello")], 30, 5) == [{"start": 0, "end": 3, "text": "hello"}]


def test_short_video_is_a_single_chunk():
    chunks = rag.make_chunks(uniform(10), 30, 5)             # 20 seconds
    assert len(chunks) == 1
    assert (chunks[0]["start"], chunks[0]["end"]) == (0, 20)
    assert chunks[0]["text"] == " ".join(f"s{i}" for i in range(10))


def test_uniform_video_chunks_are_short_overlapping_and_complete():
    segments = uniform(50)                                   # 100 seconds in 2-second segments
    ranges = rag.chunk_ranges(segments, 30, 5)
    assert ranges[0] == (0, 15)
    for first, last in ranges:
        assert segments[last - 1]["end"] - segments[first]["start"] <= 30
    for (_, last_a), (first_b, _) in zip(ranges, ranges[1:]):
        overlap = segments[last_a - 1]["end"] - segments[first_b]["start"]
        assert 0 < overlap <= 5                              # neighbours share a little, never more than asked
    assert ranges[-1][1] == len(segments)                    # the end of the video is covered


def test_zero_overlap_gives_clean_pieces():
    ranges = rag.chunk_ranges(uniform(30), 20, 0)           # 60 seconds, 20-second chunks
    assert ranges == [(0, 10), (10, 20), (20, 30)]


def test_one_very_long_segment_stays_whole():
    segments = [seg(0, 60, "long"), seg(60, 62, "next")]
    assert rag.chunk_ranges(segments, 30, 5) == [(0, 1), (1, 2)]


def test_no_chunk_is_a_copy_of_part_of_the_previous_one():
    # The window that starts at Y could not include the long Z, so it would only repeat Y.
    segments = [seg(0, 26, "X"), seg(26, 29, "Y"), seg(29, 60, "Z")]
    assert rag.chunk_ranges(segments, 30, 5) == [(0, 2), (2, 3)]


def test_empty_segments_are_ignored():
    chunks = rag.make_chunks([seg(0, 2, "a"), seg(2, 4, "  "), seg(4, 6, "b")], 30, 5)
    assert chunks == [{"start": 0, "end": 6, "text": "a b"}]


def test_bad_chunk_settings_are_rejected():
    for chunk_seconds, overlap in ((0, 0), (-5, 0), (30, 30), (30, 45), (30, -1)):
        with pytest.raises(ValueError):
            rag.chunk_ranges(uniform(5), chunk_seconds, overlap)


def test_chunking_rules_hold_for_random_videos():
    generator = random.Random(1)
    for _ in range(300):
        segments, clock = [], 0.0
        for _ in range(generator.randint(1, 60)):
            clock += generator.choice([0, 0, 0.3, 1.5])
            length = generator.choice([0.5, 1, 2, 3, 5, 8, 40])
            segments.append(seg(clock, clock + length))
            clock += length
        chunk_seconds = generator.choice([10, 15, 30, 60])
        overlap = generator.choice([0, 2, 5, 9])
        if overlap >= chunk_seconds:
            continue
        ranges = rag.chunk_ranges(segments, chunk_seconds, overlap)
        covered = set()
        for first, last in ranges:
            covered.update(range(first, last))
            too_long = segments[last - 1]["end"] - segments[first]["start"] > chunk_seconds
            assert not too_long or last - first == 1        # only a single long segment may exceed the limit
        assert covered == set(range(len(segments)))        # nothing is lost
        assert [r[1] for r in ranges] == sorted({r[1] for r in ranges})     # each chunk reaches further than the last
        assert [r[0] for r in ranges] == sorted(r[0] for r in ranges)
        for (a_first, a_last), (b_first, b_last) in zip(ranges, ranges[1:]):
            assert not (a_first <= b_first and b_last <= a_last)             # no chunk inside another


# ---------------------------------------------------------------- Ollama (talking to a fake server)
def test_embed_texts_returns_unit_vectors_in_the_same_order(fake_ollama):
    url, server = fake_ollama
    texts = ["precision and recall", "sql joins", "overfitting", "cross validation", "random forest"]
    vectors = rag.embed_texts(texts, url=url, batch_size=2)
    assert vectors.shape == (5, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert len(server.requests) == 3                        # 5 texts in batches of 2
    assert np.allclose(vectors, fake_embed(texts), atol=1e-5)


def test_embed_texts_of_nothing(fake_ollama):
    url, _ = fake_ollama
    assert rag.embed_texts([], url=url).shape[0] == 0


def test_embed_texts_reports_progress_after_every_batch(fake_ollama):
    url, _ = fake_ollama
    seen = []
    rag.embed_texts(["a one", "b two", "c three", "d four", "e five"], url=url, batch_size=2,
                    progress=lambda done, total: seen.append((done, total)))
    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_progress_printer_prints_about_ten_lines_and_ends_with_100_percent():
    lines = []
    show = rag.progress_printer("chunks", out=lines.append)
    for done in range(1, 431):                                # 430 chunks, one call per chunk
        show(done, 430)
    assert 9 <= len(lines) <= 11
    assert lines[-1] == "  embedded 430/430 chunks (100%)"
    assert lines[0].startswith("  embedded 1/430 chunks (0%)")
    small = []
    show = rag.progress_printer(out=small.append)
    for done in (1, 2, 3):
        show(done, 3)
    assert len(small) == 3 and small[-1].endswith("(100%)")   # few batches: every one is shown
    rag.progress_printer(out=small.append)(0, 0)               # nothing to do: no crash, no division by zero


def test_missing_model_gives_a_helpful_message(fake_ollama):
    url, _ = fake_ollama
    with pytest.raises(rag.OllamaError, match="ollama pull missing-model"):
        rag.embed_texts(["hello"], model="missing-model", url=url)


def test_ollama_not_running_gives_a_helpful_message():
    with pytest.raises(rag.OllamaError, match="Cannot reach Ollama"):
        rag.embed_texts(["hello"], url="http://127.0.0.1:1")
    with pytest.raises(rag.OllamaError, match="Cannot reach Ollama"):
        rag.ask_llm("hello", url="http://127.0.0.1:1")


def test_ask_llm_sends_the_prompt_and_trims_the_answer(fake_ollama):
    url, server = fake_ollama
    answer = rag.ask_llm("Say hi", url=url)
    assert answer.startswith("MOCK ANSWER") and answer == answer.strip()
    path, request = server.requests[-1]
    assert path == "/api/generate"
    assert request["prompt"] == "Say hi" and request["stream"] is False and request["model"] == config.LLM_MODEL
    assert request["options"]["temperature"] == 0


def test_ollama_status(fake_ollama):
    url, _ = fake_ollama
    assert rag.ollama_status(url) == (True, ["bge-m3:latest", "llama3.2:latest"])
    assert rag.ollama_status("http://127.0.0.1:1") == (False, [])


def test_has_model():
    names = ["bge-m3:latest", "llama3.2:3b"]
    assert rag.has_model(names, "bge-m3")
    assert rag.has_model(names, "llama3.2")
    assert rag.has_model(names, "llama3.2:3b")
    assert not rag.has_model(names, "llama3.2:1b")
    assert not rag.has_model(names, "mistral")
    assert not rag.has_model([], "bge-m3")


# ---------------------------------------------------------------- index
def test_build_index_table_and_matrix(index, transcripts):
    chunks = index.chunks
    assert list(chunks.columns) == ["chunk_id", "video", "start", "end", "text"]
    assert list(chunks["chunk_id"]) == list(range(len(chunks)))
    assert set(chunks["video"]) == set(transcripts)
    assert index.matrix.shape == (len(chunks), DIM)
    assert (chunks["end"] > chunks["start"]).all()


def test_index_round_trip(tmp_path, index):
    rag.save_index(tmp_path / "idx", index)
    loaded = rag.load_index(tmp_path / "idx")
    pd.testing.assert_frame_equal(loaded.chunks, index.chunks)
    assert np.allclose(loaded.matrix, index.matrix)
    assert loaded.settings == index.settings


def test_build_index_passes_progress_on_only_when_asked(transcripts):
    calls = []

    def embed_with_progress(texts, progress=None):
        calls.append(progress)
        return fake_embed(texts)

    def plain_embed(texts):                                    # an embedder that knows nothing about progress
        return fake_embed(texts)

    marker = lambda done, total: None
    rag.build_index(transcripts, 30, 5, embed_fn=embed_with_progress, progress=marker)
    rag.build_index(transcripts, 30, 5, embed_fn=embed_with_progress)
    assert calls == [marker, None]
    assert len(rag.build_index(transcripts, 30, 5, embed_fn=plain_embed).chunks) > 3     # no progress: old style works


def test_video_names_that_look_like_numbers_stay_text(tmp_path, transcripts):
    numbered = {"1": transcripts["01_precision_recall"]}
    saved = rag.build_index(numbered, 30, 5, embed_fn=fake_embed)
    rag.save_index(tmp_path, saved)
    assert rag.load_index(tmp_path).chunks["video"].tolist()[0] == "1"


def test_load_index_errors(tmp_path, index):
    with pytest.raises(FileNotFoundError, match="step2_build_index"):
        rag.load_index(tmp_path / "empty")
    rag.save_index(tmp_path, index)
    np.save(tmp_path / "embeddings.npy", index.matrix[:-1])
    with pytest.raises(ValueError, match="do not match"):
        rag.load_index(tmp_path)


def test_index_must_match_the_embedding_model(index, monkeypatch):
    rag.check_index_matches_config(index)                    # built with the configured model: fine
    monkeypatch.setattr(config, "EMBED_MODEL", "another-model")
    with pytest.raises(ValueError, match="another-model"):
        rag.check_index_matches_config(index)


# ---------------------------------------------------------------- search
def test_search_finds_the_right_part_of_the_right_video(index):
    hits = rag.search("What is overfitting?", index, "embeddings", top_k=3, embed_fn=fake_embed)
    assert hits.iloc[0]["video"] == "02_overfitting"
    assert hits["score"].is_monotonic_decreasing and len(hits) == 3


def test_keyword_search_finds_it_too(index):
    hits = rag.search("difference between inner join and left join", index, "keywords", top_k=2)
    assert hits.iloc[0]["video"] == "03_sql_joins"
    assert (hits["score"] > 0).all()


def test_keyword_search_handles_questions_without_usable_words(index):
    hits = rag.search("the of and", index, "keywords", top_k=2)
    assert (hits["score"] == 0).all() and len(hits) == 2


def test_unknown_search_method_is_rejected(index):
    with pytest.raises(ValueError, match="Unknown search method"):
        rag.search("anything", index, "magic", embed_fn=fake_embed)


def test_embedding_size_mismatch_is_explained(index):
    def wrong_size(texts):
        return np.ones((len(texts), 7), dtype=np.float32)
    with pytest.raises(ValueError, match="step2_build_index"):
        rag.search("anything", index, "embeddings", embed_fn=wrong_size)


# ---------------------------------------------------------------- answer
def test_answer_is_written_only_when_something_is_found(index):
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return "an answer"

    found = rag.answer_question("What is overfitting?", index, "embeddings", min_score=0.1,
                                embed_fn=fake_embed, llm_fn=llm)
    assert found["found"] and found["answer"] == "an answer" and len(calls) == 1
    assert "[02_overfitting," in calls[0] and "What is overfitting?" in calls[0]

    missing = rag.answer_question("What is the capital of France?", index, "embeddings", min_score=0.1,
                                  embed_fn=fake_embed, llm_fn=llm)
    assert not missing["found"] and missing["answer"] is None and len(calls) == 1      # the LLM was not asked

    silent = rag.answer_question("What is overfitting?", index, "embeddings", min_score=0.1,
                                 use_llm=False, embed_fn=fake_embed, llm_fn=llm)
    assert silent["found"] and silent["answer"] is None and len(calls) == 1


def test_default_cutoff_comes_from_the_config(index, monkeypatch):
    monkeypatch.setitem(config.MIN_SCORE, "embeddings", 0.99)
    result = rag.answer_question("What is overfitting?", index, "embeddings", use_llm=False, embed_fn=fake_embed)
    assert result["cutoff"] == 0.99 and not result["found"]


def test_prompt_has_labelled_passages_and_the_rules(index):
    hits = rag.search("What is overfitting?", index, "embeddings", top_k=2, embed_fn=fake_embed)
    prompt = rag.make_prompt("What is overfitting?", hits)
    assert "ONLY the video passages" in prompt and "The videos do not answer this." in prompt
    first = hits.iloc[0]
    assert f"[{first['video']}, {rag.format_time(first['start'])}-{rag.format_time(first['end'])}]" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_prompt_survives_curly_braces_in_the_transcript(index):
    hits = index.chunks.head(1).copy()
    hits["text"] = "a {weird} {0} text"
    assert "a {weird} {0} text" in rag.make_prompt("What {is} this?", hits)


def test_make_segments_helper_is_consistent():
    segments = make_segments(["a b", "c d"])
    assert segments == [{"start": 0.0, "end": 4.0, "text": "a b"}, {"start": 4.0, "end": 8.0, "text": "c d"}]
