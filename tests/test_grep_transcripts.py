"""Tests for grep_transcripts.py (the helper for checking eval/questions.csv)."""
import pytest

import config
import grep_transcripts
import rag
from conftest import make_segments


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """Two small transcripts saved in a temporary transcripts folder."""
    monkeypatch.setattr(config, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    rag.save_transcript(config.TRANSCRIPT_DIR, "3.2-V5", make_segments([
        "We pick a threshold between zero and one.",
        "A high threshold gives few false positives.",
        "The receiver operating",
        "characteristic curve plots both rates.",
        "Nothing else here.",
    ]), {})
    rag.save_transcript(config.TRANSCRIPT_DIR, "4.2-V2", make_segments([
        "A tree splits the data again and again.",
        "Each split uses one variable.",
    ]), {})
    return config.TRANSCRIPT_DIR


def test_find_lines_ignores_capital_letters_and_accepts_alternatives(stored):
    transcripts = rag.load_transcripts(stored)
    found = grep_transcripts.find_lines(transcripts, "THRESHOLD|split")
    assert [(video, start) for video, start, _ in found] == [("3.2-V5", 0.0), ("3.2-V5", 4.0),
                                                           ("4.2-V2", 0.0), ("4.2-V2", 4.0)]
    assert grep_transcripts.find_lines(transcripts, "threshold", video="4.2-V2") == []
    assert grep_transcripts.find_lines(transcripts, "k-means") == []


def test_a_phrase_that_continues_in_the_next_line_is_found_once(stored):
    transcripts = rag.load_transcripts(stored)
    found = grep_transcripts.find_lines(transcripts, "operating characteristic")
    assert [(video, start) for video, start, _ in found] == [("3.2-V5", 8.0)]       # reported at its first line
    assert len(grep_transcripts.find_lines(transcripts, "characteristic")) == 1      # no double report


def test_lines_between_returns_what_is_said_in_a_time_range(stored):
    segments = rag.load_transcripts(stored)["3.2-V5"]
    assert [s["start"] for s in grep_transcripts.lines_between(segments, 5, 9)] == [4.0, 8.0]
    assert [s["start"] for s in grep_transcripts.lines_between(segments, 4, 8)] == [4.0]     # touching is not enough
    assert grep_transcripts.lines_between(segments, 100, 200) == []


def test_main_prints_lines_with_video_and_time(stored, capsys):
    assert grep_transcripts.main(["threshold"]) == 0
    out = capsys.readouterr().out
    assert "3.2-V5   [0:00] We pick a threshold" in out and "[0:04] A high threshold" in out
    assert "2 line(s) in 1 video(s) say 'threshold'" in out
    assert grep_transcripts.main(["k-means"]) == 0
    assert "No line says 'k-means'." in capsys.readouterr().out


def test_main_show_prints_a_time_range_and_knows_the_title(stored, capsys):
    assert grep_transcripts.main(["--show", "3.2-v5", "0:04", "0:12"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("3.2-V5  ") and "0:04 - 0:12" in out.splitlines()[0]
    assert "[0:04] A high threshold" in out and "[0:08] The receiver operating" in out
    assert "We pick a threshold" not in out
    assert grep_transcripts.main(["--show", "3.2-V5", "5:00", "6:00"]) == 0
    assert "nothing is said in this range" in capsys.readouterr().out


def test_main_explains_mistakes(stored, tmp_path, monkeypatch, capsys):
    assert grep_transcripts.main([]) == 1
    assert "Give a word" in capsys.readouterr().out
    assert grep_transcripts.main(["threshold", "--video", "9.9-V9"]) == 1
    assert "No transcript called '9.9-V9'" in capsys.readouterr().out
    assert grep_transcripts.main(["(unclosed"]) == 1
    assert capsys.readouterr().out.strip() != ""                                     # a clear message, no traceback
    assert grep_transcripts.main(["--show", "3.2-V5", "abc", "0:10"]) == 1
    assert "Cannot read the time" in capsys.readouterr().out
    monkeypatch.setattr(config, "TRANSCRIPT_DIR", tmp_path / "empty")
    assert grep_transcripts.main(["threshold"]) == 1
    assert "Run step1_transcribe.py first" in capsys.readouterr().out
