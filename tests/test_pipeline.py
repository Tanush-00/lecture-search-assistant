"""Tests for the scripts: step 1, step 2, ask.py, evaluate.py, check_setup.py and the web app."""
import builtins
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ask
import check_setup
import config
import evaluate
import rag
import step1_transcribe
import step2_build_index
from conftest import QUESTIONS, fake_embed

ROOT = Path(__file__).resolve().parents[1]


def ollama_down(*_args, **_kwargs):
    raise rag.OllamaError("Cannot reach Ollama at http://localhost:11434.")


# ================================================================ step 1: Whisper
class FakeWhisperModel:
    def __init__(self):
        self.calls, self.loaded = [], []

    def transcribe(self, path, **options):
        self.calls.append((path, options))
        return {"text": "ignored", "segments": [
            {"start": 0.0, "end": 3.0, "text": " Hello there. ", "tokens": [1, 2]},
            {"start": 3.0, "end": 6.5, "text": " Second line. ", "no_speech_prob": 0.01},
        ]}


def set_ffmpeg(monkeypatch, present):
    """Make shutil.which('ffmpeg') answer 'installed' or 'not installed' whatever this computer has."""
    real_which = shutil.which

    def which(name, *args, **kwargs):
        if name == "ffmpeg":
            return "/usr/bin/ffmpeg" if present else None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", which)


@pytest.fixture
def fake_whisper(monkeypatch):
    """openai-whisper and torch (no GPU) are replaced by fakes; faster-whisper counts as not installed."""
    model = FakeWhisperModel()
    whisper = types.ModuleType("whisper")
    whisper.load_model = lambda name: (model.loaded.append(name), model)[1]
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)          # None in sys.modules = "import fails"
    set_ffmpeg(monkeypatch, True)
    return model


class FakeFasterWhisper:
    """Stands in for faster_whisper.WhisperModel: transcribe() returns (an iterator of segments, info)."""
    instances = []

    def __init__(self, name, device="auto", compute_type="default"):
        self.name, self.device, self.compute_type, self.calls = name, device, compute_type, []
        FakeFasterWhisper.instances.append(self)

    def transcribe(self, path, language=None, task="transcribe", initial_prompt=None):
        self.calls.append((path, language, task, initial_prompt))
        pieces = [types.SimpleNamespace(start=0.0, end=3.0, text=" Hello there. "),
                  types.SimpleNamespace(start=3.0, end=6.5, text=" Second line. ")]
        return iter(pieces), types.SimpleNamespace(language=language, duration=6.5)


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    FakeFasterWhisper.instances = []
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeFasterWhisper
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "whisper", None)                  # openai-whisper is not installed here
    monkeypatch.setitem(sys.modules, "torch", None)
    set_ffmpeg(monkeypatch, False)                                     # faster-whisper does not need ffmpeg
    return FakeFasterWhisper


@pytest.fixture(autouse=True)
def no_real_faster_whisper_audio(monkeypatch):
    """The audio self-test of faster-whisper (faster_whisper_audio_error) never reaches a really installed
    faster-whisper in these tests: a test that wants it to run uses fake_audio below."""
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", None)


AV_19_ERROR = TypeError("open() got an unexpected keyword argument 'metadata_errors'")


@pytest.fixture
def fake_audio(monkeypatch, fake_faster_whisper):
    """faster_whisper.audio.decode_audio: it opens the file it is given, then works or raises state.error."""
    state = types.SimpleNamespace(error=None, files=[])
    module = types.ModuleType("faster_whisper.audio")

    def decode_audio(path, sampling_rate=16000):
        state.files.append((path, Path(path).read_bytes()[:4], sampling_rate))      # the file exists right now
        if state.error:
            raise state.error
        return np.zeros(1600, dtype="float32")

    module.decode_audio = decode_audio
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", module)
    return state


@pytest.fixture
def folders(tmp_path, monkeypatch):
    for name in ("VIDEO_DIR", "TRANSCRIPT_DIR", "INDEX_DIR", "EVAL_DIR", "IMAGE_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.split("_")[0].lower())
    config.VIDEO_DIR.mkdir()
    return tmp_path


def test_transcribe_one_passes_the_settings_to_whisper():
    model = FakeWhisperModel()
    segments = step1_transcribe.transcribe_one(model, Path("x.mp4"), "hi", "translate", False, prompt="")
    assert segments == [{"start": 0.0, "end": 3.0, "text": "Hello there."},
                        {"start": 3.0, "end": 6.5, "text": "Second line."}]
    path, options = model.calls[0]
    assert path == "x.mp4"
    assert options == {"language": "hi", "task": "translate", "word_timestamps": False,
                       "fp16": False, "initial_prompt": None}


def test_step1_transcribes_new_videos_only(folders, fake_whisper, capsys):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    (config.VIDEO_DIR / "02_b.mov").write_bytes(b"x")
    assert step1_transcribe.main([]) == 0
    assert len(fake_whisper.calls) == 2 and fake_whisper.loaded == [config.WHISPER_MODEL]
    saved = rag.load_transcripts(config.TRANSCRIPT_DIR)
    assert sorted(saved) == ["01_a", "02_b"] and saved["01_a"][0]["text"] == "Hello there."
    meta = json.loads((config.TRANSCRIPT_DIR / "01_a.json").read_text(encoding="utf-8"))
    assert (meta["whisper_model"], meta["engine"], meta["source_file"]) == (config.WHISPER_MODEL, "openai-whisper", "01_a.mp4")
    assert (config.TRANSCRIPT_DIR / "01_a.txt").read_text(encoding="utf-8").startswith("[0:00] Hello there.")

    assert step1_transcribe.main([]) == 0                    # nothing new: Whisper is not run again
    assert len(fake_whisper.calls) == 2
    assert "already has a transcript" in capsys.readouterr().out

    assert step1_transcribe.main(["--only", "01_a", "--force"]) == 0           # --force: redo even if up to date
    assert len(fake_whisper.calls) == 3


def test_step1_redoes_transcripts_made_with_other_settings(folders, fake_whisper, monkeypatch):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    (config.VIDEO_DIR / "02_b.mp4").write_bytes(b"x")
    assert step1_transcribe.main(["--only", "01_a", "--model", "small"]) == 0     # a quick trial on one video
    assert fake_whisper.loaded == ["small"] and len(fake_whisper.calls) == 1
    assert step1_transcribe.main([]) == 0                                       # the real run, default model
    assert fake_whisper.loaded[-1] == config.WHISPER_MODEL
    assert len(fake_whisper.calls) == 3                                        # both videos: the trial one is redone
    assert step1_transcribe.main([]) == 0 and len(fake_whisper.calls) == 3     # now everything matches
    monkeypatch.setattr(config, "WHISPER_TASK", "translate")                   # another setting changed -> redo
    assert step1_transcribe.main([]) == 0 and len(fake_whisper.calls) == 5


def test_a_damaged_transcript_is_redone(folders):
    config.TRANSCRIPT_DIR.mkdir()
    (config.TRANSCRIPT_DIR / "01_a.json").write_text("{not json", encoding="utf-8")
    assert step1_transcribe.needs_transcript("01_a", config.WHISPER_MODEL)
    assert step1_transcribe.needs_transcript("02_missing", config.WHISPER_MODEL)


def test_step1_rejects_unknown_names_and_empty_folders(folders, fake_whisper, capsys):
    assert step1_transcribe.main([]) == 1
    out = capsys.readouterr().out
    assert "No videos found" in out and "step0_download_videos.py" in out         # tells you what to do first
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    assert step1_transcribe.main(["--only", "nope"]) == 1
    assert "No video called" in capsys.readouterr().out


def test_step1_without_whisper_installed(folders, monkeypatch, capsys):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    set_ffmpeg(monkeypatch, True)
    monkeypatch.setitem(sys.modules, "whisper", None)        # makes "import whisper" fail
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert step1_transcribe.main([]) == 1
    assert "pip install openai-whisper" in capsys.readouterr().out


def test_step1_uses_one_file_per_video_name():
    files = [Path("01_x.mp3"), Path("01_x.mp4"), Path("02_y.mp4")]
    assert [p.name for p in step1_transcribe.pick_files(files)] == ["01_x.mp3", "02_y.mp4"]


def test_step1_only_ignores_capital_letters():
    files = [Path("3.2-V5.mp4"), Path("3.2-V6.mp4"), Path("4.2-V2.mp4")]
    assert [p.name for p in step1_transcribe.pick_files(files, ["4.2-v2", "3.2-v5"])] == ["4.2-V2.mp4", "3.2-V5.mp4"]
    with pytest.raises(ValueError, match="3.9-V9"):
        step1_transcribe.pick_files(files, ["3.9-V9"])


def test_step1_shows_the_lecture_title_from_the_catalog(folders, fake_whisper, capsys):
    (config.VIDEO_DIR / "3.2-V5.mp4").write_bytes(b"x")
    assert step1_transcribe.main(["--only", "3.2-V5"]) == 0
    assert "[1/1] 3.2-V5.mp4  Thresholding ..." in capsys.readouterr().out


# ---- the two Whisper engines
def test_the_engine_is_chosen_by_gpu_and_installed_packages(monkeypatch):
    gpu = {"on": False}
    monkeypatch.setattr(step1_transcribe, "has_gpu", lambda: gpu["on"])
    installed = {"faster_whisper"}
    monkeypatch.setattr(step1_transcribe, "can_import", lambda module: module in installed)
    assert step1_transcribe.choose_engine("auto") == "faster-whisper"          # CPU + installed: the faster one
    gpu["on"] = True
    assert step1_transcribe.choose_engine("auto") == "openai-whisper"          # a GPU runs openai-whisper well
    gpu["on"] = False
    installed.clear()
    assert step1_transcribe.choose_engine("auto") == "openai-whisper"          # faster-whisper not installed
    assert step1_transcribe.choose_engine("faster-whisper") == "faster-whisper"   # an explicit choice always wins
    assert step1_transcribe.choose_engine("openai-whisper") == "openai-whisper"


def test_faster_whisper_gives_the_same_kind_of_result(fake_faster_whisper):
    model = step1_transcribe.FasterWhisperModel("small.en")
    instance = fake_faster_whisper.instances[0]
    assert (instance.name, instance.device, instance.compute_type) == ("small.en", "cpu", "int8")
    segments = step1_transcribe.transcribe_one(model, Path("x.mp4"), "en", "transcribe", False, prompt="roc curve")
    assert segments == [{"start": 0.0, "end": 3.0, "text": "Hello there."},
                        {"start": 3.0, "end": 6.5, "text": "Second line."}]
    assert instance.calls == [("x.mp4", "en", "transcribe", "roc curve")]      # extra options like fp16 are dropped


def test_faster_whisper_shows_how_far_it_is_and_wipes_the_line_afterwards(fake_faster_whisper, capsys):
    step1_transcribe.FasterWhisperModel("small.en").transcribe("x.mp4", language="en")
    out = capsys.readouterr().out
    assert "transcribing ... 0:03 of 0:06" in out and "transcribing ... 0:06 of 0:06" in out    # it moves with the speech
    assert out.endswith("\r")                                                   # the last progress line is wiped again


def test_the_progress_line_still_works_when_the_video_length_is_unknown(fake_faster_whisper, monkeypatch, capsys):
    original = fake_faster_whisper.transcribe

    def without_length(self, *args, **kwargs):
        pieces, _info = original(self, *args, **kwargs)
        return pieces, types.SimpleNamespace()                                  # no .duration

    monkeypatch.setattr(fake_faster_whisper, "transcribe", without_length)
    result = step1_transcribe.FasterWhisperModel("small.en").transcribe("x.mp4")
    out = capsys.readouterr().out
    assert "transcribing ... 0:03" in out and " of " not in out and len(result["segments"]) == 2


def test_step1_switches_off_the_hugging_face_symlink_warning():
    env = {name: value for name, value in os.environ.items() if name != "HF_HUB_DISABLE_SYMLINKS_WARNING"}
    code = "import os, step1_transcribe; print(os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'])"
    run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert run.stdout.strip() == "1", run.stderr


def test_step1_with_faster_whisper_needs_no_ffmpeg(folders, fake_faster_whisper, capsys):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    assert step1_transcribe.main(["--engine", "faster-whisper"]) == 0
    assert rag.load_transcripts(config.TRANSCRIPT_DIR)["01_a"][0]["text"] == "Hello there."
    assert json.loads((config.TRANSCRIPT_DIR / "01_a.json").read_text(encoding="utf-8"))["engine"] == "faster-whisper"
    assert "faster-whisper" in capsys.readouterr().out
    # 'auto' picks faster-whisper by itself when it is the only engine installed
    assert step1_transcribe.main(["--force"]) == 0 and len(fake_faster_whisper.instances) == 2


def test_step1_explains_a_missing_engine_or_ffmpeg(folders, monkeypatch, capsys):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    assert step1_transcribe.main(["--engine", "faster-whisper"]) == 1
    assert "pip install faster-whisper" in capsys.readouterr().out
    set_ffmpeg(monkeypatch, False)
    assert step1_transcribe.main(["--engine", "openai-whisper"]) == 1
    out = capsys.readouterr().out
    assert "ffmpeg was not found" in out and "faster-whisper" in out            # and it names the way around it


# ---- can faster-whisper read audio here? (av 19 broke faster-whisper 1.2.1: every video failed)
def test_the_audio_self_test_decodes_a_real_wav_file_and_cleans_up(fake_audio):
    assert step1_transcribe.faster_whisper_audio_error() is None
    path, header, rate = fake_audio.files[0]
    assert path.endswith(".wav") and header == b"RIFF" and rate == 16000
    assert not Path(path).exists()                                              # the temporary file is gone again


def test_the_audio_self_test_is_skipped_when_there_is_nothing_to_test(fake_faster_whisper):
    assert step1_transcribe.faster_whisper_audio_error() is None                # no faster_whisper.audio here


def test_step1_stops_before_the_big_model_loads_when_av_is_too_new(folders, fake_faster_whisper, fake_audio, capsys):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    fake_audio.error = AV_19_ERROR
    assert step1_transcribe.main(["--engine", "faster-whisper"]) == 1
    out = capsys.readouterr().out
    assert "TypeError" in out and "metadata_errors" in out
    assert 'pip install "av<19"' in out                                         # the fix is named
    assert fake_faster_whisper.instances == []                                  # the model was never loaded
    assert not config.TRANSCRIPT_DIR.exists()


def test_an_unknown_audio_problem_gets_general_advice_not_the_av_fix(folders, fake_faster_whisper, fake_audio, capsys):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    fake_audio.error = OSError("a DLL is missing")
    assert step1_transcribe.main(["--engine", "faster-whisper"]) == 1
    out = capsys.readouterr().out
    assert "OSError: a DLL is missing" in out and "--engine openai-whisper" in out
    assert "av<19" not in out                                                   # wrong advice would be worse than none


def test_step1_runs_normally_when_the_audio_self_test_passes(folders, fake_faster_whisper, fake_audio, capsys):
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    assert step1_transcribe.main(["--engine", "faster-whisper"]) == 0
    assert len(fake_audio.files) == 1 and len(fake_faster_whisper.instances) == 1


# ---- "does this transcript belong to its title?"
def test_title_warning_only_appears_when_none_of_the_expected_words_are_said(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("video_id,title,archive_file,check_regex\n"
                       "3.2-V5,Thresholding,a.mp4,threshold|sensitivity\n"
                       "3.2-V6,ROC Curves,b.mp4,\n"
                       "3.2-V7,Broken,c.mp4,(unclosed\n", encoding="utf-8")
    monkeypatch.setattr(config, "CATALOG_FILE", catalog)
    right = [{"start": 0, "end": 4, "text": "The Threshold decides who gets a positive prediction."}]    # capital T
    wrong = [{"start": 0, "end": 4, "text": "Today we talk about the weather."}]
    assert step1_transcribe.title_warning("3.2-V5", right) is None
    warning = step1_transcribe.title_warning("3.2-V5", wrong)
    assert "WARNING" in warning and "Thresholding" in warning and "threshold|sensitivity" in warning
    assert step1_transcribe.title_warning("3.2-V6", wrong) is None             # no expected words: nothing to check
    assert step1_transcribe.title_warning("3.2-V7", wrong) is None             # a broken pattern never crashes
    assert step1_transcribe.title_warning("not-in-catalog", wrong) is None


def test_step1_prints_the_warning_for_a_video_with_the_wrong_content(folders, fake_whisper, capsys):
    (config.VIDEO_DIR / "3.2-V5.mp4").write_bytes(b"x")                        # the fake model says "Hello there ..."
    assert step1_transcribe.main(["--only", "3.2-V5"]) == 0
    assert "WARNING" in capsys.readouterr().out


# ================================================================ step 2: index
def test_step2_builds_and_saves_the_index(folders, monkeypatch, transcripts, capsys):
    for video, segments in transcripts.items():
        rag.save_transcript(config.TRANSCRIPT_DIR, video, segments, {})
    monkeypatch.setattr(rag, "embed_texts", fake_embed)
    assert step2_build_index.main([]) == 0
    index = rag.load_index(config.INDEX_DIR)
    assert index.settings["n_videos"] == 3
    assert index.settings["n_chunks"] == len(index.chunks) > 3
    assert index.settings["embed_model"] == config.EMBED_MODEL
    assert index.settings["dimensions"] == index.matrix.shape[1]
    out = capsys.readouterr().out
    assert "chunks" in out and "embedded" in out and "(100%)" in out           # it shows how far it is


def test_step2_problems_are_explained(folders, monkeypatch, transcripts, capsys):
    assert step2_build_index.main([]) == 1
    assert "Run step1_transcribe.py first" in capsys.readouterr().out
    rag.save_transcript(config.TRANSCRIPT_DIR, "01_x", transcripts["01_precision_recall"], {})
    monkeypatch.setattr(rag, "embed_texts", ollama_down)
    assert step2_build_index.main([]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().out
    assert not (config.INDEX_DIR / "embeddings.npy").exists()


def test_step2_help_only_shows_the_help_and_builds_nothing(folders, monkeypatch, transcripts, capsys):
    rag.save_transcript(config.TRANSCRIPT_DIR, "01_x", transcripts["01_precision_recall"], {})
    monkeypatch.setattr(rag, "embed_texts", ollama_down)                        # a build would stop with an Ollama error
    with pytest.raises(SystemExit) as stop:
        step2_build_index.main(["--help"])
    assert stop.value.code == 0 and "usage:" in capsys.readouterr().out
    assert not config.INDEX_DIR.exists()


def test_step2_refuses_options_it_does_not_have(folders, capsys):
    with pytest.raises(SystemExit) as stop:
        step2_build_index.main(["--force"])                                     # that is a step 1 option
    assert stop.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


# ================================================================ ask.py
@pytest.fixture
def ready(folders, monkeypatch, index):
    rag.save_index(config.INDEX_DIR, index)
    monkeypatch.setattr(rag, "embed_texts", fake_embed)
    monkeypatch.setattr(rag, "ask_llm", lambda prompt: "MOCK ANSWER")
    return index


def test_ask_shows_answer_and_sources(ready, capsys):
    assert ask.main(["What", "is", "overfitting?", "--min-score", "0.1", "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "Found in the videos" in out and "MOCK ANSWER" in out
    assert "Overfitting" in out and "score" in out
    assert out.count("score 0.") >= 2                        # two passages listed


def test_ask_says_not_found_for_unrelated_questions(ready, capsys):
    assert ask.main(["What is the capital of France?", "--min-score", "0.1"]) == 0
    out = capsys.readouterr().out
    assert "Not found in the videos" in out and "MOCK ANSWER" not in out


def test_ask_without_llm_and_with_keywords(ready, capsys):
    assert ask.main(["difference between inner join and left join", "--no-llm", "--method", "keywords",
                     "--min-score", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "Found in the videos" in out and "Answer" not in out and "Sql joins" in out


def test_ask_interactive_mode(ready, capsys, monkeypatch):
    typed = iter(["what is overfitting", "", "never reached"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(typed))
    assert ask.main(["--min-score", "0.1"]) == 0
    assert capsys.readouterr().out.count("Found in the videos") == 1


def test_ask_explains_missing_index_and_ollama_problems(folders, monkeypatch, capsys):
    assert ask.main(["anything"]) == 1
    assert "step2_build_index" in capsys.readouterr().out


def test_ask_reports_ollama_down(ready, monkeypatch, capsys):
    monkeypatch.setattr(rag, "embed_texts", ollama_down)
    assert ask.main(["What is overfitting?"]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().out


def test_ask_refuses_an_index_from_another_model(ready, monkeypatch, capsys):
    monkeypatch.setattr(config, "EMBED_MODEL", "other-model")
    assert ask.main(["What is overfitting?"]) == 1
    assert "other-model" in capsys.readouterr().out
    assert ask.main(["overfitting", "--method", "keywords", "--no-llm", "--min-score", "0.01"]) == 0     # keywords: fine


# ================================================================ evaluate.py: questions
def write_csv(path, text, encoding="utf-8"):
    path.write_bytes(text.encode(encoding))
    return path


DURATIONS = {"01_precision_recall": 48.0, "02_overfitting": 48.0, "03_sql_joins": 36.0}
HEADER = "question,video,answer_start,answer_end\n"


def test_load_questions(questions_csv):
    questions = evaluate.load_questions(questions_csv, DURATIONS)
    assert len(questions) == len(QUESTIONS)
    assert questions["in_scope"].sum() == 8
    first = questions.iloc[0]
    assert (first.video, first.start, first.end) == ("01_precision_recall", 16.0, 24.0)
    assert questions.iloc[-1].video == "" and not questions.iloc[-1].in_scope


def test_load_questions_accepts_minutes_extensions_blank_lines_and_extra_columns(tmp_path):
    path = write_csv(tmp_path / "q.csv", "Question, Video ,answer_start,answer_end,notes\n"
                     "What is overfitting?,02_overfitting.mp4,0:05,0:12,my note\n\n"
                     "Capital of France?,,,,\n")
    questions = evaluate.load_questions(path, DURATIONS)
    assert questions.iloc[0].video == "02_overfitting" and (questions.iloc[0].start, questions.iloc[0].end) == (5, 12)
    assert len(questions) == 2


def test_load_questions_ignores_capital_letters_in_video_names(tmp_path):
    durations = {"3.2-V5": 600.0, "3.2-V6": 600.0}
    path = write_csv(tmp_path / "q.csv", HEADER + "What is a threshold?,3.2-v5,0:10,0:40\n"
                     "What is an ROC curve?,3.2-V6.MP4,1:00,1:30\n")
    questions = evaluate.load_questions(path, durations)
    assert questions["video"].tolist() == ["3.2-V5", "3.2-V6"]
    with pytest.raises(ValueError, match="no video called '3.2-v9'"):
        evaluate.load_questions(write_csv(tmp_path / "bad.csv", HEADER + "q?,3.2-v9,0:10,0:40\n"), durations)


def test_load_questions_handles_excel_encodings(tmp_path):
    text = HEADER + "What is caf\u00e9 data?,02_overfitting,1,5\n"
    for encoding in ("utf-8-sig", "cp1252"):
        questions = evaluate.load_questions(write_csv(tmp_path / f"{encoding}.csv", text, encoding), DURATIONS)
        assert questions.iloc[0].question == "What is caf\u00e9 data?"


@pytest.mark.parametrize("row, message", [
    ("q?,nope,0:05,0:10", "no video called 'nope'"),
    ("q?,02_overfitting,,", "fill in answer_start and answer_end"),
    ("q?,02_overfitting,0:20,0:10", "must come after"),
    ("q?,02_overfitting,0:75,0:80", "Cannot read the time"),
    ("q?,02_overfitting,5000,5100", "after the end"),
    ("q?,,0:05,0:10", "video name is empty"),
])
def test_load_questions_explains_mistakes(tmp_path, row, message):
    path = write_csv(tmp_path / "q.csv", HEADER + row + "\n")
    with pytest.raises(ValueError, match=message) as error:
        evaluate.load_questions(path, DURATIONS)
    assert "Row 2" in str(error.value)                       # tells you which row to fix


def test_load_questions_needs_the_right_columns(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        evaluate.load_questions(write_csv(tmp_path / "q.csv", "question,video\nq?,\n"), DURATIONS)
    with pytest.raises(FileNotFoundError):
        evaluate.load_questions(tmp_path / "missing.csv", DURATIONS)


def test_the_starter_questions_file_is_valid_and_all_unanswerable():
    questions = evaluate.load_questions(ROOT / "eval" / "questions.csv", DURATIONS)
    assert len(questions) >= 10 and not questions["in_scope"].any()


# ================================================================ evaluate.py: metrics
def test_recall_at_k():
    ranks = [1, 3, None, 2, None]
    answerable = [True, True, True, True, False]            # the last one is not answerable: ignored
    assert evaluate.recall_at_k(ranks, answerable, 1) == 0.25
    assert evaluate.recall_at_k(ranks, answerable, 3) == 0.75
    assert np.isnan(evaluate.recall_at_k([None], [False], 3))


def test_recall_counts_give_the_numbers_behind_the_percentage():
    ranks = [1, 3, None, 2, None]
    answerable = [True, True, True, True, False]
    assert evaluate.recall_counts(ranks, answerable, 3) == (3, 4)
    assert evaluate.recall_counts(ranks, answerable, 1) == (1, 4)
    assert evaluate.recall_counts([None], [False], 3) == (0, 0)


def test_random_recall_matches_hand_calculations():
    assert evaluate.random_recall(10, 1, 3) == pytest.approx(0.3)          # 1 - 9/10 * 8/9 * 7/8
    assert evaluate.random_recall(10, 2, 1) == pytest.approx(0.2)
    assert evaluate.random_recall(4, 4, 1) == 1.0                          # every chunk is a right one
    assert evaluate.random_recall(10, 8, 3) == 1.0                         # only 2 wrong chunks: 3 picks must hit
    assert evaluate.random_recall(10, 0, 5) == 0.0                         # nothing to find
    assert evaluate.random_recall(3, 1, 5) == 1.0                          # k is bigger than the number of chunks
    assert evaluate.random_recall(0, 0, 3) == 0.0                          # empty index


@pytest.mark.parametrize("n_chunks, n_correct, k", [(50, 3, 5), (200, 4, 3), (30, 10, 2), (12, 1, 5)])
def test_random_recall_agrees_with_a_simulation(n_chunks, n_correct, k):
    rng = np.random.default_rng(1)
    trials = 20000
    picks = np.argsort(rng.random((trials, n_chunks)), axis=1)[:, :k]      # k different random chunks per trial
    simulated = (picks < n_correct).any(axis=1).mean()                     # chunks 0..n_correct-1 are the right ones
    assert simulated == pytest.approx(evaluate.random_recall(n_chunks, n_correct, k), abs=0.015)


def test_baseline_recall_averages_over_the_answerable_questions_only():
    expected = (evaluate.random_recall(10, 1, 3) + evaluate.random_recall(10, 2, 3)) / 2
    assert evaluate.baseline_recall([1, 2, 5, 0], [True, True, False, False], 10, 3) == pytest.approx(expected)
    assert np.isnan(evaluate.baseline_recall([0], [False], 10, 3))


def test_wilson_interval_matches_scipy_and_stays_between_0_and_1():
    from scipy.stats import binomtest
    for successes, total in [(8, 10), (0, 10), (10, 10), (1, 40), (33, 40), (57, 60)]:
        low, high = evaluate.wilson_interval(successes, total)
        reference = binomtest(successes, total).proportion_ci(confidence_level=0.95, method="wilson")
        assert (low, high) == pytest.approx((reference.low, reference.high), abs=1e-3)
        assert 0 <= low <= successes / total <= high <= 1
    assert all(np.isnan(evaluate.wilson_interval(0, 0)))


def test_more_questions_give_a_narrower_interval():
    small, large = evaluate.wilson_interval(8, 10), evaluate.wilson_interval(80, 100)
    assert large[1] - large[0] < small[1] - small[0]


def test_percent_with_range():
    assert evaluate.percent_with_range(0.78, (0.65, 0.88)) == "78% (65-88)"
    assert evaluate.percent_with_range(float("nan"), (float("nan"), float("nan"))) == "n/a"


def test_best_threshold_sits_between_the_two_groups():
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    answerable = np.array([True, True, True, False, False])
    threshold = evaluate.best_threshold(scores, answerable)
    assert 0.2 < threshold < 0.7
    assert evaluate.split_rates(scores, answerable, threshold) == (1.0, 1.0)


def test_best_threshold_with_overlapping_groups():
    scores = np.array([0.9, 0.8, 0.45, 0.4, 0.3, 0.2])
    answerable = np.array([True, True, False, True, False, False])
    threshold = evaluate.best_threshold(scores, answerable)
    answered, refused = evaluate.split_rates(scores, answerable, threshold)
    assert (answered + refused) / 2 >= 0.83                  # the best possible here is 5 of 6 correct


def test_cross_validated_balance():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0.6, 0.9, 30), rng.uniform(0.1, 0.4, 10)])
    answerable = np.array([True] * 30 + [False] * 10)
    mean, std = evaluate.cross_validated_balance(scores, answerable)
    assert mean == pytest.approx(1.0) and std == pytest.approx(0.0)
    assert evaluate.cross_validated_balance(scores[:31], answerable[:31]) is None        # only 1 unanswerable question


def test_cross_validated_balance_is_lower_for_overlapping_scores():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0.3, 0.7, 30), rng.uniform(0.2, 0.6, 10)])
    answerable = np.array([True] * 30 + [False] * 10)
    mean, _ = evaluate.cross_validated_balance(scores, answerable)
    assert 0.5 <= mean < 0.9


# ================================================================ evaluate.py: whole run
@pytest.fixture
def questions(questions_csv):
    return evaluate.load_questions(questions_csv, DURATIONS)


def test_evaluate_method_finds_the_labelled_passages(questions, index):
    for method in rag.METHODS:
        run = evaluate.evaluate_method(questions, index, method, embed_fn=fake_embed)
        assert len(run["ranks"]) == len(questions) and run["scores"].shape == (len(questions), len(index.chunks))
        answerable = questions["in_scope"].to_numpy()
        assert evaluate.recall_at_k(run["ranks"], answerable, 3) >= 0.75
        assert all(rank is None for rank, flag in zip(run["ranks"], answerable) if not flag)
        in_scores, out_scores = run["top_scores"][answerable], run["top_scores"][~answerable]
        assert in_scores.min() > out_scores.max()            # answerable questions score higher


def test_time_ranges_overlap_only_when_they_share_a_moment():
    assert evaluate.overlaps(0, 10, 5, 15) and evaluate.overlaps(5, 15, 0, 10)
    assert evaluate.overlaps(0, 30, 10, 12)                  # one range inside the other
    assert not evaluate.overlaps(0, 10, 10, 20)              # touching at 10 is not enough
    assert not evaluate.overlaps(0, 10, 11, 20)


def test_overlaps_also_works_on_whole_arrays_at_once():
    starts, ends = np.array([0.0, 5.0, 10.0, 20.0]), np.array([10.0, 15.0, 20.0, 30.0])
    assert evaluate.overlaps(starts, ends, 8.0, 12.0).tolist() == [True, True, True, False]


def test_evaluate_method_counts_how_many_chunks_are_right(index):
    two = pd.DataFrame([{"question": "What is overfitting?", "video": "02_overfitting", "start": 0.0, "end": 48.0,
                         "in_scope": True},
                        {"question": "What is the capital of France?", "video": "", "start": np.nan, "end": np.nan,
                         "in_scope": False}])
    run = evaluate.evaluate_method(two, index, "keywords", fake_embed)
    assert run["n_correct"] == [int((index.chunks["video"] == "02_overfitting").sum()), 0]


def test_a_hit_needs_the_right_video_and_overlapping_time(index):
    one = pd.DataFrame([{"question": "What is overfitting?", "video": "01_precision_recall",
                         "start": 0.0, "end": 10.0, "in_scope": True}])

    def rank():                                              # looking only at the very best result
        return evaluate.evaluate_method(one, index, "keywords", fake_embed, max_k=1)["ranks"]

    assert rank() == [None]                                  # wrong video
    one["video"], one["start"], one["end"] = "02_overfitting", 100.0, 110.0
    assert rank() == [None]                                  # right video, wrong time
    one["start"], one["end"] = 0.0, 48.0
    assert rank() == [1]                                     # right video, answer covers the whole video


def test_run_evaluation_creates_all_outputs(tmp_path, transcripts, questions):
    summary = evaluate.run_evaluation(transcripts, questions, fake_embed, tmp_path / "img", tmp_path / "ev")
    assert (summary["n_answerable"], summary["n_unanswerable"], summary["n_videos"]) == (8, 3, 3)
    assert set(summary["methods"]) == {"embeddings", "keywords"}
    assert [row["chunk_seconds"] for row in summary["sweep"]] == [15, 30, 60, 90]
    assert all(0 <= row["recall@5"] <= 1 for row in summary["sweep"])
    chunks = [row["n_chunks"] for row in summary["sweep"]]
    assert chunks == sorted(chunks, reverse=True) and chunks[0] > chunks[-1]       # shorter chunks -> more chunks
    for method in rag.METHODS:
        assert 0 < summary["cutoffs"][method]["value"] < 1
    for name in ("retrieval_methods.png", "chunk_size.png", "not_found_scores.png"):
        assert (tmp_path / "img" / name).stat().st_size > 5000
    assert (tmp_path / "ev" / "failures.csv").exists()
    text = (tmp_path / "ev" / "results.md").read_text(encoding="utf-8")
    assert text == summary["markdown"]
    for part in ("Recall@1", "Keywords (TF-IDF)", "Meaning (bge-m3 embeddings)", "cut-off", "chunk_size.png",
                 "Random guessing", "95% confidence interval", "images/retrieval_methods.png"):
        assert part in text
    assert "nan" not in text.lower().replace("balance", "")


def hand_made_summary():
    sweep = [{"chunk_seconds": 15, "overlap_seconds": 3, "n_chunks": 800, "recall@1": 0.50, "recall@3": 0.70,
              "recall@5": 0.80, "random@3": 0.02},
             {"chunk_seconds": 30, "overlap_seconds": 5, "n_chunks": 400, "recall@1": 0.55, "recall@3": 0.80,
              "recall@5": 0.90, "random@3": 0.03}]
    return {
        "n_answerable": 40, "n_unanswerable": 20, "n_videos": 36, "minutes": 171.0,
        "chunk_seconds": 30, "overlap_seconds": 5, "n_chunks": 400,
        "baseline": {1: 0.01, 3: 0.03, 5: 0.05},
        "methods": {"keywords": {1: 0.40, 3: 0.60, 5: 0.70}, "embeddings": {1: 0.55, 3: 0.80, 5: 0.90}},
        "intervals": {"keywords": {1: (0.26, 0.55), 3: (0.45, 0.74), 5: (0.55, 0.82)},
                      "embeddings": {1: (0.40, 0.69), 3: (0.65, 0.89), 5: (0.77, 0.95)}},
        "ci_points": 12.4, "sweep": sweep, "cutoffs": {},
    }


def test_each_number_of_the_readme_table_sits_in_its_own_row():
    lines = evaluate.results_markdown(hand_made_summary()).splitlines()

    def row(start):
        return next(line for line in lines if line.startswith(start))

    assert row("| Random guessing (calculated)") == "| Random guessing (calculated) | 1% | 3% | 5% |"
    assert row("| Keywords") == "| Keywords (TF-IDF) | 40% (26-55) | 60% (45-74) | 70% (55-82) |"
    assert row("| Meaning") == f"| Meaning ({config.EMBED_MODEL} embeddings) | 55% (40-69) | 80% (65-89) | 90% (77-95) |"
    assert row("| 15 s") == "| 15 s (overlap 3 s) | 800 | 50% | 70% | 80% | 2% |"
    assert row("| 30 s") == "| 30 s (overlap 5 s) | 400 | 55% | 80% | 90% | 3% |"
    text = "\n".join(lines)
    assert "40 questions that the 36 lecture videos answer (171 minutes" in text
    assert "20 questions that they cannot answer" in text and "+/-12 points" in text


def test_every_number_comes_with_a_yardstick(tmp_path, transcripts, questions):
    summary = evaluate.run_evaluation(transcripts, questions, fake_embed, tmp_path / "img", tmp_path / "ev")
    for method in rag.METHODS:
        for k in evaluate.KS:
            low, high = summary["intervals"][method][k]
            assert 0 <= low <= summary["methods"][method][k] <= high <= 1
            assert summary["methods"][method][k] > summary["baseline"][k]          # real search beats luck
    baseline = [summary["baseline"][k] for k in evaluate.KS]
    assert baseline == sorted(baseline) and 0 < baseline[0] < baseline[-1] < 1      # more guesses, more luck
    assert summary["ci_points"] > 0
    assert [row["random@3"] for row in summary["sweep"]] == sorted(row["random@3"] for row in summary["sweep"])


def test_run_evaluation_tells_what_it_is_doing(tmp_path, transcripts, questions):
    messages = []
    evaluate.run_evaluation(transcripts, questions, fake_embed, tmp_path / "img", tmp_path / "ev", log=messages.append)
    text = "\n".join(messages)
    assert "main search (30 s)" in text and "chunk length 15 s" in text and "chunk length 90 s" in text
    assert "embedded" in text and "(100%)" in text
    quiet = evaluate.run_evaluation(transcripts, questions, fake_embed, tmp_path / "img2", tmp_path / "ev2")
    assert quiet["n_chunks"] == len(rag.build_index(transcripts, 30, 5, fake_embed).chunks)   # no log: same result


def test_run_evaluation_without_unanswerable_questions(tmp_path, transcripts, questions):
    only_answerable = questions[questions["in_scope"]].reset_index(drop=True)
    summary = evaluate.run_evaluation(transcripts, only_answerable, fake_embed, tmp_path / "img", tmp_path / "ev")
    assert summary["cutoffs"] == {} and "Add questions that the videos cannot answer" in summary["markdown"]
    assert not (tmp_path / "img" / "not_found_scores.png").exists()


def test_failures_list_the_questions_that_missed(tmp_path, transcripts, questions):
    questions.loc[0, ["start", "end"]] = [200.0, 210.0]       # label the answer where nothing is said
    summary = evaluate.run_evaluation(transcripts, questions, fake_embed, tmp_path / "img", tmp_path / "ev")
    failures = summary["failures"]
    assert len(failures) >= 1 and failures.iloc[0]["question"] == QUESTIONS[0][0]
    assert failures.iloc[0]["first_correct_rank"] == "not in top 5"
    saved = pd.read_csv(tmp_path / "ev" / "failures.csv")
    assert list(saved.columns) == ["question", "expected", "first_correct_rank", "top_results"]


def test_write_into_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{evaluate.README_START}\nold text \\1 \\g<0>\n{evaluate.README_END}\n\nrest\n",
                      encoding="utf-8")
    evaluate.write_into_readme(readme, "NEW TABLE \\1 | x |\n")
    text = readme.read_text(encoding="utf-8")
    assert "NEW TABLE \\1 | x |" in text and "old text" not in text
    assert text.startswith("# Title") and text.rstrip().endswith("rest")
    evaluate.write_into_readme(readme, "SECOND")
    assert text.count(evaluate.README_START) == 1 == readme.read_text(encoding="utf-8").count(evaluate.README_START)
    readme.write_text("no markers here", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        evaluate.write_into_readme(readme, "x")


def test_evaluate_main_end_to_end(folders, transcripts, questions_csv, monkeypatch, capsys):
    for video, segments in transcripts.items():
        rag.save_transcript(config.TRANSCRIPT_DIR, video, segments, {})
    config.EVAL_DIR.mkdir()
    (config.EVAL_DIR / "questions.csv").write_text(questions_csv.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", folders)
    (folders / "README.md").write_text(f"{evaluate.README_START}\n{evaluate.README_END}\n", encoding="utf-8")
    assert evaluate.main(["--write-readme"], embed_fn=fake_embed) == 0
    out = capsys.readouterr().out
    assert "Recall@3" in out and "'Not found' cut-off" in out and "README.md updated" in out
    assert "Recall@1" in (folders / "README.md").read_text(encoding="utf-8")


def test_evaluate_main_explains_missing_pieces(folders, transcripts, monkeypatch, capsys):
    assert evaluate.main([], embed_fn=fake_embed) == 1
    assert "Run step1_transcribe.py first" in capsys.readouterr().out
    rag.save_transcript(config.TRANSCRIPT_DIR, "01_precision_recall", transcripts["01_precision_recall"], {})
    assert evaluate.main([], embed_fn=fake_embed) == 1
    assert "questions.csv not found" in capsys.readouterr().out
    config.EVAL_DIR.mkdir()
    (config.EVAL_DIR / "questions.csv").write_text(
        (ROOT / "eval" / "questions.csv").read_text(encoding="utf-8"), encoding="utf-8")
    assert evaluate.main([], embed_fn=fake_embed) == 1
    assert "no answerable questions yet" in capsys.readouterr().out


# ================================================================ check_setup.py
def test_check_setup_runs_and_gives_a_next_step(folders, monkeypatch, capsys):
    monkeypatch.setattr(rag, "ollama_status", lambda url=None: (True, ["bge-m3:latest"]))
    check_setup.main()
    out = capsys.readouterr().out
    assert "[OK] Ollama running" in out and "[MISSING] LLM 'llama3.2'" in out and "ollama pull llama3.2" in out
    assert "[OK] catalog.csv" in out and "lectures" in out
    assert "Next step:" in out and "python step0_download_videos.py" in out         # nothing downloaded yet
    (config.VIDEO_DIR / "01_a.mp4").write_bytes(b"x")
    check_setup.main()
    assert "python step1_transcribe.py --only 3.2-V5" in capsys.readouterr().out


def test_check_setup_explains_the_two_whisper_engines(folders, monkeypatch, capsys):
    monkeypatch.setattr(rag, "ollama_status", lambda url=None: (False, []))
    monkeypatch.setitem(sys.modules, "whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    set_ffmpeg(monkeypatch, False)
    check_setup.main()
    out = capsys.readouterr().out
    assert "[MISSING] a Whisper engine" in out and "pip install faster-whisper" in out
    assert "[MISSING] ffmpeg" in out                       # with no engine installed, openai-whisper is the default one

    fake_module = types.ModuleType("faster_whisper")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    check_setup.main()
    out = capsys.readouterr().out
    assert "[OK] a Whisper engine" in out and "faster-whisper: installed" in out
    assert "ffmpeg" not in out.split("Ollama")[0]          # faster-whisper does not need ffmpeg: not even mentioned
    assert "No GPU found" in out


def test_check_setup_tests_that_faster_whisper_can_read_audio(folders, monkeypatch, capsys, fake_audio):
    monkeypatch.setattr(rag, "ollama_status", lambda url=None: (False, []))
    check_setup.main()
    assert "[OK] faster-whisper can read audio files" in capsys.readouterr().out
    fake_audio.error = AV_19_ERROR                          # installed, but it would fail on every video
    check_setup.main()
    out = capsys.readouterr().out
    assert "[MISSING] faster-whisper can read audio files" in out and 'pip install "av<19"' in out
    assert "[OK] a Whisper engine" in out                   # the engine is there; reading audio is what fails


def test_check_setup_says_which_python_it_runs_on(folders, monkeypatch, capsys):
    monkeypatch.setattr(rag, "ollama_status", lambda url=None: (False, []))
    check_setup.main()
    first_line = capsys.readouterr().out.splitlines()[0]
    assert first_line.startswith("Python 3.") and sys.executable in first_line


# ================================================================ app.py (Streamlit)
def test_the_web_app_never_waits_for_an_email_address():
    """Streamlit asks for an e-mail address at its very first start and waits. This file stops that."""
    text = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "showEmailPrompt = false" in text and "gatherUsageStats = false" in text


@pytest.fixture
def app(folders, monkeypatch, index):
    pytest.importorskip("streamlit")
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_resource.clear()
    rag.save_index(config.INDEX_DIR, index)
    monkeypatch.setattr(rag, "embed_texts", fake_embed)
    monkeypatch.setattr(rag, "ask_llm", lambda prompt: "APP MOCK ANSWER")
    monkeypatch.setattr(rag, "ollama_status", lambda url=None: (True, ["bge-m3:latest", "llama3.2:latest"]))
    monkeypatch.setitem(config.MIN_SCORE, "embeddings", 0.1)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)


def test_app_answers_with_sources(app):
    at = app.run()
    assert not at.exception and not at.error
    at.text_input[0].set_value("What is overfitting?").run()
    assert not at.exception and not at.error and not at.warning
    shown = " ".join(m.value for m in at.markdown)
    assert "APP MOCK ANSWER" in shown and "Overfitting" in shown and "score 0." in shown


def test_app_says_not_found(app):
    at = app.run()
    at.text_input[0].set_value("What is the capital of France?").run()
    assert not at.exception
    assert len(at.warning) == 1 and "Not found in the videos" in at.warning[0].value


def test_app_works_without_ollama(app, monkeypatch):
    monkeypatch.setattr(rag, "ollama_status", lambda url=None: (False, []))
    monkeypatch.setitem(config.MIN_SCORE, "keywords", 0.05)
    at = app.run()
    assert not at.exception and at.radio[0].value == "Keywords (TF-IDF)"
    assert at.checkbox[0].disabled
    at.text_input[0].set_value("difference between inner join and left join").run()
    assert not at.exception and not at.error
    assert "Sql joins" in " ".join(m.value for m in at.markdown)


def test_app_shows_ollama_errors_instead_of_crashing(app, monkeypatch):
    monkeypatch.setattr(rag, "embed_texts", ollama_down)
    at = app.run()
    at.text_input[0].set_value("What is overfitting?").run()
    assert not at.exception and len(at.error) == 1 and "Cannot reach Ollama" in at.error[0].value


def test_app_without_an_index_explains_what_to_do(app, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "nothing")
    at = app.run()
    assert not at.exception and "step2_build_index" in at.error[0].value


@pytest.mark.parametrize("file_name, element", [("02_overfitting.mp4", "video"), ("02_overfitting.m4a", "audio")])
def test_app_offers_a_player_when_the_recording_exists(app, file_name, element):
    (config.VIDEO_DIR / file_name).write_bytes(b"not really a recording")
    at = app.run()
    at.text_input[0].set_value("What is overfitting?").run()
    assert not at.exception
    assert len(at.get(element)) >= 1


def test_app_picks_up_a_rebuilt_index(app, index, transcripts):
    import os
    at = app.run()
    at.text_input[0].set_value("What is overfitting?").run()
    before = " ".join(m.value for m in at.markdown)
    assert "Sql joins" in before or "Precision recall" in before         # 3 videos, and top 4 of 6 chunks are shown

    smaller = rag.build_index({"02_overfitting": transcripts["02_overfitting"]}, 30, 5, embed_fn=fake_embed,
                              settings=index.settings)
    rag.save_index(config.INDEX_DIR, smaller)                              # as if step 2 was run again
    stamp = (config.INDEX_DIR / "embeddings.npy").stat().st_mtime + 5
    os.utime(config.INDEX_DIR / "embeddings.npy", (stamp, stamp))
    at.text_input[0].set_value("What is overfitting? ").run()
    after = " ".join(m.value for m in at.markdown)
    assert not at.exception
    assert "Overfitting" in after and "Sql joins" not in after and "Precision recall" not in after
