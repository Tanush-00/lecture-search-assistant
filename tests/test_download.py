"""Tests for step 0 (downloading the lectures). A small fake archive.org runs on this computer, so no internet is used."""
import hashlib
import json
import os
import shutil
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pytest

import config
import rag
import step0_download_videos as step0

FILES = {                                           # what the fake archive.org holds
    "a.mp4": bytes(range(256)) * 20,                # 5120 bytes
    "b.mp4": bytes(reversed(range(256))) * 13,      # 3328 bytes
    "c.mp4": b"third video " * 100,
}
CATALOG = ("video_id,title,archive_file,size_mb,selected\n"
           "t-V1,First lecture,a.mp4,0.005,1\n"
           "t-V2,Second lecture,b.mp4,0.003,1\n"
           "t-V3,Third lecture,c.mp4,0.001,0\n")


class FakeArchiveHandler(BaseHTTPRequestHandler):
    """Speaks the two parts of archive.org that step 0 uses: /metadata/<item> and /download/<item>/<file>."""

    def send(self, status, body=b"", headers=None):
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        server = self.server
        server.log.append((self.path, self.headers.get("Range")))
        if self.path == "/metadata/TESTITEM":
            if server.metadata_status != 200:
                return self.send(server.metadata_status)
            files = [{"name": name, "size": str(len(data)), "md5": hashlib.md5(data).hexdigest()}
                     for name, data in server.files.items()]
            files.append({"name": "TESTITEM_files.xml"})                     # no size: must be skipped
            return self.send(200, json.dumps({"files": files}).encode(), {"Content-Type": "application/json"})
        prefix = "/download/TESTITEM/"
        name = unquote(self.path[len(prefix):]) if self.path.startswith(prefix) else None
        if name not in server.files:
            return self.send(404)
        if server.fail_next > 0:
            server.fail_next -= 1
            return self.send(500)
        data = server.files[name]
        start, status, headers = 0, 200, {}
        wanted = self.headers.get("Range")
        if wanted and not server.ignore_range:
            if server.range_not_satisfiable_once:
                server.range_not_satisfiable_once = False
                return self.send(416)
            start = int(wanted.split("=")[1].rstrip("-"))
            if start >= len(data):
                return self.send(416)
            status, headers = 206, {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"}
        body = data[start:]
        if name in server.break_after:                                       # promise everything, send part, hang up
            cut = server.break_after.pop(name)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.close_connection = True
            return
        self.send(status, body, headers)

    def log_message(self, *args):
        pass


@pytest.fixture
def archive(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeArchiveHandler)
    server.files = dict(FILES)
    server.log = []
    server.metadata_status = 200
    server.ignore_range = False
    server.fail_next = 0
    server.break_after = {}
    server.range_not_satisfiable_once = False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "ARCHIVE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(config, "ARCHIVE_ITEM", "TESTITEM")
    monkeypatch.setattr(step0.time, "sleep", lambda seconds: None)           # no waiting between retries
    server.downloads = lambda: [entry for entry in server.log if entry[0].startswith("/download/")]
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An empty video folder and a small catalog.csv of three lectures (two of them chosen)."""
    monkeypatch.setattr(config, "VIDEO_DIR", tmp_path / "videos")
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(CATALOG, encoding="utf-8")
    monkeypatch.setattr(config, "CATALOG_FILE", catalog)
    return tmp_path / "videos"


def url_of(name):
    return step0.video_url(name)


# ================================================================ one file
def test_download_file_saves_the_bytes_and_leaves_no_part_file(archive, tmp_path):
    target = tmp_path / "out" / "a.mp4"
    assert step0.download_file(url_of("a.mp4"), target, len(FILES["a.mp4"])) == len(FILES["a.mp4"])
    assert target.read_bytes() == FILES["a.mp4"]
    assert not (tmp_path / "out" / "a.mp4.part").exists()


def test_download_file_works_without_knowing_the_size(archive, tmp_path):
    target = tmp_path / "a.mp4"
    step0.download_file(url_of("a.mp4"), target)
    assert target.read_bytes() == FILES["a.mp4"]


def test_a_half_finished_download_is_continued(archive, tmp_path):
    target = tmp_path / "a.mp4"
    (tmp_path / "a.mp4.part").write_bytes(FILES["a.mp4"][:1000])
    step0.download_file(url_of("a.mp4"), target, len(FILES["a.mp4"]))
    assert target.read_bytes() == FILES["a.mp4"]
    assert archive.downloads() == [("/download/TESTITEM/a.mp4", "bytes=1000-")]      # only the missing part was asked for


def test_a_server_that_ignores_the_range_request_gets_a_fresh_start(archive, tmp_path):
    archive.ignore_range = True
    (tmp_path / "a.mp4.part").write_bytes(b"garbage" * 100)
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4", len(FILES["a.mp4"]))
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"]                       # not "garbage" + the video


def test_a_server_that_ignores_the_range_request_is_also_handled_when_the_size_is_unknown(archive, tmp_path):
    archive.ignore_range = True
    (tmp_path / "a.mp4.part").write_bytes(b"garbage" * 100)
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4")                         # no expected size to fall back on
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"]


class ShortResponse:
    """A reply that promises 5120 bytes, delivers 700 and then simply ends - without any error."""
    status_code = 200
    headers = {"Content-Length": str(len(FILES["a.mp4"]))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield FILES["a.mp4"][:700]


def test_a_reply_that_ends_too_early_is_never_kept_as_the_finished_file(archive, tmp_path, monkeypatch):
    real_get, calls = requests_get(), []

    def get(url, **kwargs):
        calls.append(kwargs.get("headers", {}).get("Range"))
        return ShortResponse() if len(calls) == 1 else real_get(url, **kwargs)

    monkeypatch.setattr(step0.requests, "get", get)
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4")                         # size unknown: Content-Length decides
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"]
    assert calls == [None, "bytes=700-"]                                             # the second try continued at byte 700


def requests_get():
    import requests
    return requests.get


def test_a_part_file_that_does_not_fit_is_thrown_away(archive, tmp_path):
    archive.range_not_satisfiable_once = True                                        # the server says 416 once
    (tmp_path / "a.mp4.part").write_bytes(FILES["a.mp4"][:100])
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4", len(FILES["a.mp4"]))
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"]
    (tmp_path / "b.mp4.part").write_bytes(b"x" * 99999)                              # bigger than the real file
    step0.download_file(url_of("b.mp4"), tmp_path / "b.mp4", len(FILES["b.mp4"]))
    assert (tmp_path / "b.mp4").read_bytes() == FILES["b.mp4"]


def test_a_part_file_that_is_already_complete_is_only_renamed(archive, tmp_path):
    (tmp_path / "a.mp4.part").write_bytes(FILES["a.mp4"])
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4", len(FILES["a.mp4"]))
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"] and archive.downloads() == []


def test_a_connection_that_breaks_in_the_middle_is_continued(archive, tmp_path):
    archive.break_after["a.mp4"] = 700
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4", len(FILES["a.mp4"]))
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"]
    assert len(archive.downloads()) == 2                                             # the first try broke, the second finished


def test_a_connection_that_breaks_without_a_known_size_is_also_caught(archive, tmp_path):
    archive.break_after["a.mp4"] = 700
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4")                         # size from Content-Length only
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"]


def test_a_missing_file_is_reported_at_once_without_retrying(archive, tmp_path):
    with pytest.raises(step0.DownloadError, match="no file at"):
        step0.download_file(url_of("nope.mp4"), tmp_path / "nope.mp4", 10)
    assert len(archive.downloads()) == 1


def test_server_errors_are_retried_and_then_reported(archive, tmp_path):
    archive.fail_next = 2
    step0.download_file(url_of("a.mp4"), tmp_path / "a.mp4", len(FILES["a.mp4"]))    # the third try works
    assert (tmp_path / "a.mp4").read_bytes() == FILES["a.mp4"] and len(archive.downloads()) == 3
    archive.fail_next = 99
    with pytest.raises(step0.DownloadError, match="Could not download b.mp4.*run the same command again"):
        step0.download_file(url_of("b.mp4"), tmp_path / "b.mp4", len(FILES["b.mp4"]), retries=2)
    assert not (tmp_path / "b.mp4").exists()                                         # never a half file under the real name


def test_move_into_place_waits_when_windows_holds_the_file(tmp_path, monkeypatch):
    part, target = tmp_path / "a.mp4.part", tmp_path / "a.mp4"
    part.write_bytes(b"data")
    real_replace, calls = os.replace, []

    def sometimes_locked(source, destination):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("locked by OneDrive")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", sometimes_locked)
    monkeypatch.setattr(step0.time, "sleep", lambda seconds: None)
    step0.move_into_place(part, target)
    assert target.read_bytes() == b"data" and len(calls) == 3

    monkeypatch.setattr(os, "replace", lambda *args: (_ for _ in ()).throw(PermissionError("always")))
    with pytest.raises(step0.DownloadError, match="OneDrive"):
        step0.move_into_place(part, target, tries=2)


# ================================================================ checking
def test_fetch_remote_files_reads_sizes_and_checksums(archive):
    remote = step0.fetch_remote_files()
    assert set(remote) == set(FILES)                                                 # the entry without a size is skipped
    assert remote["a.mp4"] == {"size": len(FILES["a.mp4"]), "md5": hashlib.md5(FILES["a.mp4"]).hexdigest()}


def test_fetch_remote_files_explains_a_failure(archive):
    archive.metadata_status = 500
    with pytest.raises(step0.DownloadError, match="Could not read the file list"):
        step0.fetch_remote_files()


def test_check_file_tells_what_is_wrong(archive, tmp_path):
    remote = step0.fetch_remote_files()["a.mp4"]
    path = tmp_path / "a.mp4"
    assert step0.check_file(path, remote) == "missing"
    path.write_bytes(FILES["a.mp4"])
    assert step0.check_file(path, remote) == "ok"
    assert step0.check_file(path, None) == "not verified"                            # archive.org's list was not available
    path.write_bytes(FILES["a.mp4"][:-1])
    assert step0.check_file(path, remote) == "wrong size"
    path.write_bytes(b"\x00" * len(FILES["a.mp4"]))                                  # same size, other content
    assert step0.check_file(path, remote) == "wrong checksum"


def test_md5_of_reads_big_files_in_blocks(tmp_path):
    path = tmp_path / "big.bin"
    data = os.urandom(3 * (1 << 20) + 17)
    path.write_bytes(data)
    assert step0.md5_of(path) == hashlib.md5(data).hexdigest()


# ================================================================ choosing videos
def test_choose_videos(workspace):
    catalog = rag.catalog()
    assert [row["video_id"] for row in step0.choose_videos(catalog)] == ["t-V1", "t-V2"]            # the chosen ones
    assert [row["video_id"] for row in step0.choose_videos(catalog, everything=True)] == ["t-V1", "t-V2", "t-V3"]
    assert [row["video_id"] for row in step0.choose_videos(catalog, only=["T-V3", "t-v1", "t-V3"])] == ["t-V3", "t-V1"]
    with pytest.raises(ValueError, match=r"'t-V9'.*did you mean"):
        step0.choose_videos(catalog, only=["t-V9"])


# ================================================================ the whole command
def test_main_downloads_the_chosen_videos_and_checks_them(archive, workspace, capsys):
    assert step0.main([]) == 0
    out = capsys.readouterr().out
    assert (workspace / "t-V1.mp4").read_bytes() == FILES["a.mp4"]
    assert (workspace / "t-V2.mp4").read_bytes() == FILES["b.mp4"]
    assert not (workspace / "t-V3.mp4").exists()                                     # not chosen
    assert not list(workspace.glob("*.part"))
    assert "2 of 2 videos are ready" in out and "match the size and MD5 checksum" in out
    assert "CC BY-NC-SA 4.0" in out and "never upload them to GitHub" in out         # the license note is always shown
    assert "python step1_transcribe.py --only 3.2-V5" in out
    assert [r.name for r in rag.find_media_files(workspace)] == ["t-V1.mp4", "t-V2.mp4"]      # step 1 finds them by name


def test_main_downloads_nothing_twice(archive, workspace, capsys):
    assert step0.main([]) == 0
    first_round = len(archive.downloads())
    capsys.readouterr()
    assert step0.main([]) == 0
    assert len(archive.downloads()) == first_round and capsys.readouterr().out.count("already there") == 2
    assert step0.main(["--force", "--only", "t-V1"]) == 0                            # --force: again, even if fine
    assert len(archive.downloads()) == first_round + 1


def test_main_replaces_a_damaged_file(archive, workspace, capsys):
    workspace.mkdir()
    (workspace / "t-V1.mp4").write_bytes(b"\x00" * len(FILES["a.mp4"]))              # right size, wrong content
    assert step0.main(["--only", "t-V1"]) == 0
    assert "wrong checksum" in capsys.readouterr().out
    assert (workspace / "t-V1.mp4").read_bytes() == FILES["a.mp4"]


def test_main_all_downloads_every_video(archive, workspace):
    assert step0.main(["--all"]) == 0
    assert sorted(p.name for p in workspace.iterdir()) == ["t-V1.mp4", "t-V2.mp4", "t-V3.mp4"]


def test_main_check_downloads_nothing_and_reports_problems(archive, workspace, capsys):
    workspace.mkdir()
    (workspace / "t-V1.mp4").write_bytes(FILES["a.mp4"])
    (workspace / "t-V2.mp4").write_bytes(FILES["b.mp4"][:-5])
    assert step0.main(["--check"]) == 1
    out = capsys.readouterr().out
    assert archive.downloads() == []
    assert "t-V1" in out and "ok" in out and "wrong size" in out and "1 of 2 videos are fine" in out
    assert step0.main(["--all", "--check"]) == 1 and "missing" in capsys.readouterr().out
    (workspace / "t-V2.mp4").write_bytes(FILES["b.mp4"])
    assert step0.main(["--check"]) == 0


def test_main_explains_unknown_names_and_a_missing_catalog(archive, workspace, capsys, monkeypatch):
    assert step0.main(["--only", "t-V9"]) == 1
    assert "Not in catalog.csv" in capsys.readouterr().out
    monkeypatch.setattr(config, "CATALOG_FILE", workspace.parent / "nothing.csv")
    assert step0.main([]) == 1
    assert "nothing.csv" in capsys.readouterr().out


def test_main_still_downloads_when_archive_org_has_no_file_list(archive, workspace, capsys):
    archive.metadata_status = 503
    assert step0.main([]) == 0
    out = capsys.readouterr().out
    assert (workspace / "t-V1.mp4").read_bytes() == FILES["a.mp4"]
    assert "Could not read the file list" in out and "--check" in out                # and it says it could not verify


def test_main_reports_files_that_failed_and_keeps_going(archive, workspace, capsys):
    archive.files.pop("a.mp4")                                                       # archive.org lost the first video
    assert step0.main([]) == 1
    out = capsys.readouterr().out
    assert "archive.org does not list the file for: t-V1" in out
    assert "FAILED" in out and "Not finished: t-V1" in out
    assert (workspace / "t-V2.mp4").read_bytes() == FILES["b.mp4"]                   # the other video was still downloaded


def test_main_stops_before_filling_the_disk(archive, workspace, capsys, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: types.SimpleNamespace(total=10**9, used=10**9 - 10**6, free=10**6))
    assert step0.main([]) == 1
    assert "Not enough free disk space" in capsys.readouterr().out
    assert archive.downloads() == []


def test_main_can_be_stopped_with_ctrl_c_and_continued(archive, workspace, capsys, monkeypatch):
    real = step0.download_one
    monkeypatch.setattr(step0, "download_one", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert step0.main([]) == 1
    assert "Run the same command again" in capsys.readouterr().out
    monkeypatch.setattr(step0, "download_one", real)
    assert step0.main([]) == 0


def test_main_warns_when_the_folder_is_inside_onedrive(archive, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "VIDEO_DIR", tmp_path / "OneDrive" / "Desktop" / "videos")
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(CATALOG, encoding="utf-8")
    monkeypatch.setattr(config, "CATALOG_FILE", catalog)
    assert step0.main(["--only", "t-V1"]) == 0
    assert "LECTURE_VIDEO_DIR" in capsys.readouterr().out


def test_the_video_folder_can_be_moved_with_an_environment_variable(tmp_path):
    import subprocess
    import sys
    code = "import config; print(config.VIDEO_DIR)"
    root = Path(config.ROOT)
    env = {**os.environ, "LECTURE_VIDEO_DIR": str(tmp_path / "elsewhere")}
    result = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(root)!r}); {code}"],
                            capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == str(tmp_path / "elsewhere")
    env.pop("LECTURE_VIDEO_DIR")
    result = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(root)!r}); {code}"],
                            capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == str(root / "videos")
