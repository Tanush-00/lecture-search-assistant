"""Step 0: download the lecture videos from the Internet Archive.

    python step0_download_videos.py                 # the 36 videos chosen in catalog.csv (about 400 MB)
    python step0_download_videos.py --only 3.2-V5   # just one video (good for a first trial)
    python step0_download_videos.py --all           # all 82 videos of units 2-5 (about 1 GB)
    python step0_download_videos.py --check         # download nothing, only check the files you already have
    python step0_download_videos.py --force         # download everything again

The lectures are MIT OpenCourseWare 15.071 "The Analytics Edge" (Spring 2017, Prof. Dimitris Bertsimas),
licensed CC BY-NC-SA 4.0. Use them for learning and for this portfolio project, and do NOT upload them to GitHub.

You can stop with Ctrl+C at any time and run the command again: a half-finished download is kept as <name>.mp4.part
and continued, and every finished file is compared with the size and MD5 checksum that archive.org publishes.
"""
import argparse
import difflib
import hashlib
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote

import requests

import config
import rag

CONNECT_TIMEOUT, READ_TIMEOUT = 20, 60       # seconds
SPARE_MB = 200                               # keep this much disk space free after the download
LICENSE_NOTE = ("These lectures: MIT OpenCourseWare 15.071 The Analytics Edge (Spring 2017), Prof. Dimitris Bertsimas, "
                "CC BY-NC-SA 4.0.\nKeep the videos on your laptop and never upload them to GitHub.")


class DownloadError(RuntimeError):
    """A download or a check failed. The message says what to do next."""


class _StartAgain(Exception):
    """The half-finished file did not fit the server's file. It was deleted; try again from byte 0."""


def archive_url(path):
    return f"{config.ARCHIVE_URL.rstrip('/')}/{path}"


def video_url(file_name):
    return archive_url(f"download/{config.ARCHIVE_ITEM}/{quote(file_name)}")


def target_path(video_id):
    return Path(config.VIDEO_DIR) / f"{video_id}.mp4"


# ----------------------------------------------------------------------------
# What archive.org says about its files, and checking our copy against it
# ----------------------------------------------------------------------------
def fetch_remote_files():
    """{file name: {'size': bytes, 'md5': text}} for every file of the archive.org item."""
    url = archive_url(f"metadata/{config.ARCHIVE_ITEM}")
    try:
        response = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        response.raise_for_status()
        files = response.json()["files"]
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        raise DownloadError(f"Could not read the file list from {url} ({error}).") from None
    remote = {}
    for entry in files:
        try:
            remote[entry["name"]] = {"size": int(entry["size"]), "md5": str(entry.get("md5", "")).lower()}
        except (KeyError, ValueError, TypeError):
            continue                          # entries without a size are not videos
    return remote


def md5_of(path, block_size=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, expected):
    """'ok', 'missing', 'wrong size', 'wrong checksum', or 'not verified' (archive.org's list was not available)."""
    path = Path(path)
    if not path.exists():
        return "missing"
    if not expected:
        return "not verified"
    if path.stat().st_size != expected["size"]:
        return "wrong size"
    if expected["md5"] and md5_of(path) != expected["md5"]:
        return "wrong checksum"
    return "ok"


# ----------------------------------------------------------------------------
# Downloading one file (resumable)
# ----------------------------------------------------------------------------
def move_into_place(part, target, tries=10, wait=0.5):
    """Rename name.mp4.part -> name.mp4. On Windows, OneDrive or an antivirus program may hold the new file for a moment."""
    for attempt in range(1, tries + 1):
        try:
            os.replace(part, target)
            return
        except PermissionError:
            if attempt == tries:
                raise DownloadError(
                    f"Windows will not let me rename {Path(part).name}. Close programs that may be using it "
                    "(OneDrive, an antivirus, a video player) and run the same command again - the download is kept."
                ) from None
            time.sleep(wait)


def _stream_to(url, part, have):
    """Write the web page's bytes into the .part file, continuing after `have` bytes if the server allows.
    Returns the size the finished file should have, or None if the server did not say."""
    headers = {"Accept-Encoding": "identity"}      # the size must be the real video size, not a compressed one
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as response:
        if response.status_code == 404:
            raise DownloadError(f"archive.org has no file at {url}")
        if response.status_code == 416:            # "that range does not exist": our .part file is wrong
            Path(part).unlink()
            raise _StartAgain()
        response.raise_for_status()
        resuming = response.status_code == 206 and have > 0     # a plain 200 means: the whole file again
        length = response.headers.get("Content-Length", "")
        total = (have if resuming else 0) + int(length) if length.isdigit() else None
        with open(part, "ab" if resuming else "wb") as file:
            for block in response.iter_content(chunk_size=1 << 20):
                file.write(block)
    return total


def download_file(url, target, expected_size=None, retries=3, wait=2.0):
    """Download `url` to `target` and return the size in bytes.
    The bytes go into <target>.part first, so a stopped download never looks like a finished video.
    A later call continues a .part file. Network errors are retried `retries` times."""
    target = Path(target)
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    problem = ""
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if expected_size is not None and have > expected_size:        # bigger than the real file: it is broken
            part.unlink()
            have = 0
        try:
            if expected_size is None or have < expected_size:
                total = _stream_to(url, part, have)
            else:
                total = expected_size                                 # the .part file is already complete
            size = part.stat().st_size
            want = expected_size if expected_size is not None else total
            if want is not None and size < want:
                problem = f"the connection stopped after {size} of {want} bytes"          # keep the part, continue
            elif want is not None and size > want:
                part.unlink()
                problem = f"got {size} bytes but expected {want}"
            else:
                move_into_place(part, target)
                return size
        except _StartAgain:
            problem = "the half-finished file did not fit; started again"
        except requests.RequestException as error:
            problem = str(error)
        if attempt < retries:
            time.sleep(wait * attempt)
    raise DownloadError(f"Could not download {target.name} ({problem}). "
                        "Check your internet and run the same command again - what is finished is kept.")


# ----------------------------------------------------------------------------
# The command
# ----------------------------------------------------------------------------
def choose_videos(catalog, only=None, everything=False):
    """The catalog rows to work on. `only` = ids typed by the user (any capital letters). Raises ValueError for unknown ids."""
    if only:
        by_lower = {video_id.lower(): video_id for video_id in catalog}
        chosen, unknown = [], []
        for name in only:
            video_id = by_lower.get(name.lower())
            if video_id is None:
                unknown.append(name)
            elif catalog[video_id] not in chosen:
                chosen.append(catalog[video_id])
        if unknown:
            hints = []
            for name in unknown:
                close = difflib.get_close_matches(name, list(catalog), n=3, cutoff=0.5)
                hints.append(f"'{name}'" + (f" (did you mean {', '.join(close)}?)" if close else ""))
            raise ValueError("Not in catalog.csv: " + "; ".join(hints))
        return chosen
    return [row for row in catalog.values() if everything or row["selected"]]


def wanted_megabytes(rows, remote, force=False):
    """About how many MB still have to be downloaded (videos that are missing or have the wrong size; all if `force`)."""
    total = 0.0
    for row in rows:
        target = target_path(row["video_id"])
        expected = remote.get(row["archive_file"])
        if expected:                                            # exact size known
            size, needed = expected["size"], target_size(target) != expected["size"]
        else:                                                   # only the rounded size from catalog.csv
            size, needed = float(row.get("size_mb") or 0) * 1e6, not target.exists()
        if force or needed:
            total += size
    return total / 1e6


def target_size(path):
    return path.stat().st_size if path.exists() else -1


def main(argv=None):
    rag.setup_console()
    parser = argparse.ArgumentParser(description="Download the MIT lecture videos from archive.org.")
    parser.add_argument("--only", nargs="+", metavar="ID", help="video ids to download (like 3.2-V5)")
    parser.add_argument("--all", action="store_true", help="all 82 videos, not only the 36 chosen ones")
    parser.add_argument("--check", action="store_true", help="download nothing; only check the files you have")
    parser.add_argument("--force", action="store_true", help="download again even if the file is already there and fine")
    args = parser.parse_args(argv)

    catalog = rag.catalog()
    if not catalog:
        print(f"{config.CATALOG_FILE.name} was not found or is empty. It must be next to this script.")
        return 1
    try:
        rows = choose_videos(catalog, args.only, args.all)
    except ValueError as error:
        print(error)
        return 1
    if not rows:
        print("No video is marked selected = 1 in catalog.csv. Use --all or --only 3.2-V5")
        return 1

    folder = Path(config.VIDEO_DIR)
    print(f"{len(rows)} videos -> {folder}")
    if "onedrive" in str(folder).lower() and not args.check:
        print("Tip: this folder is inside OneDrive, which would upload every video. To keep them outside, set\n"
              "     LECTURE_VIDEO_DIR to another folder first (see README), for example D:\\lecture_videos")

    remote = {}
    try:
        remote = fetch_remote_files()
    except DownloadError as error:
        print(f"{error}\nI will only check that each download finishes, not its checksum.")

    missing_online = [row["video_id"] for row in rows if remote and row["archive_file"] not in remote]
    if missing_online:
        print(f"archive.org does not list the file for: {', '.join(missing_online)}. "
              "Check ARCHIVE_ITEM in config.py and the archive_file column in catalog.csv.")

    if args.check:
        return check_all(rows, remote)

    folder.mkdir(parents=True, exist_ok=True)
    need_mb = wanted_megabytes(rows, remote, args.force)
    free_mb = shutil.disk_usage(folder).free / 1e6
    if need_mb + SPARE_MB > free_mb:
        print(f"Not enough free disk space: about {need_mb:.0f} MB are needed (plus {SPARE_MB} MB spare), "
              f"{free_mb:.0f} MB are free. Free some space or set LECTURE_VIDEO_DIR to another drive.")
        return 1

    failed, started = [], time.time()
    try:
        for number, row in enumerate(rows, start=1):
            if not download_one(number, len(rows), row, remote.get(row["archive_file"]), args.force):
                failed.append(row["video_id"])
    except KeyboardInterrupt:
        print("\nStopped. Finished files and the half-finished one are kept. Run the same command again to continue.")
        return 1

    print()
    done = len(rows) - len(failed)
    print(f"{done} of {len(rows)} videos are ready in {folder}  ({rag.format_time(time.time() - started)}).")
    if failed:
        print(f"Not finished: {', '.join(failed)}. Run the same command again.")
    elif not remote:
        print("(Sizes and checksums could not be compared with archive.org this time. Run with --check later.)")
    else:
        print("All files match the size and MD5 checksum that archive.org publishes.")
    print(LICENSE_NOTE)
    if not failed:
        print("\nNext: try ONE video first:  python step1_transcribe.py --only 3.2-V5")
    return 1 if failed else 0


def download_one(number, count, row, expected, force):
    """Download one video unless it is already there and correct. Returns True when the file is fine afterwards."""
    video_id, title = row["video_id"], row["title"]
    target = target_path(video_id)
    size_mb = (expected["size"] / 1e6) if expected else float(row.get("size_mb") or 0)
    label = f"[{number}/{count}] {video_id}  {title} ({size_mb:.1f} MB)"
    if not force and target.exists():
        status = check_file(target, expected)
        if status in ("ok", "not verified"):
            print(f"{label}: already there" + (" (not verified)" if status == "not verified" else ""))
            return True
        print(f"{label}: the file you have has a {status} - downloading it again")
    else:
        print(f"{label} ...", flush=True)
    started = time.time()
    try:
        download_file(video_url(row["archive_file"]), target, expected["size"] if expected else None)
    except (DownloadError, OSError) as error:
        print(f"    FAILED: {error}")
        return False
    status = check_file(target, expected)
    if status in ("ok", "not verified"):
        print(f"    done in {time.time() - started:.0f} s" + (", size and checksum are correct" if status == "ok" else ""))
        return True
    print(f"    The finished file has a {status}. It is deleted; run the command again.")
    target.unlink()
    return False


def check_all(rows, remote):
    """--check: say for every video whether the file on disk is fine."""
    problems = 0
    for row in rows:
        status = check_file(target_path(row["video_id"]), remote.get(row["archive_file"]))
        print(f"{row['video_id']:<8} {status:<14} {row['title']}")
        problems += status not in ("ok", "not verified")
    print(f"\n{len(rows) - problems} of {len(rows)} videos are fine." + ("" if not problems else
          "  Run `python step0_download_videos.py` to get the others."))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
