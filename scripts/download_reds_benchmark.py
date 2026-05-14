import argparse
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


FILES = [
    # Validation GT/LR are the important benchmark files for this project. The
    # train archives are listed as optional extras because the official REDS
    # Google Drive links may hit download quota limits.
    {
        "name": "val_sharp_bicubic.zip",
        "google_id": "1sChhtzN9Css10gX7Xsmc2JaC-2Pzco6a",
        "size": 330490885,
        "extract_dir": "val_sharp_bicubic",
    },
    {
        "name": "val_sharp.zip",
        "google_id": "1MGeObVQ1-Z29f-myDP7-8c3u0_xECKXq",
        "size": 4273865704,
        "extract_dir": "val_sharp",
    },
    {
        "name": "train_sharp_bicubic.zip",
        "google_id": "1a4PrjqT-hShvY9IyJm3sPF0ZaXyrCozR",
        "size": 2652551733,
        "extract_dir": "train_sharp_bicubic",
    },
    {
        "name": "train_sharp.zip",
        "google_id": "1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-",
        "size": 34261573976,
        "extract_dir": "train_sharp",
    },
]


def file_complete(path, expected_size):
    # Google Drive downloads can leave partial files. Exact size checking lets
    # --continue resume safely and avoids extracting corrupted archives.
    return path.exists() and path.stat().st_size == expected_size


def download_with_gdown(item, archive_dir, proxy, retries):
    """Download one REDS archive with retry and size validation."""
    archive_path = archive_dir / item["name"]
    if file_complete(archive_path, item["size"]):
        print(f"[skip] {archive_path} is complete")
        return archive_path

    command = [
        sys.executable,
        "-m",
        "gdown",
        "--continue",
        item["google_id"],
        "-O",
        str(archive_path),
    ]
    if proxy:
        # AutoDL environments often use a local proxy; gdown accepts it directly.
        command[3:3] = ["--proxy", proxy]

    print(f"[download] {item['name']} -> {archive_path}", flush=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"[warn] {item['name']} failed on attempt {attempt}/{retries}: {exc}", flush=True)
            time.sleep(min(60, 5 * attempt))
            continue
        if file_complete(archive_path, item["size"]):
            break
        actual = archive_path.stat().st_size if archive_path.exists() else 0
        last_error = RuntimeError(f"{item['name']} has size {actual}, expected {item['size']}")
        print(f"[warn] {last_error}", flush=True)
        time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"{item['name']} failed after {retries} attempts") from last_error

    if not file_complete(archive_path, item["size"]):
        actual = archive_path.stat().st_size if archive_path.exists() else 0
        raise RuntimeError(f"{item['name']} has size {actual}, expected {item['size']}")
    return archive_path


def extract_zip(archive_path, output_root):
    # A marker file avoids re-extracting large archives on subsequent runs.
    marker = output_root / f".{archive_path.stem}.extracted"
    if marker.exists():
        print(f"[skip] {archive_path.name} already extracted")
        return

    print(f"[extract] {archive_path.name} -> {output_root}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(output_root)
    marker.write_text("ok\n")


def main():
    parser = argparse.ArgumentParser(description="Download the REDS VSR benchmark into data/benchmark.")
    parser.add_argument("--root", default="data/benchmark/REDS")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"))
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--retries", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)
    archive_dir = root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for item in FILES:
        # Download order starts with validation files so benchmark experiments
        # can run even if the large training GT archive later hits quota limits.
        archive_path = download_with_gdown(item, archive_dir, args.proxy, args.retries)
        if not args.no_extract:
            extract_zip(archive_path, root)

    print(f"[done] REDS benchmark is available under {root}")


if __name__ == "__main__":
    main()
