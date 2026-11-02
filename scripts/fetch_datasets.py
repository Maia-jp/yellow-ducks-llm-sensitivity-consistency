#!/usr/bin/env python3
"""Check, restore and build the five datasets with pinned SHA-256 hashes.

The data files are committed in sensitivity-consistency-LLM/data. This script:
  1. downloads a file only when it is missing or its hash is wrong,
  2. extracts WOS46985/X.txt from X.zip (the notebook reads X.txt),
  3. builds CB/test_labeled.jsonl from the original CB annotations,
  4. writes the DBPedia and WoS 2000-sample subsets with scripts/replay_sampling.py,
  5. checks every file against docs/datasets/data-manifest.json.

Usage, from the repository root:
    python3 scripts/fetch_datasets.py            # restore and verify
    python3 scripts/fetch_datasets.py --verify-only

On ITU HPC, run this with `srun` or `sbatch`. Do not run it on the login node.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/datasets/data-manifest.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def open_url(url: str, timeout: int):
    # Mendeley answers 403 to the default Python user agent.
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8"}), timeout=timeout)


def has_hash(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path.read_bytes()) == expected


def download(url: str, expected: str, path: Path) -> None:
    with open_url(url, 120) as response:
        data = response.read()
    if sha256(data) != expected:
        raise SystemExit(f"Hash mismatch for {url}. Stop.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"downloaded {path.name}")


def download_from_archive(info: dict, path: Path) -> None:
    """Download a tar.gz archive, check its hash, and write one member file."""
    with open_url(info["url"], 300) as response:
        data = response.read()
    if sha256(data) != info["archive_sha256"]:
        raise SystemExit(f"Hash mismatch for {info['url']}. Stop.")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        member = archive.extractfile(info["member"])
        if member is None:
            raise SystemExit(f"{info['member']} is not a file in the archive.")
        content = member.read()
    if sha256(content) != info["sha256"]:
        raise SystemExit(f"Hash mismatch for {info['member']}. Stop.")
    path.write_bytes(content)
    print(f"extracted {path.name} from the archive")


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def restore(out: Path, manifest: dict) -> None:
    base = manifest["upstream_raw_base"]
    for relative, info in manifest["files"].items():
        if not has_hash(out / relative, info["sha256"]):
            download(base + relative, info["sha256"], out / relative)

    x_txt = out / "WOS46985/X.txt"
    if not has_hash(x_txt, manifest["derived"]["WOS46985/X.txt"]["sha256"]):
        with zipfile.ZipFile(out / "WOS46985/X.zip") as archive:
            x_txt.write_bytes(archive.read("X.txt"))
        print("extracted WOS46985/X.txt")

    items = manifest["derived"]["CommitmentBank-items.csv"]
    if not has_hash(out / "CommitmentBank-items.csv", items["sha256"]):
        download(items["url"], items["sha256"], out / "CommitmentBank-items.csv")

    if not has_hash(out / "CB/test_labeled.jsonl", manifest["derived"]["CB/test_labeled.jsonl"]["sha256"]):
        run(
            str(ROOT / "scripts/prepare_cb_test_labels.py"),
            "--test-jsonl", str(out / "CB/test.jsonl"),
            "--items-csv", str(out / "CommitmentBank-items.csv"),
            "--output-jsonl", str(out / "CB/test_labeled.jsonl"),
        )

    for relative, info in manifest["supporting"].items():
        if has_hash(out / relative, info["sha256"]):
            continue
        if "member" in info:
            download_from_archive(info, out / relative)
        else:
            download(info["url"], info["sha256"], out / relative)

    replay = str(ROOT / "scripts/replay_sampling.py")
    subsets = manifest["subsets"]
    if not all(has_hash(out / rel, info["sha256"]) for rel, info in subsets.items() if rel.startswith("DBPEDIA")):
        run(replay, "--dataset", "dbpedia", "--input-file", str(out / "DBPEDIA/test.csv"), "--subset-dir", str(out / "DBPEDIA"))
    if not all(has_hash(out / rel, info["sha256"]) for rel, info in subsets.items() if rel.startswith("WOS")):
        run(
            replay, "--dataset", "wos",
            "--input-file", str(out / "WOS46985/X.txt"),
            "--target-file", str(out / "WOS46985/YL1.txt"),
            "--subset-dir", str(out / "WOS46985"),
        )


def verify(out: Path, manifest: dict) -> bool:
    ok = True
    for section in ("files", "derived", "subsets", "supporting"):
        for relative, info in manifest[section].items():
            path = out / relative
            if not path.is_file():
                print(f"MISSING  {relative}")
                ok = False
                continue
            actual = sha256(path.read_bytes())
            good = actual == info["sha256"]
            ok &= good
            print(f"{'OK      ' if good else 'MISMATCH'} {relative}" + ("" if good else f" (got {actual[:12]}, want {info['sha256'][:12]})"))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="Data directory. Default: data_dir from the manifest.")
    parser.add_argument("--verify-only", action="store_true", help="Only check files. Change nothing.")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = args.out or ROOT / manifest["data_dir"]
    if not args.verify_only:
        restore(out, manifest)
    ok = verify(out, manifest)
    print("All files match the manifest." if ok else "Some files do not match the manifest.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
