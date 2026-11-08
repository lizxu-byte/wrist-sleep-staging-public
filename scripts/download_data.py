"""Download the PhysioNet `sleep-accel` v1.0.0 dataset into ``data/`` (git-ignored).

Only the three folders used by the analysis are fetched (``motion``,
``heart_rate``, ``labels``) plus ``LICENSE.txt``. Every file is checked against
the published ``SHA256SUMS.txt``. Files that already exist with the correct hash
are skipped, so the script can be re-run safely.

Usage::

    python scripts/download_data.py [--dest data/sleep-accel] [--include-steps]

Dataset: Walch, O. (2019). Motion and heart rate from a wrist-worn wearable and
labeled sleep from polysomnography (version 1.0.0). PhysioNet.
https://doi.org/10.13026/hmhs-py35 - Open Data Commons Attribution License v1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://physionet.org/files/sleep-accel/1.0.0/"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "sleep-accel"


def _fetch(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read()
        except OSError as exc:  # network error: back off and retry
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry in {wait}s after error: {exc}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--include-steps", action="store_true",
                    help="also download the steps folder (not used by the analysis)")
    args = ap.parse_args(argv)

    folders = {"motion", "heart_rate", "labels"}
    if args.include_steps:
        folders.add("steps")

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    sums_text = _fetch(BASE_URL + "SHA256SUMS.txt").decode()
    (dest / "SHA256SUMS.txt").write_text(sums_text)

    wanted = []
    for line in sums_text.splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        if rel == "LICENSE.txt" or rel.split("/")[0] in folders:
            wanted.append((digest, rel))

    for i, (digest, rel) in enumerate(wanted, 1):
        out = dest / rel
        if out.exists() and _sha256(out) == digest:
            continue
        print(f"[{i}/{len(wanted)}] {rel}")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(_fetch(BASE_URL + rel))
        if _sha256(tmp) != digest:
            tmp.unlink()
            print(f"SHA256 mismatch for {rel}", file=sys.stderr)
            return 1
        tmp.replace(out)
    print(f"Done: {len(wanted)} files verified in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
