#!/usr/bin/env python3
"""Download the official logo of every laboratory the store knows.

    python scripts/fetch_logos.py                 # → localcode/hub/data/logos/*.png
    python scripts/fetch_logos.py --check         # only report what is missing

The source is the organisation's own avatar on Hugging Face — the mark the
laboratory publishes itself next to its models (DeepSeek's whale, Google's G,
Mistral's logo).  Nothing is guessed or drawn: an organisation without a usable
avatar simply gets no logo, and the app shows none for it.

Images are stored as 64×64 PNGs (a few kilobytes each) via ``sips``, which is
part of macOS, so the build needs no image library.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localcode.hub.publishers import OFFICIAL_PUBLISHERS  # noqa: E402

OUT_DIR = Path("localcode/hub/data/logos")
SIZE = 64
TIMEOUT = 20
#: Organisations whose avatar is a stock image or a person's photo rather than
#: the model's brand: better nothing than a misleading mark.
SKIP = {"teknium", "vikhyatk", "nickypro", "prithivmlmods", "hexgrad"}


def overview(org: str) -> dict:
    url = f"https://huggingface.co/api/organizations/{org}/overview"
    request = urllib.request.Request(url, headers={"User-Agent": "LocalCode/0.1"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


def download(url: str, target: Path) -> bool:
    """Fetch ``url`` and write it as a square PNG of :data:`SIZE` pixels."""
    request = urllib.request.Request(url, headers={"User-Agent": "LocalCode/0.1"})
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw"
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                raw.write_bytes(response.read())
        except Exception as exc:  # noqa: BLE001 - one missing logo is not a failure
            print(f"  ! pobieranie: {exc}", file=sys.stderr)
            return False
        if raw.stat().st_size < 200:
            return False
        converted = Path(tmp) / "out.png"
        result = subprocess.run(
            ["sips", "-s", "format", "png", "-Z", str(SIZE), str(raw), "--out", str(converted)],
            capture_output=True, text=True)
        if result.returncode != 0 or not converted.exists():
            print(f"  ! sips: {result.stderr.strip()[:120]}", file=sys.stderr)
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(converted), target)
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--check", action="store_true", help="nie pobieraj, tylko sprawdź")
    ap.add_argument("--only", nargs="*", help="wybrane organizacje")
    args = ap.parse_args()

    out_dir = Path(args.out)
    orgs = args.only or sorted(OFFICIAL_PUBLISHERS)
    have, added, missing = 0, 0, []
    for org in orgs:
        target = out_dir / f"{org}.png"
        if org in SKIP:
            continue
        if target.exists():
            have += 1
            continue
        if args.check:
            missing.append(org)
            continue
        try:
            data = overview(org)
        except Exception as exc:  # noqa: BLE001 - a renamed or private org
            print(f"{org:24} — brak organizacji ({type(exc).__name__})", file=sys.stderr)
            missing.append(org)
            continue
        url = str(data.get("avatarUrl") or "")
        if not url or url.startswith("/avatars/"):      # the generated placeholder
            print(f"{org:24} — brak własnego awatara", file=sys.stderr)
            missing.append(org)
            continue
        if download(url, target):
            added += 1
            verified = " ✓zweryfikowana" if data.get("isVerified") else ""
            print(f"{org:24} → {target.name}{verified}", file=sys.stderr)
        else:
            missing.append(org)

    print(f"\nlogotypy: {have} już było, {added} pobrano, {len(missing)} bez logo",
          file=sys.stderr)
    if missing:
        print("bez logo: " + ", ".join(missing), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
