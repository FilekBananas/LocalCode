#!/usr/bin/env python3
"""Draw LocalCode's placeholder app icon — a small robot head — as a PNG.

The bundle needs *an* icon from the first build, and a generated one beats the
empty grey rectangle macOS shows otherwise.  Replace it by dropping your own
1024×1024 PNG at ``macos/Resources/AppIcon.png``; ``build-app.sh`` prefers that
file and only falls back to this script.

Pure standard library on purpose (zlib + struct): the build must not depend on
Pillow being installed.

    python3 macos/scripts/make_icon.py macos/Resources/AppIcon.png [--size 1024]
"""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path

# Palette — dark indigo tile, off-white head, cyan eyes, amber cursor.
BG_TOP = (44, 48, 96)
BG_BOTTOM = (18, 19, 38)
GLOW = (86, 108, 220)
HEAD = (238, 241, 255)
HEAD_SHADE = (198, 205, 235)
EYE = (46, 216, 240)
EYE_CORE = (225, 253, 255)
CURSOR = (247, 170, 54)
ANTENNA = (247, 170, 54)

SS = 3  # supersampling factor per axis


def _squircle(x: float, y: float, cx: float, cy: float, rx: float, ry: float, n: float = 4.0) -> float:
    """Signed-ish distance: < 1 inside, 1 on the edge (superellipse)."""
    if rx <= 0 or ry <= 0:
        return 9.9
    dx = abs(x - cx) / rx
    dy = abs(y - cy) / ry
    return (dx ** n + dy ** n) ** (1.0 / n)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _sample(x: float, y: float, size: float) -> tuple[float, float, float, float]:
    """Colour of one sub-pixel: ``(r, g, b, a)`` in 0–255."""
    u, v = x / size, y / size

    # --- tile -------------------------------------------------------------
    inset = size * 0.035
    tile = _squircle(x, y, size / 2, size / 2, size / 2 - inset, size / 2 - inset, n=4.6)
    if tile > 1.0:
        return (0.0, 0.0, 0.0, 0.0)

    r, g, b = _mix(BG_TOP, BG_BOTTOM, v * 1.05)
    # soft light from the upper left
    d = math.hypot(u - 0.28, v - 0.2)
    halo = max(0.0, 1.0 - d / 0.85) ** 2.2
    r, g, b = (r + (GLOW[0] - r) * halo * 0.45,
               g + (GLOW[1] - g) * halo * 0.45,
               b + (GLOW[2] - b) * halo * 0.45)

    cx = size / 2
    # --- antenna ----------------------------------------------------------
    ant_w, ant_top, ant_bottom = size * 0.022, size * 0.145, size * 0.275
    if abs(x - cx) <= ant_w and ant_top <= y <= ant_bottom:
        r, g, b = ANTENNA
    knob = math.hypot(x - cx, y - ant_top) / (size * 0.055)
    if knob <= 1.0:
        r, g, b = ANTENNA if knob > 0.55 else EYE_CORE

    # --- head -------------------------------------------------------------
    head_cy = size * 0.545
    head_rx, head_ry = size * 0.315, size * 0.275
    head = _squircle(x, y, cx, head_cy, head_rx, head_ry, n=3.4)
    if head <= 1.0:
        # a touch of shading towards the bottom so the head is not a flat blob
        shade = max(0.0, (y - (head_cy - head_ry)) / (2 * head_ry))
        r, g, b = _mix(HEAD, HEAD_SHADE, shade * 0.55)

        # eyes
        eye_dx, eye_cy = size * 0.115, head_cy - size * 0.045
        eye_rx, eye_ry = size * 0.052, size * 0.062
        for sign in (-1, 1):
            e = _squircle(x, y, cx + sign * eye_dx, eye_cy, eye_rx, eye_ry, n=2.6)
            if e <= 1.0:
                r, g, b = _mix(EYE, EYE_CORE, max(0.0, 1.0 - e * 1.25) ** 1.6)
                break

        # mouth: a terminal cursor block
        m = _squircle(x, y, cx, head_cy + size * 0.115, size * 0.055, size * 0.032, n=3.0)
        if m <= 1.0:
            r, g, b = CURSOR

    # --- edge antialiasing is handled by supersampling; alpha is binary ----
    return (r, g, b, 255.0)


def render(size: int) -> bytes:
    """Raw RGBA rows for an ``size``×``size`` icon."""
    rows: list[bytes] = []
    step = 1.0 / SS
    weight = 1.0 / (SS * SS)
    for py in range(size):
        row = bytearray()
        for px in range(size):
            ar = ag = ab = aa = 0.0
            for sy in range(SS):
                y = py + (sy + 0.5) * step
                for sx in range(SS):
                    x = px + (sx + 0.5) * step
                    r, g, b, a = _sample(x, y, float(size))
                    ar += r * a / 255.0
                    ag += g * a / 255.0
                    ab += b * a / 255.0
                    aa += a
            aa *= weight
            if aa <= 0.5:
                row += b"\x00\x00\x00\x00"
                continue
            scale = 255.0 / aa if aa else 0.0
            row += bytes((
                max(0, min(255, int(ar * weight * scale + 0.5))),
                max(0, min(255, int(ag * weight * scale + 0.5))),
                max(0, min(255, int(ab * weight * scale + 0.5))),
                max(0, min(255, int(aa + 0.5))),
            ))
        rows.append(bytes(row))
    return b"".join(b"\x00" + r for r in rows)


def write_png(path: Path, size: int) -> None:
    raw = render(size)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">2I5B", size, size, 8, 6, 0, 0, 0)   # RGBA, 8-bit
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="macos/Resources/AppIcon.png")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out)
    write_png(out, args.size)
    print(f"{out} ({args.size}×{args.size}, {out.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
