"""``python -m localcode.server`` — same as ``localcode serve``."""

from __future__ import annotations

from localcode.server.main import serve_main

if __name__ == "__main__":
    raise SystemExit(serve_main())
