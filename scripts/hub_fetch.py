#!/usr/bin/env python3
"""Fetch the popular part of the Hugging Face model index into a local dump.

This is the raw-data step of ``scripts/build_catalog.py``: one pass over the
``/api/models`` listing per (format, sort) pair, keeping only the fields the
store needs.  The dump is JSON Lines so a partial run is still usable.

    python scripts/hub_fetch.py --out .cache/hub-raw.jsonl

Nothing here judges a model; that happens in ``localcode/hub/trust.py`` (facts)
and in the enrichment step (language).  The chat template is never stored — only
booleans derived from it — because templates are tens of kilobytes each.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localcode.hub.trust import license_from_tags  # noqa: E402

EXPAND = [
    "author", "downloads", "downloadsAllTime", "likes", "lastModified", "createdAt", "tags",
    "gated", "gguf", "safetensors", "pipeline_tag", "library_name", "trendingScore",
    "baseModels", "cardData", "siblings", "private", "disabled",
]

#: (format filter, sort key, how many to walk) — the union is the candidate pool.
PLAN = [
    ("gguf", "downloads", 3000),
    ("gguf", "likes", 1000),
    ("gguf", "trending_score", 600),
    ("gguf", "last_modified", 600),
    ("mlx", "downloads", 2000),
    ("mlx", "likes", 800),
    ("mlx", "trending_score", 400),
    ("mlx", "last_modified", 400),
]

_TOOL_MARKERS = ("tool_call", "tools", "function_call", "<|tool", "available_tools")
_THINK_MARKERS = ("<think>", "reasoning_content", "enable_thinking", "reasoning_effort", "<|channel|>")
_EFFORT_MARKERS = ("reasoning_effort", "effort")


def _template_flags(template: str | None) -> dict[str, bool]:
    t = (template or "").lower()
    return {
        "hasTemplate": bool(t),
        "tools": any(m in t for m in _TOOL_MARKERS),
        "thinking": any(m in t for m in _THINK_MARKERS),
        "effort": any(m in t for m in _EFFORT_MARKERS),
        "vision": "<|vision_start|>" in t or "image_pad" in t or "<image>" in t,
    }


def _card(info: Any) -> dict[str, Any]:
    data = getattr(info, "card_data", None)
    if data is None:
        return {}
    try:
        return dict(data)
    except Exception:  # noqa: BLE001 - card data is user-provided YAML
        return {}


def _base_models(info: Any) -> dict[str, Any]:
    raw = getattr(info, "base_models", None)
    if not isinstance(raw, dict):
        return {}
    models = [str(m.get("id")) for m in (raw.get("models") or []) if isinstance(m, dict) and m.get("id")]
    return {"relation": raw.get("relation"), "models": models[:4]}


def _row(info: Any, fmt: str) -> dict[str, Any]:
    gguf = getattr(info, "gguf", None) or {}
    if not isinstance(gguf, dict):
        gguf = {}
    st = getattr(info, "safetensors", None)
    st_total = None
    if st is not None:
        st_total = getattr(st, "total", None) if not isinstance(st, dict) else st.get("total")
    card = _card(info)
    siblings = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
    last = getattr(info, "last_modified", None)
    created = getattr(info, "created_at", None)
    return {
        "id": str(info.id),
        "format": fmt,
        "author": getattr(info, "author", None) or str(info.id).partition("/")[0],
        "downloads": getattr(info, "downloads", None),
        "downloadsAllTime": getattr(info, "downloads_all_time", None),
        "likes": getattr(info, "likes", None),
        "trendingScore": getattr(info, "trending_score", None),
        "createdAt": created.isoformat() if created is not None else None,
        "lastModified": last.isoformat() if last is not None else None,
        "pipelineTag": getattr(info, "pipeline_tag", None),
        "libraryName": getattr(info, "library_name", None),
        "gated": getattr(info, "gated", False) or False,
        "private": bool(getattr(info, "private", False)),
        "disabled": bool(getattr(info, "disabled", False)),
        "tags": [t for t in (getattr(info, "tags", None) or []) if len(t) < 120][:60],
        # The card field is often empty while the tag is there, so try both.
        "license": card.get("license") or license_from_tags(getattr(info, "tags", None)),
        "licenseName": card.get("license_name"),
        "baseModels": _base_models(info),
        "arch": gguf.get("architecture"),
        "ggufTotalParams": gguf.get("total"),
        "ggufContext": gguf.get("context_length"),
        "ggufTotalFileSize": gguf.get("totalFileSize"),
        "safetensorsTotal": st_total,
        "template": _template_flags(gguf.get("chat_template") or card.get("chat_template")),
        "siblings": siblings[:400],
        "siblingCount": len(siblings),
    }


def fetch_repos(repos: list[str], *, token: str | None = None,
                seen: set[str] | None = None) -> Iterator[dict[str, Any]]:
    """Fetch named repositories one by one (the staff picks).

    The popularity listing does not necessarily contain them — a hand-picked
    MLX build of an image model can sit far outside the download charts — and a
    staff pick missing from the index would be shown without a licence, a file
    list or a size.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    seen = seen if seen is not None else set()
    for repo in repos:
        if repo in seen:
            continue
        try:
            info = api.model_info(repo, expand=EXPAND)
        except Exception as exc:  # noqa: BLE001 - a renamed repo must not stop the run
            print(f"  ! {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        siblings = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
        fmt = "gguf" if any(f.lower().endswith(".gguf") for f in siblings) else "mlx"
        seen.add(repo)
        yield _row(info, fmt)


#: Packagers whose catalogues are walked past the global popularity charts:
#: the large models (30B and up) sit far down those charts — fewer people can
#: run them — so the charts alone left the store with almost nothing above
#: 20 GB for the Macs that have 64 or 128 GB of memory.
BIG_MODEL_PACKAGERS = ("unsloth", "bartowski", "lmstudio-community", "mlx-community", "ggml-org",
                       "mradermacher", "MaziyarPanahi", "QuantFactory", "second-state",
                       "gpustack", "nightmedia", "inferencerlabs", "DevQuasar")


def fetch_authors(authors: Iterable[str], *, formats: tuple[str, ...] = ("gguf", "mlx"),
                  limit: int = 400, token: str | None = None,
                  seen: set[str] | None = None) -> Iterator[dict[str, Any]]:
    """Every GGUF / MLX repository of the named organisations, most downloaded first.

    Used for the laboratories' own releases (the store must carry all of them,
    not only the ones that made a chart) and for the large-model sweep.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    seen = seen if seen is not None else set()
    for author in authors:
        for fmt in formats:
            started = time.monotonic()
            kept = 0
            for attempt in range(6):
                try:
                    for info in api.list_models(author=author, filter=fmt, sort="downloads",
                                                limit=limit, expand=EXPAND):
                        repo = str(info.id)
                        if repo in seen:
                            continue
                        seen.add(repo)
                        kept += 1
                        yield _row(info, fmt)
                    break
                except Exception as exc:  # noqa: BLE001 - one organisation must not stop the run
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status == 429 and attempt < 5:
                        # Anonymous requests share a per-IP quota; it refills.
                        wait = 90 * (attempt + 1)
                        print(f"  … {author}/{fmt}: limit zapytań, czekam {wait}s",
                              file=sys.stderr, flush=True)
                        time.sleep(wait)
                        continue
                    print(f"  ! {author}/{fmt}: {type(exc).__name__}: {status or exc}",
                          file=sys.stderr)
                    break
            time.sleep(1.0)
            if kept:
                print(f"  {author:22} {fmt:4} +{kept:5}  ({time.monotonic() - started:.0f}s)",
                      file=sys.stderr, flush=True)


def fetch(plan: list[tuple[str, str, int]], *, token: str | None = None) -> Iterator[dict[str, Any]]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    seen: set[str] = set()
    for fmt, sort, limit in plan:
        started = time.monotonic()
        kept = 0
        try:
            items = api.list_models(filter=fmt, sort=sort, limit=limit, expand=EXPAND)
            for info in items:
                repo = str(info.id)
                if repo in seen:
                    continue
                seen.add(repo)
                kept += 1
                yield _row(info, fmt)
        except Exception as exc:  # noqa: BLE001 - a failed page must not lose the rest
            print(f"  ! {fmt}/{sort}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"  {fmt:4} {sort:14} +{kept:5} new  ({time.monotonic() - started:.0f}s, total {len(seen)})",
              file=sys.stderr, flush=True)


def _http_json(url: str, headers: dict[str, str]) -> tuple[int, Any]:
    """``(status, parsed JSON or None)`` — never raises for an HTTP error status.

    The status is the whole point: a 404 means "not an organisation, try the
    user endpoint", a 429 means "wait", and both come back with a JSON body that
    must not be mistaken for an overview.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        return exc.code, None


def fetch_author_overview(name: str, *, token: str | None = None,
                          http_getter: Any = None, sleep: Any = time.sleep) -> dict[str, Any]:
    """What the Hub says about a repository's author: organisation, verified, followers.

    ``http_getter(url, headers) -> (status, json)``; injected by the tests.
    Organisations first (``isVerified``, ``plan`` live only there), then users.
    Both 404 → ``missing``. A 429 waits and retries; anything else gives up on
    that endpoint rather than inventing an answer.
    """
    getter = http_getter or _http_json
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for kind in ("organizations", "users"):
        url = f"https://huggingface.co/api/{kind}/{name}/overview"
        for attempt in range(6):
            status, data = getter(url, headers)
            if status == 429 and attempt < 5:
                wait = 90 * (attempt + 1)
                print(f"  … {name}: limit zapytań, czekam {wait}s", file=sys.stderr, flush=True)
                sleep(wait)
                continue
            break
        if status == 200 and isinstance(data, dict) and not data.get("error"):
            followers = int(data.get("numFollowers") or 0)
            if kind == "organizations":
                return {"isOrg": True, "verified": bool(data.get("isVerified")),
                        "plan": data.get("plan"), "followers": followers}
            return {"isOrg": False, "verified": False, "plan": None, "followers": followers}
        if status == 429:
            # Still limited after every retry: unknown, not "missing" — a later
            # run (resume) asks again.
            return {"isOrg": False, "verified": False, "plan": None, "followers": 0, "retry": True}
    return {"isOrg": False, "verified": False, "plan": None, "followers": 0, "missing": True}


def _main_authors(args: argparse.Namespace) -> int:
    """Fetch author information from Hugging Face and save to JSON."""
    import gzip

    index_path = Path(args.index)
    # Its own file, never --out: that one defaults to the raw dump, and the
    # first run of this mode nearly overwrote sixteen thousand fetched rows
    # with the authors map.
    out_path = Path(args.authors_out)
    if out_path.suffix != ".json":
        print(f"  ! {out_path}: mapa autorów musi być plikiem .json", file=sys.stderr)
        return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)

    authors_map: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        try:
            with out_path.open(encoding="utf-8") as fh:
                authors_map = json.loads(fh.read())
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {out_path.name}: {exc}", file=sys.stderr)

    token = os.environ.get("HF_TOKEN")

    distinct_authors: set[str] = set()
    if index_path.exists():
        if str(index_path).endswith(".gz"):
            with gzip.open(index_path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
                for model in data.get("models", []):
                    repo = model.get("repo", "")
                    author = repo.split("/")[0] if repo else None
                    if author:
                        distinct_authors.add(author)
        else:
            for line in Path(index_path).open(encoding="utf-8"):
                row = json.loads(line)
                # selected.jsonl rows carry "repo", raw dumps "id".
                author = str(row.get("repo") or row.get("id") or "").split("/")[0]
                if author:
                    distinct_authors.add(author)

    print(f"  znaleziono {len(distinct_authors)} autorów", file=sys.stderr, flush=True)

    def save() -> None:
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(authors_map, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(out_path)

    started = time.monotonic()
    n = 0
    todo = [a for a in sorted(distinct_authors)
            if a not in authors_map or authors_map[a].get("retry")]
    for author in todo:
        authors_map[author] = fetch_author_overview(author, token=token)
        n += 1
        if n % 25 == 0:
            # Written as it goes: a run cut short by the rate limit resumes
            # where it stopped instead of starting again.
            save()
            print(f"  {n}/{len(todo)} autorów", file=sys.stderr, flush=True)
        time.sleep(0.3)
    save()

    print(f"{n} nowych autorów, {len(authors_map)} razem → {out_path}  ({time.monotonic() - started:.0f}s)",
          file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".cache/hub-raw.jsonl")
    ap.add_argument("--no-curated", action="store_true",
                    help="nie dociągaj modeli polecanych spoza list popularności")
    ap.add_argument("--limit-scale", type=float, default=1.0,
                    help="multiply every page size (0.02 for a quick smoke run)")
    ap.add_argument("--extra", action="store_true",
                    help="zamiast list popularności: wszystkie repozytoria laboratoriów "
                         "i duże modele zaufanych pakujących (do --out)")
    ap.add_argument("--skip", default=None,
                    help="z --extra: pomiń repozytoria, które są już w tym pliku JSONL")
    ap.add_argument("--authors", action="store_true",
                    help="zamiast fetchowania: zbierz autorów z indeksu i pobierz ich info")
    ap.add_argument("--authors-out", default=".cache/hub-authors.json",
                    help="z --authors: gdzie zapisać mapę autorów (.json)")
    ap.add_argument("--index", default="localcode/hub/data/catalog.json.gz",
                    help="z --authors: ścieżka do indeksu (gzip JSON lub JSONL)")
    args = ap.parse_args()

    if args.authors:
        return _main_authors(args)
    if args.extra:
        return _main_extra(args)

    plan = [(f, s, max(20, int(n * args.limit_scale))) for f, s, n in PLAN]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    started = time.monotonic()
    token = os.environ.get("HF_TOKEN")
    seen: set[str] = set()
    with out.open("w", encoding="utf-8") as fh:
        for row in fetch(plan, token=token):
            seen.add(row["id"])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        if not args.no_curated:
            from localcode.hub.curated import CURATED, CURATED_IMAGES

            picks = [m["repo"] for m in [*CURATED, *CURATED_IMAGES]]
            missing = [r for r in picks if r not in seen]
            print(f"  polecane spoza list popularności: {len(missing)}", file=sys.stderr)
            for row in fetch_repos(missing, token=token, seen=seen):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} repozytoriów → {out}  ({time.monotonic() - started:.0f}s)", file=sys.stderr)
    return 0


def _main_extra(args: argparse.Namespace) -> int:
    from localcode.hub.publishers import OFFICIAL_PUBLISHERS

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if args.skip:
        for line in Path(args.skip).open(encoding="utf-8"):
            seen.add(json.loads(line)["id"])
    token = os.environ.get("HF_TOKEN")
    started = time.monotonic()
    n = 0
    scale = args.limit_scale
    with out.open("w", encoding="utf-8") as fh:
        for row in fetch_authors(sorted(OFFICIAL_PUBLISHERS), limit=max(20, int(400 * scale)),
                                 token=token, seen=seen):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        print(f"  laboratoria: {n}", file=sys.stderr, flush=True)
        for row in fetch_authors(BIG_MODEL_PACKAGERS, limit=max(20, int(2500 * scale)),
                                 token=token, seen=seen):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} repozytoriów → {out}  ({time.monotonic() - started:.0f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
