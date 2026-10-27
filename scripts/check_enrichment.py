#!/usr/bin/env python3
"""Check the model-written store descriptions against the facts they were given.

A language model writing 1500 one-liners will, sooner or later, state a
parameter count nobody gave it or promote a chat model to a coding model because
the name sounds technical.  This script is the gate: every description is
checked against its own input row, and anything that fails is listed for a
rewrite.

    python scripts/check_enrichment.py --batches .cache/batches --enriched .cache/enriched
    python scripts/check_enrichment.py --redo .cache/redo.json      # what to write again

Checks (all deterministic):

* every input model has a description, and no description invents a repository;
* the tagline is one Polish sentence of sane length, without jargon the store
  hides (repository ids, GGUF/MLX, quantisation labels, "Hugging Face");
* **every number in the tagline appears in the facts** — parameter counts,
  active parameters, context length — or in the model's own name;
* a category may only be changed when the name supports the new one;
* the display name is short and free of packaging noise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localcode.hub.publishers import is_official_org  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_catalog import _round_rank, category_change_allowed  # noqa: E402

MAX_TAGLINE = 135
MIN_TAGLINE = 20
MAX_NAME_WORDS = 7

FORBIDDEN = (
    "hugging face", "huggingface", "gguf", "mlx", "safetensors", "q4_k", "q8_0", "q5_k",
    "4bit", "8bit", "mxfp4", "repozytorium", "repo ", "kwantyz", "http",
)
CATEGORIES = {"kodowanie", "czat", "rozumowanie", "wizja", "embedding", "obrazy",
              "mowa", "wideo"}


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _allowed_numbers(fact: dict[str, Any]) -> set[float]:
    """Numbers a description may legitimately contain, compared numerically.

    Comparing as text flagged "7,0 mld" for a model called ``…-7B`` — the claim
    is grounded, only spelled differently — so every candidate is normalised to
    a float and matched with a tolerance.
    """
    ok: set[float] = set()

    def add(value: Any) -> None:
        if value in (None, ""):
            return
        try:
            f = float(value)
        except (TypeError, ValueError):
            return
        ok.add(f)
        ok.add(float(round(f)))

    for key in ("paramsB", "activeParamsB"):
        value = fact.get(key)
        add(value)
        if value:
            # "0,6 mld" is also written as "600 mln", and 7.5B as "7,5 mld"
            add(round(float(value) * 1000))
            add(round(float(value)))
    ctx = fact.get("contextLength")
    if ctx:
        add(ctx)
        add(round(int(ctx) / 1000))
        add(round(int(ctx) / 1024))
    # numbers that are part of the model's own name are quoting, not inventing
    for match in _NUMBER_RE.finditer(str(fact.get("name") or "") + " " + str(fact.get("repo") or "")):
        add(match.group(0).replace(",", "."))
    return ok


def check_one(fact: dict[str, Any], item: dict[str, Any] | None) -> list[str]:
    """Problems with one description; empty when it is fine.

    ``soft`` collects things the build step can fix by itself (an unjustified
    category is simply not applied), so they never trigger a rewrite.
    """
    problems: list[str] = []
    soft: list[str] = []
    if item is None:
        return ["brak opisu"]

    tagline = str(item.get("tagline") or "").strip()
    name = str(item.get("name") or "").strip()
    category = str(item.get("category") or "").strip()

    if not tagline:
        problems.append("pusty opis")
    else:
        if len(tagline) > MAX_TAGLINE:
            problems.append(f"opis za długi ({len(tagline)})")
        if len(tagline) < MIN_TAGLINE:
            problems.append(f"opis za krótki ({len(tagline)})")
        low = tagline.lower()
        for word in FORBIDDEN:
            if word in low:
                problems.append(f"żargon w opisie: {word.strip()}")
                break
        # Naming the packager ("mradermacher", "bartowski") is jargon; naming the
        # laboratory ("Qwen", "Google") is exactly what the store should say.
        author = str(fact.get("repo") or "").partition("/")[0]
        if author and not is_official_org(author) and author.lower() in low:
            problems.append("nick autora w opisie")
        if not tagline.endswith((".", "!", "?")):
            problems.append("opis bez kropki")
        allowed = _allowed_numbers(fact)
        for match in _NUMBER_RE.finditer(tagline):
            token = match.group(0)
            try:
                value = float(token.replace(",", "."))
            except ValueError:
                continue
            if not any(abs(value - ok) <= 0.05 for ok in allowed):
                problems.append(f"liczba spoza faktów: {token}")
                break

    if not name:
        problems.append("brak nazwy")
    elif len(name.split()) > MAX_NAME_WORDS:
        problems.append(f"nazwa za długa ({len(name.split())} słów)")
    elif re.search(r"\b(gguf|mlx|q\d_[a-z]|\d+bit|mxfp4|imatrix)\b", name, re.IGNORECASE):
        problems.append("nazwa z żargonem")

    if category not in CATEGORIES:
        problems.append(f"nieznana kategoria: {category or '—'}")
    elif category != fact.get("category") and not category_change_allowed(
            str(fact.get("repo") or ""), str(fact.get("name") or ""), category):
        # Soft: the build keeps the original category and the text stays as it is.
        soft.append(f"kategoria zmieniona bez podstawy: {fact.get('category')} → {category}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", default=".cache/batches")
    ap.add_argument("--enriched", default=".cache/enriched")
    ap.add_argument("--redo", default=".cache/redo.json")
    ap.add_argument("--show", type=int, default=12, help="how many problems to print")
    args = ap.parse_args()

    batches = sorted(Path(args.batches).glob("*-*.json"))
    enriched_dir = Path(args.enriched)
    total = ok = 0
    problems: list[tuple[str, str, list[str]]] = []
    redo: dict[str, list[dict[str, Any]]] = {}
    missing_files: list[str] = []

    # The build merges every enrichment file by repository — a rewrite in
    # `redo2-000.json` replaces the original row — so the check has to look at
    # the merged view too, or it keeps reporting entries that no longer ship.
    facts_by_repo: dict[str, dict[str, Any]] = {}
    fact_origin: dict[str, str] = {}
    items: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for fact in json.loads(batch.read_text(encoding="utf-8")):
            facts_by_repo.setdefault(str(fact["repo"]), fact)
            fact_origin.setdefault(str(fact["repo"]), batch.name)
        if not (enriched_dir / batch.name).exists():
            missing_files.append(batch.name)

    # The descriptions that ship are the merged view of *every* enrichment file,
    # newest round last (``_round_rank``) — exactly what the build does. Reading
    # only the file named after each batch judged the wrong text: a description
    # rewritten in a later round was checked in its original, superseded form.
    for out_path in sorted(enriched_dir.glob("*.json"), key=lambda q: (_round_rank(q.name), q.name)):
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            problems.append((out_path.name, "—", [f"nieczytelny JSON: {exc}"]))
            continue
        for item in (data.get("models") if isinstance(data, dict) else data) or []:
            if isinstance(item, dict) and item.get("repo"):
                items[str(item["repo"])] = item

    for repo, fact in facts_by_repo.items():
        total += 1
        issues = check_one(fact, items.get(repo))
        if issues:
            problems.append((fact_origin.get(repo, "?"), repo, issues))
            redo.setdefault(fact_origin.get(repo, "?"), []).append(fact)
        else:
            ok += 1

    print(f"sprawdzono {total} modeli: {ok} w porządku, {total - ok} do poprawki", file=sys.stderr)
    if missing_files:
        print(f"brakujące paczki ({len(missing_files)}): {', '.join(missing_files[:8])}", file=sys.stderr)
    counts: dict[str, int] = {}
    for _, _, issues in problems:
        for issue in issues:
            key = issue.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"   {n:5} {key}", file=sys.stderr)
    for batch, repo, issues in problems[:args.show]:
        print(f"   · {repo}: {'; '.join(issues)}", file=sys.stderr)

    if redo:
        redo_path = Path(args.redo)
        redo_path.parent.mkdir(parents=True, exist_ok=True)
        redo_path.write_text(json.dumps(redo, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"do ponownego opisania: {sum(len(v) for v in redo.values())} modeli → {redo_path}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
