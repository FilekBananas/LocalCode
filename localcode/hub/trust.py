"""How much a repository can be trusted, and why — in plain Polish.

The store shows thousands of models it did not vet by hand, so every row carries
a trust score.  The score is **not** a malware verdict and this module never
claims it is; it is the answer to "how much do we actually know about this file",
computed from four facts the Hub gives us for free:

=======================  ======  ===========================================
signal                   points  why it matters
=======================  ======  ===========================================
creator                  40      the laboratory's own repository, a known
                                 converter or a verified organisation is
                                 accountable; an anonymous account is not
                                 (:mod:`localcode.hub.publishers`, plus the
                                 Hub's ``authors`` overview: verified,
                                 followers)
likes                    30      people who opened the file and vouched for it
downloads this month     15      a file pulled a million times a month has
                                 been opened by a lot of people
downloads all time       15      the same, over the repository's whole life
=======================  ======  ===========================================

Two things reject a repository outright (score 0, level ``rejected``) instead
of lowering the score: files that run code when loaded (pickles, executables,
scripts) and a creator worth no points at all.  Licence, file formats and
maintenance are still reported as sentences, but add no points.

Every component contributes a documented number of points and one Polish
sentence, so the UI can always answer "why 62?" — see :meth:`Trust.to_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

from localcode.hub.publishers import Provenance, classify

#: Trust bands used by the badge colour.
TrustLevel = Literal["high", "good", "medium", "low", "unknown", "rejected"]

#: Points a perfect repository can collect (used to normalise to 0–100).
MAX_POINTS = 100

#: File extensions that execute code when a loader opens them (Python pickles).
PICKLE_EXTS = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".msgpack", ".h5", ".joblib")
#: Pure-data weight formats.
SAFE_WEIGHT_EXTS = (".gguf", ".safetensors", ".npz", ".mlmodelc", ".onnx")
#: Executable/script extensions that may execute code at load time.
SUSPICIOUS_CODE_EXTS = (".exe", ".dll", ".so", ".dylib", ".sh", ".bash", ".bat", ".cmd", ".ps1", ".pyc", ".jar", ".msi")
#: Files whose presence means "the loader is asked to run this author's code".
CODE_EXTS = (".py", ".pyc", ".so", ".dylib", ".sh", ".bash", ".exe", ".dll", ".bat", ".cmd", ".ps1", ".jar", ".msi")

#: Licences that state clearly what you may do; the value is the display name.
KNOWN_LICENSES: dict[str, str] = {
    "apache-2.0": "Apache 2.0",
    "mit": "MIT",
    "bsd": "BSD",
    "bsd-2-clause": "BSD 2-Clause",
    "bsd-3-clause": "BSD 3-Clause",
    "cc0-1.0": "CC0",
    "cc-by-4.0": "CC BY 4.0",
    "cc-by-sa-4.0": "CC BY-SA 4.0",
    "cc-by-nc-4.0": "CC BY-NC 4.0 (bez użycia komercyjnego)",
    "cc-by-nc-sa-4.0": "CC BY-NC-SA 4.0 (bez użycia komercyjnego)",
    "openrail": "OpenRAIL",
    "openrail++": "OpenRAIL++",
    "creativeml-openrail-m": "CreativeML OpenRAIL-M",
    "mpl-2.0": "MPL 2.0",
    "lgpl-3.0": "LGPL 3.0",
    "gpl-3.0": "GPL 3.0",
    "agpl-3.0": "AGPL 3.0",
    "artistic-2.0": "Artistic 2.0",
    "unlicense": "Unlicense",
    "gemma": "licencja Gemma (Google)",
    "llama2": "licencja Llama 2 (Meta)",
    "llama3": "licencja Llama 3 (Meta)",
    "llama3.1": "licencja Llama 3.1 (Meta)",
    "llama3.2": "licencja Llama 3.2 (Meta)",
    "llama3.3": "licencja Llama 3.3 (Meta)",
    "llama4": "licencja Llama 4 (Meta)",
    "qwen": "licencja Qwen (Alibaba)",
    "tongyi-qianwen": "licencja Tongyi Qianwen (Alibaba)",
    "deepseek": "licencja DeepSeek",
    "mistral-ai-research-licence": "licencja badawcza Mistral AI",
    "mrl": "Mistral Research License",
    "nvidia-open-model-license": "NVIDIA Open Model License",
    "falcon-llm-license": "licencja Falcon (TII)",
    "gpl": "GPL",
    "other": "",          # present but unnamed — handled as a partial credit
}


#: A staff pick is a model somebody here downloaded, loaded and talked to.  That
#: is a stronger signal than anything the API reports, so a staff pick never
#: shows below the "sprawdzone" band even when the Hub tells us little about the
#: repository (it may not be in the shipped index at all).
CURATED_FLOOR = 60

#: Repositories older than this without an update are called out as stale.
STALE_DAYS = 550
FRESH_DAYS = 200

_LEVEL_BANDS: tuple[tuple[int, TrustLevel], ...] = (
    (80, "high"), (60, "good"), (40, "medium"), (0, "low"),
)
LEVEL_LABELS: dict[TrustLevel, str] = {
    "high": "wysokie zaufanie",
    "good": "sprawdzone",
    "medium": "średnie zaufanie",
    "low": "ostrożnie",
    "unknown": "brak danych",
    "rejected": "odrzucony",
}


@dataclass
class Trust:
    """A trust score with the sentences that produced it."""

    score: int                                   # 0–100
    level: TrustLevel
    label: str                                   # Polish band name
    reasons: list[str] = field(default_factory=list)    # what raises trust
    warnings: list[str] = field(default_factory=list)   # what lowers it
    provenance: Provenance | None = None
    license_label: str | None = None
    unsafe_files: list[str] = field(default_factory=list)
    code_files: list[str] = field(default_factory=list)
    gated: bool = False
    components: dict[str, int] = field(default_factory=dict)  # signal → points, for tests/debug
    excluded: bool = False
    exclusion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "score": self.score,
            "level": self.level,
            "label": self.label,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "license": self.license_label,
            "gated": self.gated,
            "components": dict(self.components),
            "excluded": self.excluded,
        }
        if self.exclusion is not None:
            out["exclusion"] = self.exclusion
        if self.provenance is not None:
            out["provenance"] = self.provenance.to_dict()
        if self.unsafe_files:
            out["unsafeFiles"] = self.unsafe_files[:6]
        return out


def _level(score: int) -> TrustLevel:
    for threshold, level in _LEVEL_BANDS:
        if score >= threshold:
            return level
    return "low"


def _age_days(value: Any) -> float | None:
    """Days since an ISO timestamp / datetime / epoch-ms value."""
    if value in (None, ""):
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        seconds = float(value) / 1000.0 if float(value) > 1e11 else float(value)
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def license_from_tags(tags: Any) -> str | None:
    """The licence id carried in Hugging Face tags (``license:apache-2.0``).

    The Hub returns the licence in two places and not always in both: the model
    card's ``license`` field, and a tag.  The index was built from the card field
    alone, so every one of the six thousand rows came out without a licence —
    twelve points of trust missing and a warning shown that nothing justified.
    Reading the tag as well costs nothing and fixes all of them.
    """
    for tag in tags or ():
        if isinstance(tag, str) and tag.lower().startswith("license:"):
            value = tag.partition(":")[2].strip()
            if value:
                return value
    return None


def _license_points(license_id: str | None, license_name: str | None) -> tuple[int, str | None, str | None]:
    """``(points, display label, warning)`` for a licence field."""
    key = (license_id or "").strip().lower()
    if not key:
        return 0, None, "Repozytorium nie podaje licencji — nie wiadomo, na jakich warunkach można używać modelu."
    known = KNOWN_LICENSES.get(key)
    if known:
        return 12, known, None
    if key in ("other", "unknown"):
        named = (license_name or "").strip()
        if named:
            return 7, named, None
        return 5, "własna licencja autora", "Autor podaje licencję „other” bez nazwy — warto ją przeczytać w karcie modelu."
    # An unrecognised but explicitly named SPDX-ish id still tells you the terms.
    return 9, license_id, None


def _downloads_month_points(downloads: Any) -> tuple[int, str | None]:
    """Monthly downloads component (max 15)."""
    dl = int(downloads or 0)
    if dl >= 1_000_000:
        return 15, f"Pobrany {dl // 1_000_000} mln razy w ostatnim miesiącu."
    elif dl >= 100_000:
        return 12, f"Pobrany {dl // 1000} tys. razy w ostatnim miesiącu."
    elif dl >= 10_000:
        return 9, f"Pobrany {dl // 1000} tys. razy w ostatnim miesiącu."
    elif dl >= 1_000:
        return 6, f"Pobrany {dl} razy w ostatnim miesiącu."
    elif dl >= 100:
        return 3, f"Pobrany {dl} razy w ostatnim miesiącu."
    else:
        return 0, None


def _downloads_total_points(downloads_all_time: Any, downloads: Any) -> tuple[int, str | None]:
    """All-time downloads component (max 15). Falls back to monthly downloads if all-time not provided."""
    dl = int(downloads_all_time or 0) if downloads_all_time is not None else int(downloads or 0)
    if dl >= 10_000_000:
        return 15, f"Łącznie {dl // 1_000_000} mln pobrań."
    elif dl >= 1_000_000:
        return 12, f"Łącznie {dl // 1_000_000} mln pobrań."
    elif dl >= 100_000:
        return 9, f"Łącznie {dl // 1000} tys. pobrań."
    elif dl >= 10_000:
        return 6, f"Łącznie {dl // 1000} tys. pobrań."
    elif dl >= 1_000:
        return 3, f"Łącznie {dl} pobrań."
    else:
        return 0, None


def _likes_points(likes: Any) -> tuple[int, str | None]:
    """Likes component (max 30)."""
    lk = int(likes or 0)
    if lk >= 1000:
        return 30, f"{lk} polubień na Hugging Face."
    elif lk >= 500:
        return 26, f"{lk} polubień na Hugging Face."
    elif lk >= 200:
        return 22, f"{lk} polubień na Hugging Face."
    elif lk >= 100:
        return 18, f"{lk} polubień na Hugging Face."
    elif lk >= 50:
        return 14, f"{lk} polubień na Hugging Face."
    elif lk >= 20:
        return 10, f"{lk} polubień na Hugging Face."
    elif lk >= 5:
        return 5, f"{lk} polubień na Hugging Face."
    elif lk >= 1:
        return 2, f"{lk} polubień na Hugging Face."
    else:
        return 0, None


def _creator_points(prov: Provenance, author_info: dict[str, Any] | None = None) -> int:
    """Creator component (max 40). Evaluates provenance, known lab status, and author verification."""
    author_info = author_info or {}
    try:
        followers = int(author_info.get("followers") or 0)
    except (TypeError, ValueError):
        followers = 0

    # Tier scoring: return the max of all tiers that apply
    tiers = []

    # 40: provenance kind "official"
    if prov.kind == "official":
        tiers.append(40)

    # 34: provenance "trusted" and lab_known
    if prov.kind == "trusted" and prov.lab_known:
        tiers.append(34)

    # 34: verified Hugging Face organisation
    if author_info.get("verified") is True:
        tiers.append(34)

    # 26: provenance "trusted" and not lab_known
    if prov.kind == "trusted" and not prov.lab_known:
        tiers.append(26)

    # 22: author with >= 1000 followers
    if followers >= 1000:
        tiers.append(22)

    # 16: provenance "community"
    if prov.kind == "community":
        tiers.append(16)

    # 10: author with >= 100 followers
    if followers >= 100:
        tiers.append(10)

    return max(tiers) if tiers else 0


#: Below this many creator points the account sentence is a warning, not a reason.
CREATOR_WEAK = 22


def _creator_sentence(author: str, author_info: dict[str, Any] | None) -> str | None:
    """What the Hub says about the account behind the repository, one sentence."""
    if not author or not author_info:
        return None
    try:
        followers = int(author_info.get("followers") or 0)
    except (TypeError, ValueError):
        followers = 0
    if author_info.get("verified") is True:
        kind = "zweryfikowana organizacja"
    elif author_info.get("isOrg"):
        kind = "niezweryfikowana organizacja"
    else:
        kind = "konto prywatne"
    if followers >= 10_000:
        count = f"{followers // 1000} tys. obserwujących"
    elif followers == 1:
        count = "1 obserwujący"
    else:
        count = f"{followers} obserwujących"
    return f"Twórca repozytorium: {author} ({kind}, {count})."


def summarize_files(siblings: Iterable[str]) -> dict[str, Any]:
    """Condense a repository's file list to the three facts the score needs.

    The generated catalogue stores this summary instead of thousands of file
    names, so :func:`evaluate` can score an offline row exactly as it scores a
    live search result.
    """
    files = [str(f) for f in (siblings or ())]
    lower = [f.lower() for f in files]
    code_exts = sorted({"." + low.rsplit(".", 1)[-1] for low in lower if low.endswith(CODE_EXTS)})
    suspicious_exts = sorted(set(code_exts) & set(SUSPICIOUS_CODE_EXTS))
    return {
        "known": bool(files),
        "safe": any(low.endswith(SAFE_WEIGHT_EXTS) for low in lower),
        "pickle": sorted({"." + low.rsplit(".", 1)[-1] for low in lower if low.endswith(PICKLE_EXTS)}),
        "code": code_exts,
        "suspicious": suspicious_exts,
    }


def _file_points(
    siblings: Iterable[str] | None,
    summary: dict[str, Any] | None = None,
) -> tuple[int, list[str], list[str], list[str], list[str]]:
    """``(points, reasons, warnings, unsafe extensions, code extensions)``."""
    if summary is None:
        summary = summarize_files(siblings or ())
    unsafe = [str(e) for e in (summary.get("pickle") or [])]
    code = [str(e) for e in (summary.get("code") or [])]
    safe = bool(summary.get("safe"))
    reasons: list[str] = []
    warnings: list[str] = []
    points = 0
    if not summary.get("known"):
        return 0, reasons, ["Nie udało się sprawdzić zawartości repozytorium."], unsafe, code
    if unsafe:
        warnings.append(
            "Zawiera pliki w formacie pickle (" + ", ".join(unsafe) + "), "
            "które przy wczytaniu mogą wykonać dowolny kod. LocalCode ich nie wczytuje, "
            "ale to znak, że repozytorium nie jest przygotowane pod bezpieczny format."
        )
    elif safe:
        points += 16
        reasons.append("Same pliki z danymi (GGUF/safetensors) — nic się nie wykonuje przy wczytaniu.")
    if code:
        warnings.append(
            "Zawiera kod autora (" + ", ".join(code[:3]) + ") — "
            "biblioteki takie jak transformers uruchamiają go przy wczytaniu modelu."
        )
        points -= 4
    return max(0, points), reasons, warnings, unsafe, code


def _is_excluded_by_files(summary: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Check if repository should be excluded based on file content (pickle or suspicious code)."""
    if summary is None:
        summary = {}
    pickle = summary.get("pickle") or []
    # Summaries stored in the shipped index predate the "suspicious" key; their
    # executables sit in "code" and must be caught all the same.
    suspicious = summary.get("suspicious") or [
        ext for ext in (summary.get("code") or []) if str(ext).lower() in SUSPICIOUS_CODE_EXTS]

    if pickle or suspicious:
        exts = sorted(set(pickle) | set(suspicious))
        exts_str = ", ".join(exts)
        return True, f"Odrzucony: zawiera pliki, które mogą wykonać kod przy wczytaniu ({exts_str})."

    return False, None


def _maintenance_points(last_modified: Any, created_at: Any) -> tuple[int, list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    age = _age_days(last_modified)
    if age is None:
        return 0, reasons, warnings
    if age <= FRESH_DAYS:
        return 6, ["Repozytorium aktualizowane w ostatnich miesiącach."], warnings
    if age <= STALE_DAYS:
        return 3, reasons, warnings
    years = age / 365.0
    warnings.append(f"Nieaktualizowane od ponad {years:.0f} lat — może nie działać z nowym silnikiem.")
    return 0, reasons, warnings


def evaluate(
    repo: str,
    *,
    author: str | None = None,
    downloads: Any = None,
    downloads_all_time: Any = None,
    likes: Any = None,
    license_id: str | None = None,
    license_name: str | None = None,
    siblings: Iterable[str] | None = None,
    file_summary: dict[str, Any] | None = None,
    base_models: Any = None,
    tags: Iterable[str] | None = None,
    last_modified: Any = None,
    created_at: Any = None,
    gated: Any = False,
    curated: bool = False,
    provenance: Provenance | None = None,
    author_info: dict[str, Any] | None = None,
) -> Trust:
    """Score one repository.

    Only ``repo`` is required; every missing signal simply contributes nothing,
    which is why a model the Hub tells us little about lands in a lower band
    instead of being rejected.  ``curated=True`` marks a model this project has
    run itself — a human-verified signal no API provides.
    """
    prov = provenance or classify(repo, base_models=base_models, tags=tags)
    if file_summary is None:
        file_summary = summarize_files(siblings or ())

    components: dict[str, int] = {}
    reasons: list[str] = []
    warnings: list[str] = []

    # Add provenance reasons/warnings
    reasons.extend(prov.reasons)
    if prov.kind == "unknown":
        warnings.append(prov.detail)

    # License points (for display, but no longer part of score)
    lic_points, lic_label, lic_warning = _license_points(license_id, license_name)
    if lic_label:
        reasons.append(f"Licencja: {lic_label}.")
    if lic_warning:
        warnings.append(lic_warning)

    # Downloads (monthly) component - max 15
    dl_month_points, dl_month_reason = _downloads_month_points(downloads)
    components["downloads_month"] = dl_month_points
    if dl_month_reason:
        reasons.append(dl_month_reason)

    # Downloads (all-time) component - max 15
    dl_total_points, dl_total_reason = _downloads_total_points(downloads_all_time, downloads)
    components["downloads_total"] = dl_total_points
    if dl_total_reason:
        reasons.append(dl_total_reason)

    # Likes component - max 30
    likes_points, likes_reason = _likes_points(likes)
    components["likes"] = likes_points
    if likes_reason:
        reasons.append(likes_reason)

    # Creator component - max 40
    creator_points = _creator_points(prov, author_info)
    components["creator"] = creator_points
    creator_line = _creator_sentence(author or repo.partition("/")[0], author_info)
    if creator_line:
        (warnings if creator_points < CREATOR_WEAK else reasons).append(creator_line)

    # File safety check and warnings
    file_points, file_reasons, file_warnings, unsafe, code = _file_points(siblings, file_summary)
    reasons.extend(file_reasons)
    warnings.extend(file_warnings)

    # Maintenance check
    maint_points, maint_reasons, maint_warnings = _maintenance_points(last_modified, created_at)
    reasons.extend(maint_reasons)
    warnings.extend(maint_warnings)

    # Check exclusion criteria
    excluded = False
    exclusion_text = None

    # Exclusion (a): suspicious files or pickle
    excluded_by_files, file_exclusion = _is_excluded_by_files(file_summary)
    if excluded_by_files:
        excluded = True
        exclusion_text = file_exclusion

    # Exclusion (b): no known creator (only if not curated)
    if not excluded and not curated and creator_points == 0:
        excluded = True
        exclusion_text = "Odrzucony: nie wiadomo, kto stoi za tym modelem — twórca nie jest znanym laboratorium, zaufanym autorem konwersji ani zweryfikowaną organizacją."

    # Compute score from components
    total = sum(components.values())
    score = int(round(total * 100 / MAX_POINTS))
    score = max(0, min(100, score))

    # Apply curated floor (unless excluded)
    if curated and not excluded:
        reasons.insert(0, "Sprawdzony przez LocalCode.")
        score = max(score, CURATED_FLOOR)

    # If excluded, set score to 0 and level to rejected
    if excluded:
        score = 0
        level: TrustLevel = "rejected"
    else:
        level = _level(score)

    # Gated model warning
    is_gated = bool(gated) and gated is not False
    if is_gated:
        warnings.append(
            "Wymaga zaakceptowania licencji na stronie modelu i tokenu Hugging Face.")

    return Trust(
        score=score,
        level=level,
        label=LEVEL_LABELS[level],
        reasons=reasons,
        warnings=warnings,
        provenance=prov,
        license_label=lic_label,
        unsafe_files=unsafe,
        code_files=code,
        gated=is_gated,
        components=components,
        excluded=excluded,
        exclusion=exclusion_text,
    )


def evaluate_row(row: dict[str, Any], *, curated: bool = False) -> Trust:
    """:func:`evaluate` for a raw row from ``scripts/hub_fetch.py`` / the Hub API."""
    return evaluate(
        str(row.get("id") or ""),
        author=row.get("author"),
        downloads=row.get("downloads"),
        downloads_all_time=row.get("downloadsAllTime"),
        likes=row.get("likes"),
        license_id=row.get("license"),
        license_name=row.get("licenseName"),
        siblings=row.get("siblings"),
        file_summary=row.get("files") or None,
        base_models=row.get("baseModels"),
        tags=row.get("tags") or (),
        last_modified=row.get("lastModified"),
        created_at=row.get("createdAt"),
        gated=row.get("gated"),
        curated=curated,
        author_info=row.get("authorInfo"),
    )
