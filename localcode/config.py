"""Paths and persistent settings shared by every LocalCode component.

Everything lives under ``LOCALCODE_HOME`` (default ``~/.localcode``)::

    ~/.localcode/
      settings.json          user settings (this module)
      server.json            {port, token, pid} of a running `localcode serve`
      auto_precedents.json   the user's answers on permission cards, per project
      models/<org>/<repo>/   downloaded models (LM Studio-like layout)
      data/chat.db           chatbot database (conversations, folders, memory, RAG)
      data/uploads/          attachments and knowledge-base files
      sessions/<project>/    coding-agent transcripts (JSONL)
      logs/                  engine and server logs

Settings are stored as camelCase-free snake_case JSON and deep-merged over the
defaults below, so adding a field never breaks an older file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

VERSION = "0.1.0"


def home() -> Path:
    return Path(os.environ.get("LOCALCODE_HOME", "~/.localcode")).expanduser()


def install_command(extra: str) -> str:
    """Polecenie, które doinstaluje dodatek ``extra`` **w tym** środowisku.

    Samo „pip install …" jest dla użytkownika aplikacji bezużyteczne: `pip` w
    jego PATH to zwykle zupełnie inny Python niż ten, w którym chodzi serwer
    (aplikacja z /Applications uruchamia ``~/.localcode/venv``). Dlatego w
    komunikatach o brakującym dodatku podajemy pełną ścieżkę interpretera —
    takie polecenie można wkleić i zadziała.
    """
    python = sys.executable or "python3"
    return f'{python} -m pip install "localcode[{extra}]"'


def settings_path() -> Path:
    return home() / "settings.json"


def server_info_path() -> Path:
    return home() / "server.json"


def data_dir() -> Path:
    return home() / "data"


def sessions_dir() -> Path:
    return home() / "sessions"


def logs_dir() -> Path:
    return home() / "logs"


def ensure_dirs(settings: "Settings | None" = None) -> None:
    for p in (home(), data_dir(), data_dir() / "uploads", sessions_dir(), logs_dir()):
        p.mkdir(parents=True, exist_ok=True)
    models = Path((settings or load_settings()).models_dir).expanduser()
    models.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Settings dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class RuntimeSettings:
    """How many models/instances may run and how much memory they may take."""

    max_loaded_models: int = 2          # distinct loaded instances at once (all models)
    max_instances_per_model: int = 1    # copies of the same model (true parallelism)
    default_parallel: int = 1           # request slots per instance (llama.cpp -np / MLX batch size)
    ram_limit_gb: float | None = None   # hard budget for all instances; None = auto (70% of physical RAM)
    idle_ttl_minutes: int = 30          # auto-unload idle instances (0 = never)
    default_ctx: int | None = None      # None = auto (min(model max, what fits in RAM, 32768))
    flash_attn: str = "auto"            # gguf: on|off|auto
    kv_cache_type: str = "f16"          # gguf: f16|q8_0|q4_0 ; mlx: f16|8bit|4bit
    gpu_layers: int = -1                # gguf: -1 = all layers on GPU
    llama_server_path: str | None = None  # None = find on PATH / Homebrew
    python_path: str | None = None        # interpreter for MLX workers; None = sys.executable
    evict_policy: str = "lru"           # lru | none (refuse to load when over budget)
    #: Models to keep in memory: never idle-unloaded, never evicted to make room.
    #: For the small assistant that downloads and switches models on request —
    #: it is only useful if it answers immediately, which means staying loaded.
    pinned_models: list[str] = field(default_factory=list)


@dataclass
class AgentSettings:
    default_model: str | None = None    # LocalModel.key
    permission_mode: str = "default"    # default | accept_edits | auto | yolo | plan
    max_steps: int = 40                 # tool-use iterations per user turn
    thinking: bool = True               # allow <think> for models that support it
    #: Let the assistant find, download and load models by itself ("uruchom mi
    #: Qwen3 Coder").  Off by default: it downloads gigabytes and takes memory.
    manage_models: bool = False
    # Requested reasoning effort: "off" | "low" | "medium" | "high". Resolved
    # down to what the model's chat template can express (a Qwen3-style
    # on/off switch gets "on"); ignored by models without any control.
    # "ultracode" = think at "high" and orchestrate parallel helpers
    # (runtime/templates.py ULTRACODE, agent/loop.py).
    reasoning_effort: str = "medium"
    temperature: float = 0.3
    max_output_tokens: int | None = None  # None = scaled to context
    auto_compact_threshold: float = 0.80  # fraction of context that triggers compaction
    show_reasoning: bool = False
    #: Let the agent hand a self-contained search to a helper with its own
    #: context ("delegate"; see localcode/agent/tools/subagent.py). Off by
    #: default like manage_models: it is a second model turn the user did not
    #: ask for, and on an 8k model it is rarely worth the round trip.
    subagents: bool = False
    #: Which model the helper runs on; None = the one the agent is already on.
    #: Only the user sets this. A model must never be able to choose it: the
    #: helper reads project files, so picking a remote one would send them to
    #: somebody else's API without the user having decided that.
    subagent_model: str | None = None
    #: Tool-use iterations the helper gets. Lower than the agent's own: a
    #: helper that needs forty steps was handed the wrong job.
    subagent_max_steps: int = 12
    #: Effort "ultracode": how many ``delegate`` calls of one step run at the
    #: same time (the rest wait for a free place). Each one is a model turn of
    #: its own — on a remote model, a paid request.
    ultracode_parallel: int = 6
    #: ``/goal``: how many times the agent is sent back to work after a final
    #: answer before it has to stop and report. Each one can use up to
    #: ``max_steps`` steps, so this is the brake on a goal it cannot reach.
    goal_max_turns: int = 20
    #: The agent's shell commands stay inside the folder LocalCode was started
    #: in: macOS' sandbox closes the rest of the home directory to reading and
    #: everything but the project, temp files and build caches to writing, and
    #: a command that names a path outside asks first, in every mode (see
    #: ``localcode/agent/tools/shell.py``). Off = commands run with the user's
    #: full rights, as before.
    confine_to_project: bool = True
    #: Auto mode's reviewer — the model that judges every call that would ask;
    #: None = the model the agent runs on. Only the user sets this: the
    #: reviewer reads the conversation, so a remote one sends it to that
    #: provider (and every review is a billed request).
    auto_reviewer_model: str | None = None
    #: Auto mode's second gate (``agent/guard.py``): a local model that judges
    #: every shell command — and anything reaching outside the project —
    #: without seeing the conversation, so text in files, in the chat or from a
    #: remote model cannot talk it round. None = automatic: the agent's own
    #: model when it is local, else the help bot's small model, else the
    #: smallest chat model on disk. Never a remote key: the check stays here.
    auto_guard_model: str | None = None
    #: Learn from the user's answers on permission cards: what they approved or
    #: declined (shell commands, calls outside the project) is remembered per
    #: project in ``LOCALCODE_HOME`` and shown to auto mode's reviewer and guard
    #: for similar calls (``agent/precedents.py``). Never runs anything itself.
    auto_precedents_enabled: bool = True


@dataclass
class ChatSettings:
    default_model: str | None = None
    embedding_model: str | None = None  # LocalModel.key of a GGUF embedding model; None = FTS only
    # Fraction of the prompt budget that triggers summarising compaction of a
    # conversation; None = follow the agent's auto_compact_threshold.
    auto_compact_threshold: float | None = 0.80
    # Requested reasoning effort for the chatbot; see AgentSettings.
    reasoning_effort: str = "medium"


@dataclass
class ImageSettings:
    """Local image generation (``localcode/images/``, see docs/IMAGES.md)."""

    default_model: str | None = None    # LocalModel.key of an image model
    max_loaded: int = 1                 # image models kept in memory at once
    idle_ttl_minutes: int = 10          # auto-unload an idle image model (0 = never)
    output_dir: str = "data/images"     # relative paths are resolved under LOCALCODE_HOME
    default_steps: int | None = None    # None = model default (FLUX schnell 4, SD 20…)
    default_size: str = "1024x1024"
    default_guidance: float | None = None
    keep_metadata: bool = True          # write the generation parameters into the PNG


@dataclass
class StoreSettings:
    """How the store picks what to download when nobody says otherwise.

    Apple Silicon runs MLX faster than GGUF for the same model, so MLX wins a
    tie; GGUF is the fallback, and it is also the only option for models nobody
    has converted to MLX.  Four-bit is the default quality because it is the
    level at which a 30B model fits in a 32 GB Mac at all — the picker lets
    anyone trade that for precision.
    """

    #: 4bit | 5bit | 6bit | 8bit | max  (max = the highest the repository has)
    quality: str = "4bit"
    #: mlx | gguf | auto  — which packaging wins when both exist
    prefer_format: str = "mlx"
    #: Downloaded on first run so that a fresh install can answer a question
    #: straight away instead of showing an empty screen and 6000 choices.
    #: Small on purpose: about a gigabyte, runs on any Apple Silicon Mac.
    starter_model: str | None = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
    #: Set to false by anyone who would rather pick their own first model.
    auto_install_starter: bool = True


#: Service tiers a remote provider can be asked for. ``standard`` sends nothing
#: (the provider's default); ``flex`` is cheaper but may queue a request for
#: minutes when capacity is short; ``priority`` costs more and skips the queue.
SERVICE_TIERS: tuple[str, ...] = ("standard", "flex", "priority")
#: Providers whose chat API takes ``service_tier`` (all speak OpenAI's dialect).
#: DeepInfra and OpenAI document it; Unlid forwards unknown fields to the model's
#: upstream unchanged, so the setting reaches it — whether it bites is up to the
#: model. Anthropic and Gemini have no such field and are not offered one.
TIERED_PROVIDERS: tuple[str, ...] = ("openai", "deepinfra", "unlid")


def service_tier(provider: str, value: Any) -> str:
    """``value`` as a tier ``provider`` can use: ``"standard"`` for anything else."""
    tier = str(value or "").strip().lower()
    return tier if provider in TIERED_PROVIDERS and tier in SERVICE_TIERS else "standard"


@dataclass
class RemoteProvider:
    """One cloud provider the app may borrow models from."""

    api_key: str | None = None          # None/"" = not configured
    enabled: bool = True                # configured but temporarily switched off
    base_url: str | None = None         # override for proxies and compatible endpoints
    #: standard | flex | priority (see :data:`SERVICE_TIERS`); only the
    #: :data:`TIERED_PROVIDERS` send it. A value this version does not know
    #: (a hand-edited file, a tier from a newer release) loads as standard.
    service_tier: str = "standard"

    def __post_init__(self) -> None:
        tier = str(self.service_tier or "").strip().lower()
        self.service_tier = tier if tier in SERVICE_TIERS else "standard"


@dataclass
class RemoteSettings:
    """Models that run somewhere else.

    LocalCode is a local-first app and stays one: nothing here is used unless
    somebody pastes a key, remote models are always marked as remote in the UI,
    and every request that leaves this machine goes to the provider whose key it
    is.  The point is the awkward case — a 400B model, or a machine that is
    busy — not to become a browser for somebody else's API.
    """

    openai: RemoteProvider = field(default_factory=RemoteProvider)
    anthropic: RemoteProvider = field(default_factory=RemoteProvider)
    gemini: RemoteProvider = field(default_factory=RemoteProvider)
    deepinfra: RemoteProvider = field(default_factory=RemoteProvider)
    unlid: RemoteProvider = field(default_factory=RemoteProvider)


@dataclass
class HelpbotSettings:
    """The help bot: a small model that runs the rest of the app for the user.

    "Pobierz mi coś do kodowania, co się zmieści" is a sentence; the help bot
    turns it into a store search, a download, a load configuration and a load,
    and can change any setting the app has. It is *small* on purpose — it has to
    be there all the time without taking the memory the real models need.

    Qwen3 4B Instruct and not something smaller: this project measured the
    smaller ones (``hub/metadata.py``, ``AGENT_MIN_PARAMS_B``) — below about
    three billion parameters a model calls the right tool and then fills it with
    nonsense, which for a bot that deletes and downloads gigabytes is the one
    failure that matters. At 4 bits it is ~2.3 GB.
    """

    model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    #: Download the model on first start when it is missing — "it comes with
    #: the app". Skipped, with a notice, when the disk does not have room.
    auto_download: bool = True


@dataclass
class SecuritySettings:
    """Keeping a downloaded model from reaching the rest of the machine.

    ``sandbox`` applies to the engine subprocesses that parse model files —
    ``llama-server``, the MLX worker, the image workers (see
    :mod:`localcode.security.sandbox`):

    ``auto``
        sandbox where the platform allows it, and fall back to a normal process
        if the sandboxed one cannot start (the fallback is logged and shown in
        the runtime status);
    ``strict``
        sandbox or refuse to load the model;
    ``off``
        no sandbox — for debugging, or a platform without one.
    """

    sandbox: str = "auto"
    #: Scan a model's files after it is downloaded or imported
    #: (:mod:`localcode.security.scan`).
    scan_downloads: bool = True


@dataclass
class CerberSettings:
    """Cerber — the user's own antivirus for macOS — checks a model before it downloads.

    Cerber.app has a local HTTP API (Integracje → API (localhost)): loopback
    only, a bearer token on every request, and a ``POST /api/huggingface`` scan
    that reads a repository's code, configs, chat templates and pickle weights
    in its sandbox. With this on, every download first asks Cerber and starts
    only on a clean verdict (:mod:`localcode.security.cerber`).
    """

    enabled: bool = False
    url: str = "http://127.0.0.1:8765"
    #: Cerber's API token (shown by Cerber in Integracje → API). A secret: the
    #: app-settings DTO only says whether it is set.
    token: str | None = None
    #: Verdicts that stop a download. INFO and CLEAN pass.
    block: list[str] = field(default_factory=lambda: ["SUSPICIOUS", "PUA", "MALICIOUS"])
    #: How long one check may take before the download gives up. Cerber pulls
    #: and scans the repository in its sandbox: a 0.5B model took 7.7 minutes
    #: in a real run, and scans queue one at a time.
    timeout_s: int = 3600


@dataclass
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8421
    openai_api_enabled: bool = True     # expose /v1/* for other apps (like LM Studio's server)
    openai_api_requires_token: bool = False
    cors_origins: list[str] = field(default_factory=list)


@dataclass
class Settings:
    models_dir: str = field(default_factory=lambda: str(home() / "models"))
    # Off by default: the library shows only models installed through LocalCode itself.
    # Both can be switched on in the app's settings to also list models already on disk.
    scan_lmstudio: bool = False         # also list models from ~/.lmstudio/models
    scan_hf_cache: bool = False         # also list models already in ~/.cache/huggingface/hub
    hf_token: str | None = None
    language: str = "pl"
    #: "Ulubione": model keys the user starred — local keys and
    #: ``remote:provider:id`` alike. Every model picker (the Mac app's and the
    #: terminal's) lists them first; the order here is the order they were starred.
    favorite_models: list[str] = field(default_factory=list)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    chat: ChatSettings = field(default_factory=ChatSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    images: ImageSettings = field(default_factory=ImageSettings)
    store: StoreSettings = field(default_factory=StoreSettings)
    remote: RemoteSettings = field(default_factory=RemoteSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    helpbot: HelpbotSettings = field(default_factory=HelpbotSettings)
    cerber: CerberSettings = field(default_factory=CerberSettings)


def favorites_first(items: list[Any], favorites: list[str] | None, key: Any = None) -> list[Any]:
    """``items`` with the favourite models moved to the front, both groups in their own order.

    ``key`` maps an item to its model key (default: the item itself, or its
    ``"key"`` when it is a dict). A stable partition rather than a sort: whatever
    order the caller chose — loaded first, by name — stays inside each group.
    """
    wanted = set(favorites or ())
    if not wanted:
        return list(items)

    def key_of(item: Any) -> Any:
        if key is not None:
            return key(item)
        return item.get("key") if isinstance(item, dict) else item

    return sorted(items, key=lambda item: key_of(item) not in wanted)


def _merge(dc_type: type, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, ignoring unknown keys, recursing into nested ones."""
    kwargs: dict[str, Any] = {}
    defaults = dc_type()
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(defaults, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[f.name] = _merge(type(current), value)
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)


def load_settings(path: Path | None = None) -> Settings:
    p = path or settings_path()
    try:
        raw = json.loads(p.read_text())
    except FileNotFoundError:
        return Settings()
    except (OSError, json.JSONDecodeError):
        return Settings()
    return _merge(Settings, raw if isinstance(raw, dict) else {})


def write_private_json(path: Path, data: Any, *, indent: int | None = None) -> None:
    """Write ``data`` to ``path`` atomically, readable by the user only (0600).

    A temp file next to the target (``mkstemp`` creates it 0600) is renamed over
    it, so a reader never sees half a file and a crash leaves the old one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_settings(settings: Settings, path: Path | None = None) -> None:
    # the file may hold an HF token: written 0600
    write_private_json(path or settings_path(), asdict(settings), indent=2)


def update_settings(patch: dict[str, Any], path: Path | None = None) -> Settings:
    """Deep-merge ``patch`` into the stored settings and persist them."""
    current = asdict(load_settings(path))

    def deep(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep(dst[k], v)
            else:
                dst[k] = v
        return dst

    merged = _merge(Settings, deep(current, patch))
    save_settings(merged, path)
    return merged


def set_favorite_model(key: str, favorite: bool, path: Path | None = None) -> Settings:
    """Star or unstar ``key`` in the stored settings and persist them.

    Read from the file, not from whoever's copy is in memory: the app's server
    and a terminal session both write here, and each must keep the other's stars.
    """
    stored = load_settings(path).favorite_models
    if favorite:
        # Starring twice keeps the first place it got.
        current = stored if key in stored else [*stored, key]
    else:
        current = [k for k in stored if k != key]
    return update_settings({"favorite_models": current}, path)
