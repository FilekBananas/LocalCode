# LocalCode — architecture contract

LocalCode is an LM Studio-like local app with two faces: a **coding agent**
(Claude Code-style, terminal + Mac app) and a **chatbot** (UX ported from
`the reference project`). Models come from Hugging Face in
**GGUF** (llama.cpp) or **MLX** format. Small local models have small context
windows, so context management is a first-class feature everywhere.

UI language: **Polish** (all user-facing strings). Code, comments, identifiers,
model-facing system prompts: **English** (small models follow English
instructions better; prompts tell the model to answer in the user's language).

```
                 ┌──────────── macOS app (SwiftUI, macos/) ────────────┐
                 │ Czat │ Kod (agent) │ Odkrywaj │ Moje modele │ Runtime│
                 └──────────────┬──────────────────────────────────────┘
                                │ HTTP + SSE + WebSocket (127.0.0.1, bearer token)
 terminal: `localcode` (TUI)    ▼
 ┌──────────────────────── Python core (localcode/) ───────────────────────┐
 │ tui/  ──in-process──┐          server/ (aiohttp: /api/*, /v1/*)          │
 │                     ▼                  │                                 │
 │   agent/ (loop, context mgr, tools)    chat/ (db, rag, context, service) │
 │                     │                  │                                 │
 │                     ▼                  ▼                                 │
 │   runtime/ (RuntimeManager: instances, slots, RAM budget, LRU eviction;  │
 │             chat templates, tool-call parsing, memory estimation)        │
 │        │ spawns                        │ spawns                          │
 │   llama-server (GGUF)            mlx_worker (MLX, python -m ...)         │
 │   hub/ (store index + trust, HF search, downloads, library, metadata)    │
 └──────────────────────────────────────────────────────────────────────────┘
```

Python: 3.12 venv at `.venv/` (`.venv/bin/python`, `.venv/bin/pytest`).
Installed: mlx 0.32, mlx-lm 0.31.3, huggingface_hub 1.32, gguf 0.19, jinja2,
rich 15, prompt_toolkit 3.0.53, aiohttp 3.14, pytest 9 (+pytest-asyncio,
`asyncio_mode=auto`). `llama-server` 0.4.1 (build 10964) at
`/opt/homebrew/bin/llama-server`. Add deps to `pyproject.toml` if needed and
`pip install` them into `.venv`.

Test models on disk (downloaded by the orchestrator; tests needing them are
marked `@pytest.mark.real_model` and skip if the path is missing):

| key | path under `~/.localcode/models/` |
|---|---|
| MLX tiny  | `mlx-community/Qwen3-0.6B-4bit/` |
| MLX small | `mlx-community/Qwen3-4B-Instruct-2507-4bit/` |
| GGUF tiny | `unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf` |
| GGUF small| `unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| GGUF embed| `nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf` |

Hardware: Apple M5, 32 GB unified memory, macOS 26. Keep tests light: never
load more than ~6 GB at once, always stop spawned processes (fixtures with
`finally`).

---

## Shared contracts (already written — do not change signatures without need)

* `localcode/config.py` — paths (`home()`, `data_dir()`, …), `Settings`
  (`runtime: RuntimeSettings`, `agent: AgentSettings`, `chat: ChatSettings`,
  `server: ServerSettings`), `load_settings()`, `save_settings()`, `update_settings(patch)`.
* `localcode/hub/types.py` — `ModelMeta`, `LocalModel` (stable `key`), `MemoryEstimate`.
* `localcode/runtime/base.py` — `LoadConfig`, `SamplingParams`, `GenStats`,
  `CompletionChunk`, `TemplateInfo`, `Engine` ABC, `EngineError`.
* `localcode/events.py` — agent events (`to_dict()` → camelCase JSON with `type`).
* `localcode/agent/tools/base.py` — `Tool`, `ToolContext`, `ToolLimits`, `ToolResult`, `PermissionInfo`.

Message format used everywhere inside the core (OpenAI style):

```python
{"role": "system"|"user"|"assistant"|"tool",
 "content": str,
 "tool_calls": [{"id": str, "type": "function",
                 "function": {"name": str, "arguments": dict}}],   # assistant only; arguments is a dict
 "tool_call_id": str, "name": str,                                  # tool only
 "reasoning": str}                                                  # assistant, NEVER sent back to the model
```

---

## Module map and public APIs

### runtime/

* `runtime/llama_engine.py` — `class LlamaServerEngine(Engine)`: spawns
  `llama-server -m <gguf> --host 127.0.0.1 --port <free> -c <n_ctx*n_parallel>
  -np <n_parallel> -ngl <gpu_layers> -fa <flash_attn> --jinja
  [--cache-type-k/v …] [--mmproj …] [--embeddings --pooling …] --no-webui`.
  Uses `/health`, `/props` (n_ctx, chat template, modalities), `/apply-template`
  (fallback: render GGUF template locally via `runtime/templates.py`),
  `/tokenize`, `/completion` (stream, `cache_prompt: true`, `id_slot` not pinned),
  `/v1/embeddings` or `/embedding`. Logs to `logs/llama-<instance>.log`.
  `find_llama_server(settings) -> str | None`.
* `runtime/mlx_worker.py` — `python -m localcode.runtime.mlx_worker --model <dir>
  --port <p> --parallel N --ctx N [--kv-bits 8|4]`: aiohttp server exposing the
  **same** endpoints as llama-server that the engine uses: `GET /health`,
  `GET /props`, `POST /apply-template`, `POST /tokenize`, `POST /completion`
  (SSE `data: {"content": "...", "stop": false}` … final `{"stop": true,
  "stop_type"/"finish_reason", "timings": {...}, "tokens_evaluated",
  "tokens_cached", "tokens_predicted"}`). Parallel slots via `mlx_lm`
  `BatchGenerator` (or sequential queue if batching is unavailable for the arch),
  prefix reuse via `LRUPromptCache`/trimmable prompt cache. Sets
  `mx.set_wired_limit`/`set_memory_limit` from `--memory-limit-mb`.
* `runtime/mlx_engine.py` — `class MlxEngine(Engine)`: spawns the worker with
  `settings.runtime.python_path or sys.executable`, talks HTTP exactly like the
  llama engine (share the HTTP/SSE client code: `runtime/http_engine.py` base
  class `HttpEngine(Engine)` owned by the llama agent; the MLX agent subclasses it).
* `runtime/templates.py` — local Jinja rendering of HF/GGUF chat templates
  (sandboxed env, `raise_exception`, `strftime_now`, `tojson`), template
  capability detection → `TemplateInfo` (tools? tool role? thinking switch?).
  `render_template(template, messages, tools, add_generation_prompt, bos, eos, **kwargs) -> str`,
  `inspect_template(template) -> TemplateInfo`.
* `runtime/toolcalls.py` — tool-call and reasoning parsing for all common
  families (hermes/qwen `<tool_call>{json}</tool_call>`, qwen3-coder XML
  `<function=…><parameter=…>`, llama3 JSON / `<|python_tag|>`, mistral
  `[TOOL_CALLS]`, gpt-oss harmony, GLM `<arg_key>/<arg_value>`, DeepSeek,
  Gemma/function_gemma, kimi, minimax, pythonic, fenced JSON) plus a tolerant
  generic fallback with JSON repair:
  * `detect_tool_format(model: LocalModel, template: str | None) -> str`
  * `parse_output(text, *, tool_format=None, known_tools=None) -> ParsedOutput(content, reasoning, tool_calls: list[ToolCall(id, name, arguments: dict, raw)], errors: list[str])`
  * `class StreamSplitter(tool_format)`: `feed(delta) -> list[tuple[kind, text]]` with kind in
    `"text" | "reasoning"`; hides tool-call markup from the visible stream; `finish()` flushes.
  * `prompted_tools_block(tools: list[dict]) -> str` — compact tool instructions for
    templates without native tool support (model answers with `<tool_call>{"name":…,"arguments":{…}}</tool_call>`).
* `runtime/formatter.py` — `class ChatFormatter(engine)`: turns core messages
  (+ tool schemas) into a prompt string: native tools when
  `template_info.supports_tools`, else `prompted_tools_block` injected into the
  system message; tool results as role `tool` or wrapped
  `<tool_response>` user turns; strips `reasoning` from history; thinking on/off
  via `template_info.thinking_kwarg`. `async render(messages, tools, *, thinking: bool) -> str`,
  `async count(messages, tools) -> int` (exact, via engine tokenizer, cached per message).
  Also `stop_strings()`.
* `runtime/memory.py` — `system_memory() -> dict(total, available, used)`
  (psutil/vm_stat), `default_ram_budget(settings) -> int`,
  `estimate(model: LocalModel, config: LoadConfig) -> MemoryEstimate` (weights =
  file size; KV = 2·n_layers_full·n_kv_heads·head_dim·bytes(kv type)·n_ctx·n_parallel
  (+ SWA layers capped at window); overhead ≈ 0.4 GB + 5 %), `max_ctx_that_fits(model, budget, n_parallel) -> int`,
  `auto_ctx(model, settings) -> int`.
* `runtime/manager.py` — `class RuntimeManager`:
  ```python
  RuntimeManager(settings: Settings, library: ModelLibrary, on_event: Callable[[dict], None] | None = None,
                 engine_factory: Callable[[str, LocalModel, LoadConfig], Engine] | None = None)
  async acquire(model_key, *, n_ctx=None, purpose="chat") -> Lease   # async ctx manager; lease.engine; holds 1 slot
  async load(model_key, config: LoadConfig | None = None, *, new_instance=False) -> InstanceInfo
  async unload(instance_id) ; async unload_all() ; async shutdown()
  status() -> dict            # RuntimeStatus DTO (camelCase, see HTTP API)
  estimate(model_key, config=None) -> MemoryEstimate
  async update_settings(patch: dict) -> dict  # persists via config.update_settings, applies limits
  ```
  Rules: JIT load on `acquire`; reuse a ready instance of the model with a free
  slot and `n_ctx >= requested`; otherwise start another instance if
  `max_instances_per_model` allows, else wait for a slot (FIFO). Before loading,
  enforce `max_loaded_models` and the RAM budget: evict idle instances LRU
  (`evict_policy="lru"`), never evict busy ones; if it still cannot fit, raise
  `EngineError` with a Polish explanation. Idle TTL reaper. Emits
  `{"type":"runtime", ...status}` via `on_event` on every change.

### hub/

* `hub/metadata.py` — `read_gguf_meta(path) -> ModelMeta` (gguf.GGUFReader,
  metadata only, fast), `read_mlx_meta(dir) -> ModelMeta` (config.json +
  tokenizer_config.json / chat_template.jinja), `detect_capabilities(meta, files) -> list[str]`.
* `hub/library.py` — `class ModelLibrary(settings)`: `scan() -> list[LocalModel]`
  (models_dir + LM Studio + HF cache, GGUF shards grouped, mmproj detected,
  metadata cached in `~/.localcode/data/library-cache.json` keyed by path+mtime),
  `list()`, `get(key) -> LocalModel | None`, `resolve(query) -> LocalModel | None`
  (fuzzy by key/name), `delete(key)`, `load_defaults(key)`/`set_load_defaults(key, dict)`.
* `hub/search.py` — `class HubClient(settings)`: `search(query, format="any"|"gguf"|"mlx",
  sort="trending"|"downloads"|"likes"|"recent", limit=30) -> list[dict HubModel]`,
  `details(repo) -> dict HubModelDetail` (files with sizes, quant variants grouped incl.
  split GGUFs, mmproj, memory estimate per variant, recommended variant, README excerpt),
  `curated() -> list[dict]` (from `hub/curated.py`).
* `hub/store_data.py` — the shipped store index: `search(query, category=, fmt=, min_score=,
  official_only=, limit=) -> list[dict]`, `get(repo)`, `trust_of(row) -> Trust`,
  `count()`, `generated_at()`, `load_index(path)`.  Reads
  `localcode/hub/data/catalog.json.gz` (thousands of installable models, facts only)
  lazily; a missing or damaged file means "no extra models", never an error.
* `hub/publishers.py` — who published a model: `classify(repo, base_models=, tags=) -> Provenance`
  (`official` | `trusted` | `community` | `unknown`, with the laboratory, the packager and the
  upstream repository), `family_publisher(name)`, `OFFICIAL_PUBLISHERS`, `TRUSTED_QUANTIZERS`.
* `hub/trust.py` — `evaluate(repo, …) -> Trust` (score 0-100, band, Polish reasons and
  warnings) from provenance, file formats (pickle vs GGUF/safetensors), licence, popularity
  and last update; `summarize_files(siblings)` condenses a file list for the index.
  Both `PICKLE_SCORE_CAP` and `UNKNOWN_ORIGIN_CAP` sit below the "sprawdzone" band, so
  neither pickles nor an untraceable origin can present as trusted.
* `hub/download.py` — `class DownloadManager(settings, on_event)`: `start(repo, files) -> dict job`,
  `cancel(job_id)`, `list() -> list[dict]`; parallel-safe, resumable (hf `local_dir`
  into `<models_dir>/<org>/<repo>/`), progress (bytes/speed) via polling file sizes or
  a custom `tqdm_class`; emits `{"type":"download","job":{...}}`; on completion triggers
  `library.scan()`.

### agent/

* `agent/tools/*.py` — `read_file`, `write_file`, `edit_file` (exact →
  whitespace-tolerant → fuzzy match, unified diff), `list_dir`, `glob`, `grep`
  (ripgrep if present, else Python), `bash` (timeout, output truncation, process-group
  kill, cwd sticky), `todo_write`. `agent/tools/__init__.py`: `default_tools() -> list[Tool]`,
  `tools_for_mode(mode) -> list[Tool]`.
* `agent/permissions.py` — modes `default` (reads auto, writes+bash ask),
  `accept_edits` (edits auto, bash ask), `auto` (decided like default, but what would ask
  is judged by a reviewer model — `agent/autoreview.py`), `yolo` (all auto), `plan`
  (read-only tools only);
  read-only bash allowlist (`ls`, `cat`, `git status/diff/log`, `rg`, …); "always allow"
  memory per session (per tool / per command prefix).
* Auto mode (`agent/autoreview.py`): a call that would ask goes to a reviewer model —
  `settings.agent.auto_reviewer_model` (Settings, `autoReviewerModel`), else the agent's own
  model — which answers `TAK`/`NIE` + `Co robi: …` + `Powód: …`. The prompt is the agent's
  messages up to (not including) the assistant message with the pending calls, plus one user
  message with the call and the reviewer's rules, same tools, temperature 0, thinking off
  (unless switching it off would change the prompt prefix, e.g. gpt-oss) — so with the agent's
  own model it reuses the step's prefix cache. `TAK` runs the call; `NIE`, an unclear answer,
  a tool call, an engine error or a timeout → the normal permission question, whose
  `PermissionRequest.review = {verdict, reason, model, guard?}` shows the verdict on the card
  (the user may overrule a `NIE`; declined → the agent reads the reviewer's reason; with no one
  to ask a `NIE` refuses as before). Calls `PermissionPolicy.needs_human()` names
  (`⚠ dane do dostawcy`, `⚠ niebezpieczne`, catastrophic) always go to the user; they are still
  reviewed, as advice for the card only (no `auto_review` event, a `TAK` approves nothing).
* Auto mode's local guard (`agent/guard.py`, `settings.agent.auto_guard_model` /
  `autoGuardModel`, local keys only — the API refuses `remote:…`): every shell command and every
  call reaching outside the project is also judged, at the same time as the reviewer, by a
  local model that sees no conversation — a fresh prompt with short rules and the raw command
  (or arguments) in an escaped `<action>` block plus the project path; never the assistant's
  text, `bash`'s `description` or the reviewer's reason. Threat model: a remote agent model and
  its provider may be hostile. Automatic choice: the agent's model when local, else the help
  bot's model, else the smallest local chat model. Fail-closed: the call runs on its own only
  when both say `TAK`; no local model, unclear answer, error or timeout → the user is asked.
  Its opinion travels as `guard: {verdict, reason, model}` on `auto_review`, the stored item's
  `review` and the request's `review`.
* Precedents (`agent/precedents.py`, `settings.agent.auto_precedents_enabled` /
  `autoPrecedentsEnabled`): every user answer on a card for a shell command or an outside-project
  call (any mode; not a `dismissed` card nobody answered, not a yes to a `needs_human` call) is
  stored as parser-computed keys only (`command_prefix` per non-read-only simple command, or
  `"<tool> <real dir>"`; nothing for substitution/redirection/inline code) in
  `<LOCALCODE_HOME>/auto_precedents.json` per project realpath (LRU 100, 0600); auto mode shows
  the reviewer the matching ones (≤5, before its rules) and the guard a `<history>` block
  (matching first, then recent, ≤8) — never for `needs_human`, both `TAK` still required.
  `GET/DELETE /api/agent/auto-precedents[?cwd=]` lists / clears (no `cwd`: all projects).
* Folder confinement (`settings.agent.confine_to_project`, on by default): `bash`
  runs under `sandbox-exec` with `security/sandbox.py:shell_profile(project)` — no reading
  the home directory beyond the project and tool chains, no writing beyond the project,
  temp dirs and build caches. `permissions.outside_paths()` flags a command that names a
  path outside (`~`, `$HOME`, `/Users`, `/Volumes`, `/`, `..`) or passes
  `outside_project=true` with `⚠ poza projektem`, which asks in every mode; an approved
  call runs unsandboxed. File tools already ask for any path outside the project.
* `agent/context.py` — `class ContextManager(n_ctx, counter)`: budgets
  (reserve for output, system, tools, history), `limits() -> ToolLimits` scaled by n_ctx,
  truncation of tool outputs (head+tail with marker), progressive shrinking:
  1) mask old tool results (keep last K verbatim), 2) summarize old turns via the model
  (compaction, keeps task + decisions + files touched + todo), 3) drop oldest.
  `async fit(messages, tools) -> (messages, list[Compaction])`, `usage(...) -> ContextUpdate`.
* `agent/prompts.py` — compact system prompt(s) scaled to n_ctx (tiny ≤8k, small ≤32k, large),
  project instructions loading (`LOCALCODE.md`, `AGENTS.md`, `CLAUDE.md`, truncated to budget),
  environment block (cwd, OS, date, git branch), compaction prompt.
* `agent/loop.py` — `class Agent(runtime, model_key, cwd, settings, on_event, ask_permission)`:
  `async run_turn(user_text) -> None` (streams events), `interrupt()`, `set_mode()`,
  `set_model()`, `compact()`, `clear()`; loop guard for repeated identical calls,
  malformed call recovery, max steps. `agent/session.py` — JSONL persistence/resume.
* `agent/skills.py` — instructions loaded on demand (`SKILL.md`, and an optional
  `SKILL-mini.md` for a small window: `variant_for(strong, n_ctx)`, `MINI_CTX = 16384`).
  Only the names and one-line descriptions sit in the prompt; the menu itself is
  budgeted for a small model and falls back to bare names. See `docs/SKILLS.md`.
* `helpbot/` — Pomocnik: the agent loop with `system_prompt=` and `tool_state=`, ten tools over
  `HelpbotOps` (a facade on `Core`: models, downloads, load config, every setting, doctor). A
  session is the help bot's when its cwd is `<LOCALCODE_HOME>/helpbot`. See `docs/HELPBOT.md`.
* Effort `ultracode` (`runtime/templates.py` `ULTRACODE`, `AGENT_EFFORT_CHOICES`): the agent's
  level above `high`, offered in the Kod header ("Ultracode"), `/think ultracode`, `--effort` and
  `setEffort`; the chat maps it to `high`. Templates and engines never see it
  (`engine_effort()` → `high`). With it the agent gets `delegate` even when
  `settings.agent.subagents` is off (not below a 16k window — logged), an orchestrator section
  in the prompt ("fan out FIRST": background helpers for every independent part in the first
  step, wait=false by default, verify, helpers review the changed files), the `delegate` calls
  of one step run concurrently (`settings.agent.ultracode_parallel`, default 6, also the cap on
  background helpers; other tools stay sequential, results are appended in call order, each
  helper gets a `ToolContext.call_id` so its `subagent_*` events stay apart; `interrupt()`
  cancels them all), helpers get `subagent_max_steps × 2` and up to 20 per turn, still depth 1.
  `Agent._nudge_to_delegate`: after `ULTRACODE_NUDGE_STEPS` (4) steps with tool calls and no
  `delegate`, a one-line user note (`prompts.ULTRACODE_NUDGE`), not again before 4 more, at
  most 5 a turn, never while helpers run in the background.
* `agent/tools/shell.py` — `bash` takes an optional `description` ("what the command does");
  the permission preview is `Co robi: <description>`, a blank line, the command (then the
  outside-project note), and the card title is `Polecenie: <description>`. The policy reads the
  command from the arguments, never from the preview or the title.
* `agent/tools/subagent.py` — `delegate(task, wait?)`: one self-contained task handed to a
  helper `Agent` ("h1", "h2" … per session) with its own context, `session=None` and a menu of
  `read_file`/`list_dir`/`grep`/`glob`/`bash` (no `bash` in plan mode; never `write_file` or
  `edit_file`). Off unless `settings.agent.subagents` or ultracode. The helper's calls go
  through the parent's `PermissionPolicy` object, mode and approver (`SubagentHost.bind`, set
  every step): yolo runs, plan denies, default asks, auto reviews (reviewer + local guard +
  risk facts). Its `permission_request` (title "Pomocnik h2: …", `helper: "h2"`) and
  `auto_review` (`helper`) are forwarded to the client; everything else is wrapped as
  `subagent_start/step/end` — a helper's `turn_end` would end the parent's turn. Each helper
  has its own shell cwd and a provider cache key `<parent key>-h2`. Never deeper than one
  level; runs on a model only the user may choose.
  `wait=false` (the default under ultracode): the call returns at once ("Pomocnik h2 pracuje
  w tle"), `SubagentHost.spawn` runs the helper as a task; `subagent_start/end` carry
  `background: true`, and the end carries the `report`. At every step boundary
  `Agent._take_helper_reports` appends finished reports as a user-role note
  ("[Background helper h2 finished — its report:] …"); when the model ends its turn while
  helpers still run, `_wait_for_helpers` waits (interruptibly) for the next one and the turn
  continues; at turn end anything still running is cancelled and named in a notice.
  `interrupt()` cancels every background helper. The app shows "w tle" on the card and
  "Pomocnicy w tle: N" in the header and queues several pending permission questions.

### chat/

* `chat/db.py` — SQLite (WAL) schema: folders, conversations, messages, attachments,
  memories, kb_documents, kb_chunks (+ FTS5 over chunks and messages, embeddings BLOB),
  settings (AppSettings JSON). `chat/repo.py` — CRUD returning ai_ux-shaped dicts.
* `chat/context.py` — port of ai_ux `src/lib/context.ts` (shared vs isolated memory,
  global + folder prompts, profile, memories, RAG citations) with a **token budget**
  (history trimmed newest-first, memories/RAG capped).
* `chat/rag.py` — chunking (port of `chunk.ts`), FTS5 BM25 retrieval, optional embeddings
  (GGUF embedding model via runtime) with reciprocal-rank fusion.
* `chat/extract.py` — text extraction (txt/md/code/csv/json/html/pdf/docx).
* `chat/service.py` — `ChatService.send(req) -> AsyncIterator[dict]` producing ai_ux SSE
  frames (`start`, `title`, `citations`, `reasoning`, `text`, `usage`, `done`, `error`),
  title generation, optional memory extraction, stop/regenerate/edit.

### server/, tui/, cli

* `server/app.py` — `create_app(core: Core) -> web.Application`; `server/core.py` — `Core`
  holding settings, library, hub, downloads, runtime, chat service, agent sessions.
  `localcode serve [--port] [--token] [--parent-pid]` prints `LOCALCODE_READY port=<p>` on stdout.
* `tui/` — prompt_toolkit + rich REPL (agent mode default, `/chat` mode), slash commands,
  model browser/downloader, runtime panel. `cli.py` — argparse entry (`localcode`, `lc`).

---

## HTTP API (server ↔ Swift app)

Base `http://127.0.0.1:<port>`. All `/api/*` except `/api/health` require
`Authorization: Bearer <token>` (WebSocket: header or `?token=`). JSON keys are
camelCase. Timestamps are **milliseconds** since epoch (as in ai_ux). Errors:
HTTP 4xx/5xx with `{"error": "<Polish message>", "code": "<slug>"}`.

### Chat API — identical to ai_ux (`clients/apple/Sources/Shared/Networking/APIClient.swift`, `Models/Models.swift`)

`GET /api/bootstrap` → `Bootstrap` (`user: null`, `providers: [{provider:"local",…}]`,
`activeProvider: "local"`, `models`: local chat-capable models as `ModelInfo`
{id: key, provider:"local", label, contextWindow, maxOutput, supportsVision,
supportsReasoning, effortLevels: [string] (`[]` = no reasoning-effort control;
`["off","on"]` Qwen3-style; `["low","medium","high"]` gpt-oss),
inputPricePerM:null, outputPricePerM:null, description}).
`GET/POST /api/conversations[?folderId]`, `GET/PATCH/DELETE /api/conversations/{id}`,
`GET /api/conversations/{id}/export?format=md|json`,
`POST /api/conversations/{id}/title[?auto=1]`, `POST /api/conversations/{id}/compact`,
`DELETE /api/messages/{id}`,
`GET/POST /api/folders`, `PATCH/DELETE /api/folders/{id}[?deleteChats=1]`,
`GET/POST /api/memories?scope&folderId`, `PATCH/DELETE /api/memories/{id}`,
`GET/POST(multipart files, scope, folderId) /api/knowledge`, `DELETE /api/knowledge/{id}`,
`POST /api/upload` (multipart) → `[Attachment]`, `GET /api/attachments/{id}`,
`GET/PATCH /api/settings` (`AppSettings`), `GET /api/providers`, `GET /api/models[?refresh=1]`,
`GET /api/search?q=` → `[SearchHit]`,
`Conversation` carries `reasoningEffort: "off"|"low"|"medium"|"high"|null` (null = follow
`settings.chat.reasoningEffort`); it is accepted on `PATCH /api/conversations/{id}` and as
`reasoningEffort` in `ChatSendRequest`. A model whose template has only a boolean
thinking switch advertises `effortLevels: ["off","on"]`, so `"on"` is accepted as a
request everywhere too and stored as the canonical `"medium"`.
`/v1/chat/completions` honours `reasoning_effort`.
`GET/PATCH /api/app-settings` carry `agent.reasoningEffort` and `chat.reasoningEffort`.
`POST /api/chat` (SSE, body `ChatSendRequest`) frames:
`{"type":"start","conversationId","userMessage","assistantMessageId"}`,
`{"type":"title","conversationId","title"}`, `{"type":"citations","citations":[…]}`,
`{"type":"reasoning","delta"}`, `{"type":"text","delta"}`,
`{"type":"usage","tokensIn","tokensOut","costUsd":0}`, `{"type":"done","message"}`,
`{"type":"error","message","code"}`. Extra frames (new, ignored by old clients):
`{"type":"status","message"}` (e.g. "Ładowanie modelu…"),
`{"type":"compaction","before","after","strategy"}` and
`{"type":"context","used","limit",…}` (the chat's own context gauge).
A model that is missing or cannot chat is reported as an `error` frame *inside*
the stream (the user message is stored first); a malformed request body is a
plain `{"error","code","details"}` JSON response instead.

### LocalCode API (new)

```
GET  /api/health                         → {ok:true, version}
GET  /api/events                         SSE: {"type":"download","job":DownloadJob} |
                                              {"type":"runtime", ...RuntimeStatus} |
                                              {"type":"library"}   (library changed)
GET  /api/hub/search?q&format=any|gguf|mlx&sort=trending|downloads|likes|recent&limit
     → [HubModel {id, author, name, format:"gguf"|"mlx", downloads, likes, updatedAt,
                  tags, pipelineTag, paramsB, gated, curated, description}]
GET  /api/hub/curated                    → [HubModel]
GET  /api/hub/model?repo=                → HubModelDetail {model: HubModel, readme,
        contextLength, arch, variants: [HubVariant {id, label, quant, files:[string],
        sizeBytes, estimate: MemoryEstimate, recommended, isVisionProjector}],
        files: [{name, size}]}
POST /api/library/import {path, name?, move?} → LocalModelDTO (a model already on disk)
GET  /api/catalog?category&q&installed&limit&minTrust&official&sort
     → [CatalogEntry {id, name, tagline, category, format,
        sizeBytes, sizeLabel, fit, fitDetail, speedHint, installed, installedKey, recommendedVariant,
        variants: [CatalogVariant], badges, source, alternatives, requiresToken, curated,
        downloads, likes, watchOut,
        trust: {score 0-100, level, label, reasons[], warnings[], license, gated, components},
        publisher: {kind, publisher, packager, upstream, official, labKnown, label, detail}}]
     (the friendly in-app store, served from the shipped index + staff picks + a live search;
      header X-Catalog-Offline: 1 when answered without the network,
      header X-Catalog-Total: how many models matched before `limit`, default limit 200)
GET  /api/catalog/lookup?ref=            → CatalogEntry for a pasted repo id or HF URL
                                           (external:true when it is not in our catalogue)
GET  /api/catalog/{id}                   → one CatalogEntry with its full variant list
GET  /api/logos/{slug}                   → PNG: the laboratory's own mark (publisher.logo)
POST /api/catalog/install {id, variantId?} → DownloadJob
GET  /api/agent/commands                 → {commands:[{name, summary, usage, aliases, values}],
                                            initPrompt}  (the slash commands a graphical client
                                            offers — one table, shared with the terminal:
                                            localcode/tui/commands.py, field in_app)
GET  /api/helpbot                        → {model, modelKey, displayName, sizeBytes,
                                            state: ready|downloading|missing, loaded, download,
                                            cwd, autoDownload}  (the help bot, localcode/helpbot)
POST /api/helpbot/install                → 202 + status (download started; checks free disk
                                            space first, 409 when there is no room)
POST /api/helpbot/session {new?}         → AgentSession of the help bot (the latest one, or a
                                            fresh one); talk to it over the usual agent WS
GET  /api/remote/providers               → [{id, label, hasKey, enabled, baseUrl, keyUrl, modelCount}]
GET  /api/remote/models?provider&refresh → [RemoteModel {key:"remote:<provider>:<id>", provider,
                                            providerLabel, modelId, displayName, contextLength,
                                            supportsTools, remote:true}]
     (providers: openai, anthropic, gemini, deepinfra, unlid — all but anthropic/gemini speak
      OpenAI's dialect; a gateway that nests its metadata under its own name, as unlid.ai does
      with {"unlid": {context_length, supports_tools, available}}, is read from there too)
PUT  /api/remote/{provider}/key {key, baseUrl?}  → providers (the key is verified before it is saved)
PATCH /api/remote/{provider} {enabled}   → providers
POST /api/speech/transcribe {path, modelKey?, language?, segments?, srt?}
     → Transcript {text, language, durationSeconds, segments:[{start,end,text}], modelKey, elapsedMs, srt?}
POST /api/speech/speak {text, modelKey?, voice?, speed?, output?}
     → VoiceResult {path, text, voice, durationSeconds, sampleRate, modelKey, elapsedMs, sizeBytes}
GET  /api/speech/voices?modelKey=        → [{id, label, language}]
GET  /api/video/models                   → [LocalModel DTO + {videoKind:"ltx-2b"|"ltx-other"|"other-video"|"unknown",
                                            supported}]      (video-looking models on disk, supported or not)
POST /api/video/estimate {width?,height?,numFrames?,fps?,steps?}
     → {seconds, message, width, height, numFrames, fps, steps}
     (rough, honestly-unverified time estimate — no prompt required, see docs/VIDEO.md)
POST /api/video/generate {prompt, modelKey?, negativePrompt?, width?, height?, numFrames?, fps?,
        steps?, guidance?, seed?, force?, output?} → SSE
     frames {"type":"progress", ...VideoProgress{stage,step,steps,elapsedMs,message,fraction}}
          | {"type":"video","video":VideoResult} | {"type":"done"} | {"type":"error","message","code"}
     (closing the connection cancels the job, same shape as /api/images/generate; see docs/VIDEO.md)
GET  /api/downloads                      → [DownloadJob {id, repo, files, label, format,
        totalBytes, doneBytes, speedBps, state:"queued"|"downloading"|"done"|"error"|"cancelled",
        error, startedAt, finishedAt, modelKey}]
POST /api/downloads {repo, variantId? , files?} → DownloadJob
DELETE /api/downloads/{id}               → cancel
GET  /api/library                        → [LocalModel.to_dict() + {loaded:bool, loadDefaults: LoadConfigDTO|null,
                                            effortLevels: [string]}]   # [] = model has no reasoning-effort control
POST /api/library/rescan                 → same list
PATCH /api/library/{key} {loadDefaults}  → model DTO           (key is URL-encoded, may contain "/")
DELETE /api/library/{key}                → 204 (only source=="localcode")
GET  /api/runtime                        → RuntimeStatus {
        instances: [{id, modelKey, displayName, format, state:"loading"|"ready"|"busy"|"error"|"stopping",
                     nCtx, nParallel, busySlots, memoryBytes, estimatedBytes, loadedAt, lastUsedAt,
                     pid, port, genTps, error}],
        system: {totalBytes, usedBytes, availableBytes},
        budget: {limitBytes, usedBytes},
        settings: {maxLoadedModels, maxInstancesPerModel, defaultParallel, ramLimitGb, idleTtlMinutes,
                   defaultCtx, flashAttn, kvCacheType, gpuLayers, evictPolicy}}
PATCH /api/runtime/settings {partial settings}  → RuntimeStatus
POST /api/runtime/load {modelKey, config?: LoadConfigDTO, newInstance?} → Instance (state may be "loading")
POST /api/runtime/unload {instanceId} | {all:true} → RuntimeStatus
POST /api/runtime/estimate {modelKey, config?}   → MemoryEstimate.to_dict()
POST /api/runtime/tune {modelKey?, useModel?, apply?} → {recommendation: TuneProposal DTO
        {machine, modelKey, source:"deterministic"|"model", note, recommendations:
        [{key, value, reason, basis}], patch}, status?: RuntimeStatus}
     (runtime/tuner.py: a deterministic recommendation per "Zasady ładowania" knob, each with a
      Polish reason and the number it was derived from; useModel defaults to true and asks the
      already-loaded modelKey for a second opinion, clamped to validate_settings_patch's ranges;
      apply:true writes the result through RuntimeManager.update_settings, present as status)
GET  /api/app-settings                   → {modelsDir, hfTokenSet, language, scanLmstudio, scanHfCache,
                                            favoriteModels:[key], version, agent:{…}, chat:{defaultModel, embeddingModel,
                                            reasoningEffort}, server:{host, port, openaiApiEnabled,
                                            openaiApiRequiresToken, corsOrigins},
                                            catalog:{categories:[{id,label}], sourceNote, offlineNote}}
PATCH /api/app-settings {partial}        → same ; PUT /api/app-settings/hf-token {token|null}
     (all three answer with the *whole* DTO, ``catalog`` included: the Mac app replaces its settings
      object with the body it gets back)
     cerber:{enabled, url, tokenSet, installed, block} — Cerber (the user's antivirus) checks every
     download first: PATCH {cerber:{enabled, url}} (url must be http://127.0.0.1|localhost:PORT),
     PUT /api/app-settings/cerber-token {token|null} (refused only when Cerber itself rejects it),
     POST /api/cerber/check → {ok, message, installed}. A download job then has stage "cerber"
     before "download"; a blocked one fails with code cerber_blocked, a closed Cerber with
     cerber_unavailable (fails closed). See localcode/security/cerber.py.
     favoriteModels — "Ulubione", model keys (local or remote:provider:id) every picker lists
     first: PUT /api/app-settings/favorite-models {key, favorite:bool} stars or unstars one
     (read-modify-write of settings.json, so the app and the terminal keep each other's stars);
     PATCH {favoriteModels:[key]} replaces the whole list. GET /api/models (the chat picker)
     returns them first too.

POST /api/agent/sessions {cwd, modelKey?, mode?} → AgentSession {id, cwd, modelKey, mode, title,
                                                   createdAt, updatedAt, contextUsed, contextLimit, busy,
                                                   effort, effortLevels, goal}
     goal = null | {text, state:"active"|"done"|"blocked"|"limit"|"cleared", turns, maxTurns, summary}
GET  /api/agent/sessions[?cwd=]          → [AgentSession]
GET  /api/agent/sessions/{id}            → {session: AgentSession, items: [AgentItem], todos: [Todo]}
     AgentItem = {kind:"user"|"assistant"|"reasoning"|"tool"|"notice", id, text?, toolName?, title?,
                  arguments?, ok?, summary?, output?, diff?, createdAt}
DELETE /api/agent/sessions/{id}
GET  /api/agent/sessions/{id}/ws  (WebSocket)
   client → {"type":"user","text"} | {"type":"permission","requestId","decision":"allow"|"deny"|"always"}
          | {"type":"interrupt"} | {"type":"setMode","mode"} | {"type":"setModel","modelKey"}
          | {"type":"setEffort","effort":"off"|"low"|"medium"|"high"|"ultracode"}
          | {"type":"compact"} | {"type":"clear"}
          | {"type":"setGoal","text","message"?} (/goal: set it, then run `message` or `text` as a turn)
          | {"type":"clearGoal"}
   server → events.py dicts (turn_start, text_delta, reasoning_delta, tool_call, tool_result,
            permission_request, auto_review, context, compaction, status, todos, goal, turn_end, error)
            auto_review = {callId, tool, verdict: allow|deny|ask, reason, reviewerModel, guard?}: auto
            mode's reviewer on one call (also stored on the tool item as review {verdict, reason, model,
            guard?}); guard = {verdict, reason, model} of the context-free local guard.
            permission_request = {requestId, tool, title, preview, options, review?}: review = what
            auto mode's reviewer (and guard) said before the user is asked.
            goal = {text, state, turns, maxTurns, summary}: set, sent back to work after a
            final answer ("active", turns n), reached ("done"), "blocked", "limit", "cleared".
            The whole run is ONE turn (one turn_start / turn_end); Esc ends it.
            + {"type":"session","session":AgentSession} on connect / change

OpenAI-compatible (like LM Studio's local server; optional token; browser Origins rejected unless allowed):
GET /v1/models · GET /v1/models/{model} · POST /v1/chat/completions (stream + tools) · POST /v1/completions · POST /v1/embeddings
· POST /v1/images/generations (see docs/IMAGES.md)
```

Server process: `localcode serve --port P --token T --parent-pid PID` → prints
`LOCALCODE_READY port=P` when listening; writes `~/.localcode/server.json`
(0600); exits when the parent pid dies.

---

## Context strategy for small models (applies to agent AND chat)

1. Know the real window: `engine.n_ctx` (per slot); budget = n_ctx − reserve_output.
2. Keep the prompt prefix stable (system + tools first, append-only history) so the
   engine's prefix cache is reused; only compaction rewrites history.
3. Small tool schemas and a compact system prompt; tiny-context mode (≤8k) uses an
   even shorter prompt and fewer tools.
4. Every tool output is truncated at insertion (head+tail, explicit
   "[… N lines omitted — use offset/limit or grep …]" markers); read_file pages by lines.
5. Before each request: if over the soft threshold → mask old tool outputs → summarize
   old turns → drop oldest. Report every step as a `compaction` event.
6. Reasoning (`<think>`) is shown but never re-sent.
7. Chat: history trimmed to budget; memories/RAG chunks given a fixed share of the budget.

## Conventions

* Every module ships tests in `tests/<area>/test_*.py`; `.venv/bin/pytest tests/<area> -q` must pass.
  Real-model tests: `@pytest.mark.real_model`, skip when the file is missing.
* No global state except explicit singletons in `server/core.py`.
* asyncio everywhere in the core; blocking work (`gguf` parsing, file IO, sqlite) via
  `asyncio.to_thread` when on a request path.
* Polish user-facing strings; English model-facing prompts.
