# Fixtures — real payloads of the LocalCode model, download, runtime and settings API

Captured with `curl` from a running server (`.venv/bin/localcode serve --port 0
--token test`) on the development Mac, and used as the reference the Codable
DTOs in `Sources/ModelsFeature/Store/ModelsDTO.swift` were written against.
Where these bytes and `docs/ARCHITECTURE.md` disagreed, the bytes won.

| file | endpoint | what it pins down |
|---|---|---|
| `api_library.json` | `GET /api/library` | 41 models from all three sources (`localcode`, `lmstudio`, `hf-cache`); note `modifiedAt` is in **seconds**, unlike every other timestamp in the API |
| `api_runtime.json` | `GET /api/runtime` | empty `instances`, `system`/`budget`/`settings`; `ramLimitGb` and `defaultCtx` are `null` when automatic |
| `api_events_runtime.json` | `GET /api/events` | a `{"type":"runtime", …}` frame — the `RuntimeStatus` is inline, not nested under a key |
| `api_downloads.json` | `GET /api/downloads` | the empty queue |
| `api_downloads_jobs.json` | `GET /api/events` | the last frames of a real download: `queued` → `downloading` (with `speedBps`, `etaSeconds`) → `done` |
| `api_app-settings.json` | `GET /api/app-settings` | `modelsDir`, `hfTokenSet`, and the nested `agent` / `chat` / `server` objects |
| `api_hub_curated.json` | `GET /api/hub/curated` | 30 curated entries; `downloads`, `likes` and `updatedAt` are `null` here, and `category` / `minRamGb` / `recommendedQuant` only exist on curated entries |
| `api_hub_search.json` | `GET /api/hub/search?q=qwen3` | search hits with real counters and a millisecond `updatedAt` |
| `api_hub_model.json` | `GET /api/hub/model?repo=unsloth/Qwen3-4B-Instruct-2507-GGUF` | 26 variants, each with its own `MemoryEstimate`, plus `readme` and `files` |
| `api_hub_model_gated.json` | `GET /api/hub/model?repo=google/gemma-3-1b-it-qat-q4_0-gguf` | `gated: true` and the ready-made Polish `accessError` sentence the detail pane shows |
| `api_runtime_estimate.json` | `POST /api/runtime/estimate` | the estimate the load sheet renders while the context slider moves |

There is no Swift test target in this package (see the module notes); the DTOs
are verified by decoding the live server's answers in the render harness and by
the smoke run described in the ModelsFeature hand-off.
