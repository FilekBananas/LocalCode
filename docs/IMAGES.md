# LocalCode — image generation (addendum to ARCHITECTURE.md)

Local text-to-image, in the same shape as the LLM side: models are downloaded
from Hugging Face into the LocalCode library, loaded into a managed instance
that shares the RAM budget, and driven from the Mac app ("Obrazy"), the terminal
(`localcode image`) and the local OpenAI-compatible API.

Two backends, mirroring GGUF/MLX:

| format | runtime | models |
|---|---|---|
| `mlx` | **mflux** ≥0.20 (pip, Apple Silicon) | FLUX.1 schnell/dev/Krea, FLUX.2 klein, Qwen-Image, … (`*-mflux-4bit`, `mlx-community/Qwen-Image-*`) |
| `gguf` | **stable-diffusion.cpp** via `stable-diffusion-cpp-python` ≥0.4 (Metal) | SD 1.5, SDXL, SD 3.5, FLUX GGUF (`second-state/*`, `gpustack/*`, `city96/*`) |

Both run in their own **worker subprocess** (like the LLM engines), so unloading
frees memory and a crash cannot take the app down.

## Model identification

* `LocalModel.capabilities` contains `"image"`; `ModelMeta.image_family` is one of
  `flux`, `flux2`, `qwen-image`, `sd1`, `sdxl`, `sd3`, `unknown`, and
  `ModelMeta.image_components` maps `clip_l`/`t5xxl`/`vae`/`unet` to file paths for
  multi-file GGUF setups (empty when the file is self-contained).
* Detection: MLX — `model_index.json` / a `transformer/` + `vae/` layout, or an
  mflux repo layout; GGUF — diffusion architectures in the GGUF metadata
  (`sd`, `sdxl`, `sd3`, `flux`) or a `text-to-image` pipeline tag recorded at
  download time. Companion files (CLIP/T5/VAE) that are not models on their own
  never appear as separate library entries.

## Python API

`localcode/images/`:

```python
@dataclass
class ImageParams:
    prompt: str; negative_prompt: str = ""; width: int = 1024; height: int = 1024
    steps: int | None = None          # None = model default (FLUX schnell 4, SD 20…)
    guidance: float | None = None
    seed: int | None = None           # None = random, echoed back in the result
    count: int = 1
    init_image: str | None = None; strength: float = 0.6   # img2img when supported

@dataclass
class ImageResult:
    id: str; path: str; width: int; height: int; seed: int; model_key: str
    prompt: str; steps: int; guidance: float | None; elapsed_ms: int; created_at: float

class ImageEngine(ABC):          # images/base.py — worker subprocess, HTTP like runtime/http_engine.py
    async def start() / stop(); running
    def generate(params) -> AsyncIterator[ImageProgress | ImageResult]   # progress: step, steps, preview_path?
    async def memory_bytes() -> int | None

class ImageRuntime:              # images/runtime.py
    def __init__(self, settings, library, on_event=None, runtime: RuntimeManager | None = None)
    async def generate(model_key, params, *, on_progress=None) -> list[ImageResult]
    def status() -> list[dict]    # instance DTOs, same shape as RuntimeManager's with "kind": "image"
    async def unload(instance_id) / unload_all() / shutdown()
```

`ImageRuntime` keeps at most `settings.images.max_loaded` (default 1) image
models in memory, reports its instances into `/api/runtime`, and before loading
asks `RuntimeManager.make_room(bytes)` (new, additive method) to evict idle LLM
instances so the shared RAM budget still holds. Generation is serialised per
instance (one image job at a time), cancellable, and emits progress events.

`ImageGallery` (`images/gallery.py`): SQLite at `data/images.db` + files under
`data/images/<yyyy-mm>/<id>.png` with the full generation parameters stored per
image; `list(limit, offset, favorite_only)`, `get(id)`, `delete(id)`,
`set_favorite(id, bool)`, `add(result)`.

Settings (`config.ImageSettings`, under `Settings.images`): `default_model`,
`max_loaded=1`, `idle_ttl_minutes=10`, `output_dir` (default `data/images`),
`default_steps`, `default_size="1024x1024"`, `default_guidance`, `keep_metadata=True`.

## HTTP API (additions)

```
GET    /api/images/models                       → [LocalModel DTO + {imageFamily, defaults:{steps,guidance,sizes}}]
POST   /api/images/generate                     → SSE
       body {modelKey, prompt, negativePrompt?, width, height, steps?, guidance?, seed?, count? (≤8)}
       frames {"type":"status","message"} | {"type":"progress","index","step","steps"}
              | {"type":"image","image":ImageDTO} | {"type":"done"} | {"type":"error","message"}
GET    /api/images?limit&offset&favorite        → [ImageDTO]
GET    /api/images/{id}/file                    → image/png bytes
PATCH  /api/images/{id} {favorite}              → ImageDTO
DELETE /api/images/{id}
GET    /api/images/stats                        → {count, favorites, sizeBytes, oldestAt, newestAt,
                                                   models:[{modelKey, count}]}
POST   /v1/images/generations                   → OpenAI shape {created, data:[{b64_json|url, revised_prompt}]}
       body {prompt, model?, n? (≤4), size?, seed?, response_format?,
             negative_prompt?, steps?, guidance?}
```

`ImageDTO = {id, url, path, prompt, originalPrompt, negativePrompt, modelKey,
modelName, width, height, seed, steps, guidance, favorite, sizeBytes, createdAt,
elapsedMs}`. `originalPrompt` is what the user typed when the prompt was
translated to English first (`localcode/images/prompt.py`), `""` otherwise —
diffusion models understand English far better, so a non-English prompt is
translated by a loaded language model of at least ~2B parameters before it
reaches the image model. `POST /api/images/generate` accepts
`translatePrompt: true|false` to force or disable that; the terminal has
`--translate` / `--no-translate`.

## Terminal

```
localcode image "kot w kapeluszu" [-m MODEL] [--size 1024x1024] [--steps 4] [--seed 42] [-n 2] [-o plik.png] [--open]
localcode images [list|rm ID|open ID]
/obraz <prompt>        # inside the TUI (agent and chat mode): generates and shows the path (iTerm2/kitty inline preview when available)
```

## Mac app — section "Obrazy"

Prompt box + model picker (only image models, with a memory-fit badge), size
presets (kwadrat/pion/poziom + custom), steps/guidance/seed (seed with a "losowy"
toggle and a "użyj ponownie" button), a generate button with live progress, and a
gallery grid (hover: prompt tooltip; click: detail with all parameters, "Użyj
ponownie", "Zapisz jako…", "Pokaż w Finderze", "Ulubione", "Usuń"). Drag-and-drop
of a generated image out of the app must work. Discover gains a "Obrazy" filter
(`pipeline_tag=text-to-image`) so image models are downloaded the same way.
