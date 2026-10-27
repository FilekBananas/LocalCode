# Fixtures: sklep z modelami, obrazy, myślenie, kompaktowanie

Captured with `curl` from a real server (`.venv/bin/localcode serve --port 0
--token test`) with a real library on disk: a small GGUF LLM, an MLX LLM, an
embedding model, an SD 1.5 image model and an MLX FLUX.2 image model. Every DTO
in `LocalCodeKit`, `ModelsFeature` and `ImagesFeature` is written against these
exact bytes, not against the documentation.

| plik | endpoint | co sprawdza |
|---|---|---|
| `catalog.json` | `GET /api/catalog` | 59 wpisów, 6 kategorii, `fit`/`fitDetail`/`badges`/`installed` |
| `catalog-entry.json` | `GET /api/catalog/{id}` | pełna lista `variants[]` dla SD 1.5 (7 wariantów) |
| `app-settings.json` | `GET /api/app-settings` | `catalog.categories`, `sourceNote`, `offlineNote`, `chat/agent.reasoningEffort` |
| `models.json` | `GET /api/models` | `effortLevels` — `[]` oraz `["off","on"]` |
| `images-models.json` | `GET /api/images/models` | `imageFamily`, `defaults.{steps,guidance,sizes}`, `estimatedBytes` |
| `images-list.json` | `GET /api/images` | `ImageDTO` z pełnym kompletem parametrów |
| `chat-stream.sse` | `POST /api/chat` | ramki `start/title/text/usage/context/done` |

## Jak je odtworzyć

```bash
P=<port>; H='Authorization: Bearer test'
curl -s -H "$H" "http://127.0.0.1:$P/api/catalog"       -o catalog.json
curl -s -H "$H" "http://127.0.0.1:$P/api/images/models" -o images-models.json
# identyfikator w ścieżce jest repozytorium — trzeba go zakodować:
curl -s -H "$H" "http://127.0.0.1:$P/api/catalog/second-state%2Fstable-diffusion-v1-5-GGUF"
```

## Co potwierdziły

* `X-Catalog-Offline: 1` nie pojawia się przy działającej sieci — banner
  offline jest sterowany nagłówkiem, nie treścią odpowiedzi.
* `effortLevels` to `["off","on"]` dla Qwen3, ale **`PATCH /api/conversations`
  odrzuca `"on"`** (`EFFORT_CHOICES = off|low|medium|high`). Dlatego
  `EffortLevel.wireValue` wysyła `medium` dla „Włączone”, a `EffortLevel.resolve`
  mapuje zapisane `medium` z powrotem na „Włączone”.
* `GET /api/catalog/{id}` potrafi zwrócić inny rozmiar i inny `recommendedVariant`
  niż wiersz listy (patrz `CatalogEntry.merging(detail:)`).
