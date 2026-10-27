# LocalCode — lokalne API HTTP

Referencja dla programistów: jak połączyć się z serwerem LocalCode, jak
uwierzytelnić żądania i jakie endpointy są dostępne. Opis dla użytkownika
aplikacji jest w [`UZYCIE.md`](UZYCIE.md), kontrakt architektury w
[`ARCHITECTURE.md`](ARCHITECTURE.md).

Serwer to jeden proces Pythona (aiohttp) — ten sam, z którego korzystają
aplikacja na Maca i terminal. Wystawia trzy rzeczy:

| co | ścieżki | dla kogo |
|---|---|---|
| **API zgodne z OpenAI** | `/v1/*` | zewnętrzne narzędzia i biblioteki (`openai`, Continue, Open WebUI…) |
| **API aplikacji** (REST + SSE) | `/api/*` | aplikacja na Maca, terminal, własne skrypty |
| **Agent przez WebSocket** | `/api/agent/sessions/{id}/ws` | klienci agenta kodującego |

Spis treści: [Połączenie](#połączenie) · [Uwierzytelnianie](#uwierzytelnianie) ·
[Origin, CORS i Host](#origin-cors-i-host) · [Błędy](#błędy) ·
[API zgodne z OpenAI](#api-zgodne-z-openai) · [Endpointy `/api/*`](#endpointy-api) ·
[Strumienie SSE](#strumienie-sse) · [Agent: WebSocket](#agent-websocket) ·
[Przykłady](#przykłady) · [LocalCode jako backend innych narzędzi](#localcode-jako-backend-innych-narzędzi)

---

## Połączenie

| ustawienie | wartość domyślna | jak zmienić |
|---|---|---|
| adres nasłuchu | `127.0.0.1` | `localcode serve --host ADRES` albo `server.host` w `~/.localcode/settings.json` |
| port | `8421` | `localcode serve --port N` albo `server.port` w `settings.json`; `--port 0` = wolny port wybrany przez system |
| token dostępu | losowy (`secrets.token_urlsafe(32)`) | `localcode serve --token T` albo zmienna `LOCALCODE_TOKEN` |

Serwer uruchamiasz poleceniem `localcode serve` (opcje: `--host`, `--port`,
`--token`, `--parent-pid PID`, `--log-level`, `--verbose`). Po starcie:

* na stdout trafia jedna linia `LOCALCODE_READY port=<p>` (nic więcej),
* powstaje plik `server.json` (prawa `600`) z kluczami
  `port`, `token`, `pid`, `version`, `startedAt`, `host`; znika przy zamknięciu
  serwera,
* logi idą do `~/.localcode/logs/server.log`.

> **Aplikacja na Maca** startuje własny serwer na **losowym porcie**
> (`--port 0`) z losowym tokenem — jedynym źródłem prawdy jest wtedy
> `server.json`. Jeśli chcesz stały adres, uruchom `localcode serve` przed
> aplikacją: aplikacja i terminal dołączają do działającego serwera zamiast
> startować drugi.

Katalog danych to `~/.localcode` albo `$LOCALCODE_HOME`, jeśli ustawiłeś tę
zmienną. Port i token do skryptów:

```bash
LC_INFO="${LOCALCODE_HOME:-$HOME/.localcode}/server.json"
PORT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['port'])" "$LC_INFO")
TOKEN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['token'])" "$LC_INFO")
BASE="http://127.0.0.1:$PORT"

curl -s "$BASE/api/health"
# {"ok": true, "version": "0.1.0"}
```

Przykłady poniżej zakładają zmienne `BASE` i `TOKEN` z tego fragmentu.

> Zmiany w `server.*` zapisane przez `PATCH /api/app-settings`
> (`openaiApiEnabled`, `openaiApiRequiresToken`, `corsOrigins`) działają od
> razu. Zmiana `host` albo `port` obowiązuje po ponownym uruchomieniu serwera.

---

## Uwierzytelnianie

Token to zwykły **bearer token**: nagłówek `Authorization: Bearer $TOKEN`.
Nigdy nie wklejaj go do repozytorium ani do logów — plik `server.json` jest
dostępny tylko dla Twojego użytkownika.

| ścieżka | token |
|---|---|
| `GET /api/health` | **niepotrzebny** (służy do sprawdzenia, czy port odpowiada) |
| pozostałe `/api/*` | **zawsze wymagany** |
| `/api/events` i ścieżki kończące się na `/ws` (WebSocket agenta) | nagłówek **albo** `?token=$TOKEN` w adresie — `EventSource` i `WebSocket` w przeglądarce nie ustawiają nagłówków |
| `/v1/*` | **domyślnie niepotrzebny**; wymagany po włączeniu `server.openai_api_requires_token` |

Brak lub zły token: `401` z ciałem
`{"error": "Brak autoryzacji — nieprawidłowy token dostępu.", "code": "unauthorized"}`
(pod `/v1` w kształcie OpenAI, patrz [Błędy](#błędy)).

Wymóg tokenu na `/v1/*` włączysz w `~/.localcode/settings.json`:

```json
{ "server": { "openai_api_requires_token": true } }
```

albo przez API: `PATCH /api/app-settings` z `{"server": {"openaiApiRequiresToken": true}}`.
Wtedy klient OpenAI używa tokenu z `server.json` jako klucza API.

> **Uwaga przy `--host 0.0.0.0`.** Domyślnie `/v1/*` działa bez tokenu, bo
> serwer słucha tylko na `127.0.0.1`. Jeśli wystawiasz go w sieci, włącz
> `openai_api_requires_token`.

---

## Origin, CORS i Host

Serwer chroni się przed stronami w przeglądarce:

* **Origin.** Żądanie z nagłówkiem `Origin` (czyli z przeglądarki) dostaje `403`
  (`origin_not_allowed`), dopóki nie dopiszesz tego adresu do
  `server.cors_origins` (np. `["http://localhost:3000"]`; wartość `"*"` pozwala
  na wszystkie). Dla dozwolonego origin serwer odpowiada nagłówkami CORS
  (`Access-Control-Allow-Origin`, `-Headers`, `-Methods`, `-Max-Age: 600`)
  i obsługuje preflight `OPTIONS` (`204`). Klienty natywne — `curl`, Python,
  Swift, terminal — nie wysyłają `Origin` i nie są dotknięte.
* **Host.** Żądanie zaadresowane do nazwy DNS (innej niż `localhost`,
  `*.localhost` i skonfigurowany `server.host`) dostaje `403`
  (`host_not_allowed`) — to ochrona przed „DNS rebinding”. Adresy IP są
  zawsze dozwolone.
* **Rozmiar.** `Content-Length` powyżej 201 MB → `413` (`payload_too_large`).
  Wgrywane pliki: najwyżej 10 na żądanie, każdy do 20 MB.

---

## Błędy

Pod `/api/*`:

```json
{"error": "Nie znaleziono rozmowy.", "code": "not_found"}
```

Komunikat `error` jest po polsku, `code` to stabilny identyfikator. Błędy
walidacji mają dodatkowo `details: [{"path", "message"}]`
(`code: "validation_error"`).

Pod `/v1/*` ten sam błąd ma kształt OpenAI, dzięki czemu biblioteki klienckie
wyświetlają go poprawnie:

```json
{"error": {"message": "Nie znaleziono modelu „qwen3-4b”. Sprawdź GET /v1/models.",
           "type": "invalid_request_error", "code": "model_not_found", "param": null}}
```

`type` to `invalid_request_error` dla `4xx` i `server_error` dla `5xx`.

Najczęstsze kody:

| status | `code` | kiedy |
|---|---|---|
| 400 | `invalid_json` | ciało nie jest poprawnym JSON-em albo nie jest obiektem |
| 400 | `validation_error` | złe pole (lista w `details`) |
| 400 | `invalid_request` | zły `POST /api/chat` (lista w `details`) |
| 401 | `unauthorized` | brak lub zły token |
| 403 | `origin_not_allowed`, `host_not_allowed`, `forbidden` | patrz wyżej |
| 404 | `not_found` | brak zasobu (rozmowa, projekt, obraz, sesja…) |
| 404 | `model_not_found` | nieznany model (`/v1`, sesje agenta) |
| 404 | `openai_api_disabled` | `/v1/*` wyłączone w ustawieniach |
| 405 / 413 / 415 | `method_not_allowed`, `payload_too_large`, `unsupported_media_type` | |
| 409 | `engine_error` | silnik modelu nie mógł wykonać zadania (pod `/v1`: `503`) |
| 500 | `internal_error` | „Wewnętrzny błąd serwera.” — szczegóły w `server.log` |

Błąd w trakcie strumienia SSE (odpowiedź jest już rozpoczęta) przychodzi jako
ramka `{"type": "error", "message", "code"}`, nie jako status HTTP.

---

## API zgodne z OpenAI

Prefiks `/v1`, jak w lokalnym serwerze LM Studio. Wyłączysz go w
**Runtime → Serwer zgodny z OpenAI** (przełącznik „Włączony”) albo przez
`server.openai_api_enabled: false` — wtedy każde żądanie `/v1/*` kończy się
`404 openai_api_disabled`.

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/v1/models` | lista modeli z biblioteki (lokalnych) |
| `GET` | `/v1/models/{id}` | jeden model; `404 model_not_found` |
| `POST` | `/v1/chat/completions` | czat: strumieniowanie, narzędzia, rozumowanie |
| `POST` | `/v1/completions` | surowe dopełnianie tekstu, bez szablonu czatu |
| `POST` | `/v1/embeddings` | wektory z modelu embeddingowego |
| `POST` | `/v1/images/generations` | lokalne generowanie obrazów |

### Nazwa modelu (`model`)

* Dokładny klucz z `GET /v1/models` (`id`), np.
  `unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M`,
  **albo dowolny jednoznaczny fragment klucza**, np. `qwen3-4b`.
* Model ładuje się przy pierwszym żądaniu (może to potrwać); korzysta z tej
  samej instancji co aplikacja i terminal.
* **Modele zdalne nie są dostępne pod `/v1`.** Klucz `remote:<dostawca>:<model>`
  (np. `remote:openai:gpt-4o`; dostawcy: `openai`, `anthropic`, `gemini`,
  `deepinfra`, `unlid`) działa w sesjach agenta (`modelKey`, `setModel`)
  i w `POST /api/chat` (`model`), a lista `GET /api/models` i
  `GET /api/library` zawiera je z flagą zdalności — natomiast `/v1/*` odpowie
  `404 model_not_found`.
* Brak pola: `400 missing_model`.

Element `data[]` z `GET /v1/models`:

```json
{"id": "…/Qwen3-4B-Instruct-2507-Q4_K_M", "object": "model", "created": 1760000000,
 "owned_by": "localcode", "type": "llm", "format": "gguf", "quantization": "Q4_K_M",
 "context_length": 262144, "capabilities": ["tools"], "state": "not-loaded"}
```

`type` to `llm` albo `embeddings`, `state` to `loaded` / `not-loaded`.
(Kształt z kodu; w środowisku testowym bez modeli lista jest pusta:
`{"object": "list", "data": []}`.)

### `POST /v1/chat/completions`

Pola żądania (pozostałe są ignorowane):

| pole | opis |
|---|---|
| `model` | patrz wyżej (wymagane) |
| `messages` | niepusta lista `{role, content, tool_calls?, tool_call_id?, name?}`; `content` to tekst albo lista części — **używane są tylko części tekstowe, obrazy w `messages` są pomijane** |
| `stream` | `true` → SSE w formacie OpenAI, zakończone `data: [DONE]` |
| `stream_options.include_usage` | `true` → dodatkowa ramka z `usage` |
| `max_tokens` / `max_completion_tokens` | dodatnia liczba całkowita; **domyślnie 1024** |
| `temperature` | 0–2 (domyślnie 0,3) |
| `top_p` | 0–1 (0,95) · `top_k` 0–1000 (40) · `min_p` 0–1 (0,05) · `repetition_penalty` 0,5–2 |
| `stop` | tekst albo lista tekstów (najwyżej 8) |
| `seed` | liczba całkowita |
| `n` | tylko `1` (inaczej `400`) |
| `tools`, `tool_choice` | narzędzia w formacie OpenAI; `tool_choice: "none"` wyłącza je; stare `functions` też działają |
| `reasoning_effort` | `none`/`minimal`/`off` wyłączają myślenie; `low`/`medium`/`high`/`on` włączają je na wskazanym poziomie (jeśli model to obsługuje) |
| `chat_template_kwargs.enable_thinking` | bezpośredni przełącznik szablonu; ma pierwszeństwo przed `reasoning_effort` |

Odpowiedź (bez `stream`) ma kształt `chat.completion`: `choices[0].message`
z `content`, opcjonalnie `reasoning_content` (tok myślenia modelu) i
`tool_calls` (`function.arguments` jako **tekst JSON**, jak w prawdziwym API;
wtedy `finish_reason` = `tool_calls`), `finish_reason` = `stop` / `length`
oraz `usage` z licznikami z silnika. Ceny nie istnieją — model jest lokalny.

W strumieniu: pierwsza ramka niesie `delta.role`, potem `delta.content` i
`delta.reasoning_content`, wywołania narzędzi przychodzą **jedną ramką
`delta.tool_calls` na końcu**, potem ramka z `finish_reason`.

Domyślne parametry próbkowania są dobrane do małych modeli lokalnych (inne niż
w api.openai.com). Limitu „bez limitu” nie ma w tym API — ustaw `max_tokens`.

### Pozostałe endpointy `/v1`

* **`POST /v1/completions`** — `{model, prompt, stream?, max_tokens?, …}`;
  `prompt` to tekst (albo lista z jednym tekstem). Odpowiedź `text_completion`.
* **`POST /v1/embeddings`** — `{model, input}`; `input` to tekst albo lista
  tekstów; wymaga modelu embeddingowego. Odpowiedź: `data[].embedding`, `usage`.
* **`POST /v1/images/generations`** — `{prompt, model?, n?, size?, seed?,
  response_format?, negative_prompt?, steps?, guidance?}`. `n` ≤ 4, `size` w
  formacie `SZERxWYS` (`auto` = domyślny rozmiar z ustawień),
  `response_format`: `b64_json` (**domyślnie**, inaczej niż w api.openai.com)
  albo `url`. Adres z `url` prowadzi do `/api/images/{id}/file`, który wymaga
  tokenu — `openai` SDK nie doda go przy pobieraniu, więc używaj `b64_json`.
  Bez zainstalowanego modelu obrazów: `404 no_image_models`. Szczegóły modeli:
  [`IMAGES.md`](IMAGES.md).

Bez żadnego modelu w bibliotece dostaniesz (sprawdzone na czystym serwerze):

```text
$ curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-4b","messages":[{"role":"user","content":"Cześć!"}]}'
{"error": {"message": "Nie znaleziono modelu „qwen3-4b”. Sprawdź GET /v1/models.", "type": "invalid_request_error", "code": "model_not_found", "param": null}}
```

---

## Endpointy `/api/*`

Wszystkie wymagają tokenu (poza `/api/health`). Ciała i odpowiedzi to JSON
w camelCase; `PATCH` zmienia tylko podane pola. Nie wszystkie obiekty są tu
rozpisane pole po polu — pełne DTO opisuje [`ARCHITECTURE.md`](ARCHITECTURE.md).

### Zdrowie, zdarzenia, runtime (`routes_runtime.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/health` | `{"ok": true, "version": "…"}` — jedyna trasa bez tokenu |
| `GET` | `/api/events` | strumień SSE zdarzeń całej aplikacji ([niżej](#strumienie-sse)) |
| `GET` | `/api/runtime` | stan: `instances[]`, `system`, `budget`, `settings` |
| `PATCH` | `/api/runtime/settings` | ustawienia środowiska (`maxLoadedModels`, `idleTtlMinutes`, `ramLimitGb`, `kvCacheType`…); zwraca stan |
| `POST` | `/api/runtime/load` | `{modelKey, config?, newInstance?}` → instancja; ładuje w tle |
| `POST` | `/api/runtime/unload` | `{instanceId}` albo `{"all": true}` → stan |
| `POST` | `/api/runtime/estimate` | `{modelKey, config?}` → szacunek pamięci |
| `POST` | `/api/runtime/tune` | `{modelKey?, useModel?, apply?}` → `{recommendation, status?}` |

### Czat (`routes_chat.py`, tabela `ROUTES` w `chat/api.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/bootstrap` | ustawienia czatu, projekty, rozmowy, modele — jednym strzałem |
| `GET` | `/api/conversations` | lista; query: `folderId` (puste/`null`/`none` = bez projektu), `archived` (`1` = tylko zarchiwizowane, `all` = wszystkie; domyślnie aktywne), `limit` |
| `POST` | `/api/conversations` | utwórz (`201`); ciało: `title`, `folderId`, `model`, `systemPrompt`, `temperature`, `maxTokens`, `pinned`, `archived`, `reasoningEffort` |
| `GET` | `/api/conversations/{id}` | `{conversation, messages}` |
| `PATCH` | `/api/conversations/{id}` | te same pola; zmiana `title` blokuje automatyczne tytuły |
| `DELETE` | `/api/conversations/{id}` | `{"ok": true}` |
| `GET` | `/api/conversations/{id}/export` | `?format=md` (domyślnie) albo `json`; plik do pobrania |
| `POST` | `/api/conversations/{id}/title` | wygeneruj tytuł (`?auto=1`) |
| `POST` | `/api/conversations/{id}/compact` | skróć starsze tury teraz → `{summary, before, after, changed, strategy, messages}` |
| `DELETE` | `/api/messages/{id}` | usuń wiadomość |
| `GET` `POST` | `/api/folders` | projekty (`201`); ciało: `name`, `description`, `color`, `icon`, `systemPrompt`, `extraInfo`, `memoryMode` (`shared`/`isolated`), `sortOrder` |
| `PATCH` `DELETE` | `/api/folders/{id}` | edycja; usunięcie z `?deleteChats=1` kasuje też rozmowy |
| `GET` `POST` | `/api/memories` | pamięć; query: `scope` (`global`/`folder`), `folderId`; ciało: `content` (wymagane), `scope`, `folderId`, `kind` (`fact`/`preference`/`note`), `source`, `pinned` |
| `PATCH` `DELETE` | `/api/memories/{id}` | |
| `GET` `POST` | `/api/knowledge` | baza wiedzy; `POST` to `multipart/form-data`: pola `file` (do 10), `scope`, `folderId` |
| `DELETE` | `/api/knowledge/{id}` | |
| `POST` | `/api/upload` | załączniki do wiadomości (`multipart`: `file`, opcjonalnie `conversationId`) → lista z `id` do `attachmentIds` |
| `GET` | `/api/attachments/{id}` | plik załącznika |
| `GET` `PATCH` | `/api/settings` | ustawienia czatu: `globalSystemPrompt`, `memoryEnabled`, `theme`, `profile.*`, `defaults.*` (`model`, `temperature`, `maxTokens` — **`0` = bez limitu**, `ragTopK`, `autoTitle`, `showReasoning`, `historyLimit`) |
| `GET` | `/api/providers` | jedyny „dostawca” czatu to `local` |
| `GET` | `/api/models` | modele czatu: lokalne i zdalne, ulubione pierwsze; `?refresh=1` przeskanuje dysk |
| `GET` | `/api/search?q=` | szukanie w wiadomościach (do 50 trafień) |
| `POST` | `/api/chat` | **SSE**: jedna tura rozmowy ([ramki niżej](#strumienie-sse)) |

Ciało `POST /api/chat`: `content`, `conversationId` (pominięte = nowa rozmowa),
`folderId`, `model`, `attachmentIds` (do 50), `temperature` (0–2),
`maxTokens` (0–1 000 000, **0 = bez limitu**), `reasoningEffort`
(`off`/`low`/`medium`/`high`; `on` = `medium`), `editMessageId`, `regenerate`.
Kolejność wyboru modelu: `model` > model rozmowy > domyślny z ustawień >
pierwszy model czatu. Walidacja kończy się zwykłym błędem JSON, zanim
rozpocznie się strumień.

### Modele, katalog, pobieranie, ustawienia (`routes_models.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/catalog` | sklep: query `category`, `q`, `installed`, `sort` (`polecane`/`popularne`/`lubiane`/`zaufanie`/`oficjalne`/`najmniejsze`), `limit` (domyślnie 200), `minTrust` (0–100), `official`. Nagłówki: `X-Catalog-Total`, `X-Catalog-Offline: 1` |
| `GET` | `/api/catalog/{id}` | pozycja z listą wariantów |
| `GET` | `/api/catalog/lookup?ref=` | pozycja dla wklejonego repozytorium lub adresu huggingface.co |
| `POST` | `/api/catalog/install` | `{id, variantId?}` → zadanie pobierania (`201`) |
| `GET` | `/api/hub/search` | Hugging Face: `q`, `format`, `sort` (domyślnie `trending`), `limit` (30, max 100) |
| `GET` | `/api/hub/curated` | polecane |
| `GET` | `/api/hub/model?repo=org/nazwa` | szczegóły repozytorium (`refresh=1` omija pamięć podręczną) |
| `GET` `POST` | `/api/downloads` | lista zadań / start `{repo, variantId?, files?}` (`201`) |
| `DELETE` | `/api/downloads/{id}` | anuluj; zwraca zadanie |
| `GET` | `/api/library` | modele na dysku (+ zdalne) z `loaded`, `effortLevels`, `loadDefaults` |
| `POST` | `/api/library/rescan` | przeskanuj katalog modeli |
| `POST` | `/api/library/import` | `{path, name?, move?}` — dodaj własny `.gguf`/katalog MLX (`201`) |
| `PATCH` | `/api/library/{klucz}` | `{loadDefaults}` — domyślna konfiguracja ładowania |
| `DELETE` | `/api/library/{klucz}` | usuń z dysku (`204`) |
| `GET` | `/api/logos/{slug}` | logo wydawcy (PNG) |
| `GET` `PATCH` | `/api/app-settings` | ustawienia aplikacji (sekcje `agent`, `chat`, `store`, `security`, `helpbot`, `cerber`, `server`; patrz niżej) |
| `PUT` | `/api/app-settings/hf-token` | `{token}` albo `{"token": null}`; token nigdy nie wraca w odpowiedzi (`hfTokenSet`) |
| `PUT` | `/api/app-settings/cerber-token` | j.w. dla Cerbera |
| `PUT` | `/api/app-settings/favorite-models` | `{key, favorite: true\|false}` — gwiazdka „Ulubione”; działa też dla kluczy `remote:…` |
| `POST` | `/api/cerber/check` | `{ok, installed, message}` |

Klucz modelu w `/api/library/{klucz}` zawiera ukośniki
(`org/repo/plik`) — wstawiasz go bez kodowania `/`.

Przydatne pola `PATCH /api/app-settings` (camelCase, tylko podane są zmieniane):

| sekcja | klucze |
|---|---|
| `agent` | `defaultModel`, `permissionMode` (`default`/`accept_edits`/`auto`/`yolo`/`plan`), `maxSteps` (**`0` = bez limitu**), `ultracodeParallel`, `thinking`, `temperature`, `maxOutputTokens`, `reasoningEffort`, `subagents`, `subagentModel`, `subagentMaxSteps`, `autoReviewerModel` (recenzent trybu Auto; `null` = model agenta), `manageModels`, … |
| `chat` | `defaultModel`, `embeddingModel`, `reasoningEffort` |
| `server` | `openaiApiEnabled`, `openaiApiRequiresToken`, `corsOrigins`, `port` |
| góra | `modelsDir`, `language`, `scanLmstudio`, `scanHfCache`, `favoriteModels` |

Błędne ustawienie → `400 validation_error`; puste ciało → `400 empty_patch`.
Tokenów (`hfToken`, `cerber.token`) nie zmienisz tym endpointem.

### Modele zdalne (`routes_remote.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/remote/providers` | dostawcy: `id`, `label`, `hasKey`, `enabled`, `baseUrl`, `keyUrl`, `modelCount` (`-1` = nie odpowiada), `serviceTier` (`standard`/`flex`/`priority`), `serviceTiers` (dostępne tryby; `[]` = dostawca ich nie ma). **Klucz nigdy nie jest zwracany** |
| `GET` | `/api/remote/models` | modele w kształcie wierszy biblioteki; query `provider`, `refresh=1` |
| `PUT` | `/api/remote/{dostawca}/key` | `{key, baseUrl?}` — klucz jest **sprawdzany u dostawcy przed zapisem**; `{"key": null}` kasuje |
| `PATCH` | `/api/remote/{dostawca}` | `{enabled?: bool, serviceTier?: "standard"\|"flex"\|"priority"}` (co najmniej jedno) — wyłącz dostawcę bez kasowania klucza; ustaw tryb zapytań. Tryb mają tylko `openai`, `deepinfra`, `unlid` (inaczej `400 tier_unsupported`); zła wartość → `400 invalid_service_tier` |

`{dostawca}` to `openai`, `anthropic`, `gemini`, `deepinfra` albo `unlid`
(inna wartość: `404 unknown_provider`). Zdalne zapytania są rozliczane przez
dostawcę, nie przez LocalCode (`costUsd` w LocalCode zawsze wynosi 0).

### Agent (`agent_ws.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `POST` | `/api/agent/sessions` | `{cwd?, modelKey?, mode?}` → sesja (`201`) |
| `GET` | `/api/agent/sessions` | lista; `?cwd=` zawęża do projektu |
| `GET` | `/api/agent/sessions/{id}` | `{session, items, todos}` — stan i zapis rozmowy |
| `DELETE` | `/api/agent/sessions/{id}` | `204` |
| `GET` | `/api/agent/sessions/{id}/ws` | WebSocket, [niżej](#agent-websocket) |
| `GET` | `/api/agent/commands` | polecenia ukośnikowe dostępne w aplikacji + `initPrompt` |

### Pomocnik (`routes_helpbot.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/helpbot` | stan: `state` (`missing`/`downloading`/`ready`), model, `loaded`, `download` |
| `POST` | `/api/helpbot/install` | rozpocznij pobieranie modelu Pomocnika (**pobiera ok. 2,3 GB**) — `202`/`200` |
| `POST` | `/api/helpbot/session` | sesja agenta Pomocnika: ostatnia albo `{"new": true}` (`201`); dalej jak zwykła sesja agenta (WebSocket) |

Pomocnik pobiera się sam przy pierwszym starcie serwera; zmienna środowiskowa
`LOCALCODE_NO_HELPBOT_DOWNLOAD=1` to wyłącza (testy, CI, instalacje offline).
Szczegóły: [`HELPBOT.md`](HELPBOT.md).

### Obrazy (`routes_images.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/images/models` | modele obrazów + domyślne `steps`/`guidance`/rozmiary |
| `POST` | `/api/images/generate` | **SSE**; ciało: `prompt` (wymagane), `modelKey`, `width`, `height`, `steps`, `guidance`, `seed`, `count` (1–8), `negativePrompt`, `translatePrompt` |
| `GET` | `/api/images` | galeria: `limit` (50, max 200), `offset`, `favorite=1` |
| `GET` | `/api/images/stats` | `{count, favorites, sizeBytes, oldestAt, newestAt, models}` |
| `GET` | `/api/images/{id}/file` | bajty PNG |
| `PATCH` | `/api/images/{id}` | `{favorite: bool}` |
| `DELETE` | `/api/images/{id}` | `204`; plik też znika |

Bez zainstalowanego modelu: `404 no_image_models`. Bez dodatku
`localcode[images]` endpointy generowania odpowiadają `503` z wyjaśnieniem.
Opis funkcji: [`IMAGES.md`](IMAGES.md).

### Wideo (`routes_video.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/video/models` | modele wideo na dysku (z flagą `supported`) |
| `POST` | `/api/video/estimate` | `{width?, height?, numFrames?, fps?, steps?}` → `{seconds, message, width, height, numFrames, fps, steps}` |
| `POST` | `/api/video/generate` | **SSE**; ciało: `prompt` (wymagane), `modelKey`, `negativePrompt`, `seed`, `guidance`, `force`, `output` (nazwa pliku w katalogu wideo, bez ścieżki) + pola z `estimate` |

Czas liczy się w minutach na sekundę materiału — zobacz [`VIDEO.md`](VIDEO.md).

### Mowa (`routes_speech.py`)

| metoda | ścieżka | opis |
|---|---|---|
| `GET` | `/api/speech/voices` | `?modelKey=` → `[{id, label, language}]` |
| `POST` | `/api/speech/transcribe` | `{path, modelKey?, language?, segments?, srt?}` — **`path` to ścieżka pliku na tym komputerze**, nie upload |
| `POST` | `/api/speech/speak` | `{text, modelKey?, voice?, speed? (0,5–2,0), output?}` → plik `.wav` (`201`) |

Bez modelu mowy: `400 no_model`. Zobacz [`SPEECH.md`](SPEECH.md).

---

## Strumienie SSE

Odpowiedzi `text/event-stream`: linie `data: {json}` rozdzielone pustą linią;
co 15 s bez ruchu serwer wysyła komentarz `: ping` (klient ma go pomijać).
**Zamknięcie połączenia przerywa zadanie** — tak działa przycisk „Stop” (w czacie
częściowa odpowiedź zostaje zapisana jako `stopped`).

| endpoint | typy ramek (`type`) |
|---|---|
| `POST /api/chat` | `start` (`conversationId`, `userMessage`, `assistantMessageId`) → `status` → `citations` → `compaction` → `context` → `reasoning` / `text` (pole `delta`) → `usage` (`tokensIn`, `tokensOut`, `costUsd`) → `done` (`message`) → `title`; porażka po starcie: `error` (`message`, `code`) |
| `POST /api/images/generate` | `status`, `progress` (`index`, `step`, `steps`), `image` (`image`: obiekt z `id`, `url`, `width`, `height`, `seed`…), `done`, `error` |
| `POST /api/video/generate` | `progress`, `video` (`video`: wynik z ścieżką pliku), `done`, `error` |
| `GET /api/events` | pierwsza ramka `{"type":"runtime", …stan}`, potem `runtime`, `download` (`job`), `library` |

---

## Agent: WebSocket

Agent kodujący to ta sama pętla co w terminalu i w zakładce **Kod**.
Przebieg:

1. **Utwórz sesję** — `POST /api/agent/sessions` z `{cwd, modelKey?, mode?}`.
   `cwd` musi istnieć (`400 bad_cwd`; pominięty = katalog procesu serwera),
   `mode` to `default` / `accept_edits` / `auto` / `yolo` / `plan`
   (`400 invalid_mode`), `modelKey` może być także kluczem zdalnym
   `remote:<dostawca>:<model>` (dostawca musi mieć zapisany klucz; inaczej
   `404 model_not_found`). Odpowiedź (`201`): `id`, `cwd`, `modelKey`, `mode`,
   `title`, `createdAt`, `updatedAt`, `contextUsed`, `contextLimit`, `busy`,
   `effort`, `effortLevels`, `goal`.
2. **Połącz się** — `ws://127.0.0.1:$PORT/api/agent/sessions/{id}/ws`, z
   nagłówkiem `Authorization` albo `?token=$TOKEN`. Brak tokenu: `401` przy
   handshake. Do jednej sesji może być podłączonych wielu klientów naraz
   (np. terminal i aplikacja) — wszyscy dostają wszystkie zdarzenia.
3. Po połączeniu serwer wysyła `{"type":"session","session":{…}}`,
   `{"type":"history","items":[…]}`, ewentualnie `todos` oraz jeszcze
   nierozstrzygnięte `permission_request`.
4. **Wyślij polecenie** i czytaj zdarzenia, aż przyjdzie `turn_end`.

### Klient → serwer

| `type` | pola | działanie |
|---|---|---|
| `user` | `text`, `images?` | nowa wiadomość; `@ścieżka` w tekście serwer rozwija do zawartości pliku (względem `cwd` sesji, jak w terminalu). `images`: do 6 pozycji `{name, mime, data}` (`data` = base64; `mime` to `image/png`, `image/jpeg`, `image/gif` albo `image/webp`; ok. 20 MB na obraz) — dla modeli z wizją |
| `permission` | `requestId`, `decision` | odpowiedź na prośbę o zgodę: `allow`, `deny` albo `always` |
| `interrupt` | — | przerwij bieżącą turę (odrzuca też oczekujące prośby o zgodę) |
| `setMode` | `mode` | zmień tryb uprawnień |
| `setModel` | `modelKey` | zmień model (lokalny lub `remote:…`) |
| `setEffort` | `effort` | `off` / `low` / `medium` / `high` / `ultracode` (tylko agent: wysokie myślenie + równolegli pomocnicy; zob. [UZYCIE.md](UZYCIE.md#ultracode)) |
| `compact` | — | skróć kontekst teraz |
| `clear` | — | wyczyść rozmowę (serwer odpowie `history` z pustą listą) |
| `setGoal` | `text`, `message?` | `/goal`: ustaw cel i zacznij turę |
| `clearGoal` | — | usuń cel |

Wiadomość `user` wysłana w trakcie tury trafia do kolejki. Serwer odpowiada
`{"type":"ack","of":"user","status":"started"}` albo `"status":"queued"`.

### Serwer → klient

Zdarzenia agenta (pola w camelCase):

| `type` | pola |
|---|---|
| `turn_start` | `turnId`, `userText` |
| `text_delta` | `text` — kawałek odpowiedzi |
| `reasoning_delta` | `text` — kawałek toku myślenia |
| `tool_call` | `callId`, `name`, `arguments`, `title` |
| `tool_result` | `callId`, `name`, `ok`, `summary`, `output`, `diff?` |
| `permission_request` | `requestId`, `tool`, `title`, `preview` (polecenie albo diff), `options` (`["allow","deny","always"]`) |
| `auto_review` | `callId`, `tool`, `verdict`, `reason`, `reviewerModel` — tylko w trybie `auto` |
| `context` | `used`, `limit`, `system`, `tools`, `history`, `reserve` |
| `compaction` | `before`, `after`, `strategy` |
| `status` | `message`, `phase` (`loading`/`thinking`/`tool`/`info`/`queued`) |
| `todos` | `items` — `[{content, status}]` |
| `goal` | `text`, `state`, `turns`, `maxTurns`, `summary` |
| `subagent_start` / `subagent_step` / `subagent_end` | praca pomocnika zleconego przez agenta; każde niesie `agentId` i `parentCallId` (karta `delegate`, pod którą się zagnieżdża) |
| `turn_end` | `turnId`, `reason` (`done` / `interrupted` / `max_steps` / `error`), `stats` |
| `error` | `message`, `code?` |

Oraz ramki protokołu: `session` (przy połączeniu i po każdej zmianie sesji),
`history`, `ack` (`of`: `user` / `permission` / `setGoal`, `status`) i
`permission_resolved` (`requestId`, `decision`).

### Prośby o zgodę i tryby

| `mode` | zachowanie |
|---|---|
| `default` | odczyt bez pytania; zapis plików i polecenia powłoki → `permission_request` |
| `accept_edits` | zapis plików w projekcie bez pytania; polecenia → `permission_request` |
| `auto` | o zgodzie decyduje model-recenzent ([niżej](#tryb-auto-i-auto_review)) |
| `plan` | tylko odczyt; zmiany i polecenia są odrzucane |
| `yolo` | wszystko bez pytania, poza poleceniami katastrofalnymi (`rm -rf ~`, `curl … \| sh`) |

Zapis poza folderem projektu pyta w `default`, `accept_edits` i `yolo`
(w `auto` ocenia go recenzent), a polecenia katastrofalne pytają zawsze
(poza `plan`, gdzie są odrzucane). Gdy agent czeka na zgodę, przychodzi
`permission_request`; odpowiedz tym samym `requestId`:

```json
{"type": "permission", "requestId": "<requestId z prośby>", "decision": "allow"}
```

* `allow` — jednorazowo; `deny` — odmowa (agent dostaje informację);
  `always` — pamięta zgodę do końca sesji (dla poleceń powłoki na poziomie
  prefiksu polecenia, np. `git commit`). Dozwolone wartości podaje pole `options`.
* Wygrywa pierwszy klient, który odpowie; reszta dostaje
  `permission_resolved`. Spóźniona odpowiedź daje
  `{"type":"ack","of":"permission","status":"stale","requestId":"…"}`.
* Zła wartość `decision`: `{"type":"error","code":"invalid_decision",
  "message":"Nieznana decyzja „maybe”. Dozwolone: allow, deny, always."}`.
* `interrupt`, koniec tury albo zamknięcie sesji odrzuca wszystkie oczekujące prośby. Gdy do sesji nie jest podłączony żaden klient WebSocket, prośba jest odrzucana od razu.
* Gdy nikt nie odpowiada (skrypt bez człowieka), użyj `accept_edits`, `auto`
  albo `yolo` — inaczej tura będzie czekać na decyzję.

### Tryb Auto i `auto_review`

W trybie `auto` zamiast Ciebie decyduje model-recenzent (wybierany w
Ustawieniach → Agent albo przez `agent.autoReviewerModel`; domyślnie model
agenta). Dla każdej akcji, która w trybie `default` zapytałaby o zgodę,
przychodzi najpierw zdarzenie `auto_review`:

```json
{"type": "auto_review", "callId": "…", "tool": "bash", "verdict": "allow",
 "reason": "Co robi: … Powód: …", "reviewerModel": "…"}
```

`verdict` to `allow` (akcja się wykona), `deny` (nie — agent dostaje `reason`
i wybiera inną drogę) albo `ask` (recenzent nie dał użytecznej odpowiedzi;
zaraz potem przyjdzie zwykłe `permission_request`, a `reason` mówi dlaczego).
`reason` jest po polsku. Polecenia katastrofalne oraz przekazanie rozmowy
zdalnemu dostawcy nigdy nie trafiają do recenzenta — o nich zawsze pyta
człowiek (zwykłe `permission_request`).

### Błędy protokołu

Błąd dotyczący jednej wiadomości wraca tylko do nadawcy jako
`{"type":"error","message":"…","code":"…"}`; połączenie zostaje otwarte.
Napotkane kody: `invalid_mode`, `invalid_effort`, `invalid_decision`,
`empty_message`, `model_required` („Nie wybrano modelu dla sesji agenta.”),
`model_not_found`, `invalid_images`, `too_many_images`, `image_too_large`,
`empty_goal`. Nieznany `type`: „Nieznany typ wiadomości „…”.”, tekst, który
nie jest JSON-em: „Nieprawidłowy JSON w wiadomości WebSocket.”

---

## Przykłady

Zmienne `BASE` i `TOKEN`: patrz [Połączenie](#połączenie). Przykłady z
modelem zakładają zainstalowany model (`localcode models install …`).

### curl

```bash
# zdrowie (bez tokenu) i lista modeli lokalnych
curl -s "$BASE/api/health"
curl -s "$BASE/v1/models"

# API aplikacji — z tokenem
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/api/library"
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/api/runtime"

# czat OpenAI, bez strumienia
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3-4b",
  "messages": [{"role": "user", "content": "Napisz haiku o kawie."}],
  "max_tokens": 200
}'

# strumieniowanie (-N wyłącza buforowanie curla)
curl -N "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3-4b", "stream": true,
  "messages": [{"role": "user", "content": "Policz do pięciu."}]
}'

# narzędzia
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3-4b",
  "messages": [{"role": "user", "content": "Jaka jest pogoda w Gdańsku?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather", "description": "Pogoda w mieście",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}'

# nowa rozmowa w czacie aplikacji + tura w strumieniu SSE
curl -s -X POST "$BASE/api/conversations" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"title": "Z terminala"}'
curl -N -X POST "$BASE/api/chat" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"content": "Cześć!"}'

# zmiana ustawienia: agent bez limitu kroków, tryb Auto
curl -s -X PATCH "$BASE/api/app-settings" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"agent": {"maxSteps": 0, "permissionMode": "auto"}}'
```

### Python: biblioteka `openai`

```python
import json, os
from pathlib import Path
from openai import OpenAI

home = Path(os.environ.get("LOCALCODE_HOME", "~/.localcode")).expanduser()
info = json.loads((home / "server.json").read_text())

client = OpenAI(base_url=f"http://127.0.0.1:{info['port']}/v1",
                api_key=info["token"])   # ignorowany, dopóki nie włączysz wymogu tokenu

print([m.id for m in client.models.list().data])

stream = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Cześć!"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(delta.content or "", end="", flush=True)
```

Sprawdzone bez modelu: `models.list()` zwraca `[]`, a `chat.completions.create`
rzuca `NotFoundError` z komunikatem „Nie znaleziono modelu „qwen3-4b”…”.

### Python: `requests` / `httpx`

```python
import json, os
from pathlib import Path
import requests

home = Path(os.environ.get("LOCALCODE_HOME", "~/.localcode")).expanduser()
info = json.loads((home / "server.json").read_text())
base = f"http://127.0.0.1:{info['port']}"
auth = {"Authorization": f"Bearer {info['token']}"}

print(requests.get(f"{base}/api/health").json())            # {'ok': True, 'version': '…'}
print(requests.get(f"{base}/api/library", headers=auth).json())

r = requests.post(f"{base}/v1/chat/completions", json={
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "hej"}],
})
print(r.status_code, r.json())
```

Strumień SSE przez `httpx` (te same ramki co w `curl -N`):

```python
import httpx

with httpx.stream("POST", f"{base}/api/chat", headers=auth,
                  json={"content": "Cześć!"}, timeout=None) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            frame = json.loads(line[6:])
            if frame["type"] == "text":
                print(frame["delta"], end="", flush=True)
```

### JavaScript (`fetch`, Node 18+)

```js
import { readFileSync } from "node:fs";
import { homedir } from "node:os";

const home = process.env.LOCALCODE_HOME ?? `${homedir()}/.localcode`;
const { port, token } = JSON.parse(readFileSync(`${home}/server.json`, "utf8"));
const base = `http://127.0.0.1:${port}`;

console.log(await (await fetch(`${base}/api/health`)).json());

const res = await fetch(`${base}/api/conversations`, {
  method: "POST",
  headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
  body: JSON.stringify({ title: "Z JavaScriptu" }),
});
console.log(res.status, (await res.json()).title);   // 201 Z JavaScriptu
```

Z **przeglądarki** dopisz adres swojej strony do `server.cors_origins` — bez
tego serwer odpowie `403 origin_not_allowed`. Token trzymaj po stronie serwera
swojej aplikacji; nie wklejaj go w kod wysyłany do przeglądarek.

### Agent przez WebSocket (Python)

Biblioteka `websockets` (≥ 14; `pip install websockets httpx`):

```python
import asyncio, json, os
from pathlib import Path
import httpx
import websockets

home = Path(os.environ.get("LOCALCODE_HOME", "~/.localcode")).expanduser()
info = json.loads((home / "server.json").read_text())
base = f"http://127.0.0.1:{info['port']}"
auth = {"Authorization": f"Bearer {info['token']}"}

async def main(cwd: str) -> None:
    session = httpx.post(f"{base}/api/agent/sessions", headers=auth,
                         json={"cwd": cwd, "mode": "default"}).json()
    url = f"ws://127.0.0.1:{info['port']}/api/agent/sessions/{session['id']}/ws"
    async with websockets.connect(url, additional_headers=auth) as ws:
        await ws.send(json.dumps({"type": "user", "text": "co jest w tym katalogu?"}))
        async for raw in ws:
            event = json.loads(raw)
            kind = event["type"]
            if kind == "text_delta":
                print(event["text"], end="", flush=True)
            elif kind == "permission_request":
                print(f"\n[zgoda] {event['title']}\n{event['preview']}")
                await ws.send(json.dumps({"type": "permission",
                                          "requestId": event["requestId"],
                                          "decision": "allow"}))
            elif kind == "auto_review":
                print(f"\n[auto: {event['verdict']}] {event['reason']}")
            elif kind == "error":
                print("\n[błąd]", event["message"])
                break
            elif kind == "turn_end":
                print("\n[koniec tury]", event["reason"])
                break

asyncio.run(main(os.getcwd()))
```

Uwaga: ten skrypt zatwierdza **każdą** prośbę — w prawdziwym użyciu pokaż
`preview` człowiekowi albo użyj trybu `auto`. Bez wybranego modelu
(`modelKey` lub `agent.defaultModel`) serwer odpowie
`[błąd] Nie wybrano modelu dla sesji agenta.`

---

## LocalCode jako backend innych narzędzi

Każde narzędzie, które umie rozmawiać z „serwerem zgodnym z OpenAI”
(edytory, wtyczki, interfejsy czatu, własne skrypty), podłączasz tak samo:

| ustawienie w narzędziu | wartość |
|---|---|
| **Base URL** | `http://127.0.0.1:8421/v1` (albo port z `server.json`) |
| **API key** | dowolny tekst, jeśli `openai_api_requires_token` jest wyłączone (domyślnie); w przeciwnym razie token z `server.json` |
| **Model** | klucz albo fragment klucza z `GET /v1/models` (np. `qwen3-4b`) |

Co warto wiedzieć:

* Serwer musi działać — uruchom `localcode serve` (stały port 8421) albo
  aplikację na Maca (port losowy, odczytaj z `server.json`).
* Pierwsze żądanie ładuje model do pamięci, więc może trwać dłużej; obowiązuje
  budżet pamięci z ustawień Runtime.
* Narzędzia obsługiwane są natywnie, jeśli szablon modelu je zna, a w przeciwnym
  razie przez prompt — mały model może jednak źle trafić w format wywołania.
* Obrazów w `messages` serwer nie przyjmuje (części inne niż tekst są pomijane);
  modele zdalne nie są dostępne pod `/v1`.
* Narzędzie działające w przeglądarce musi mieć swój adres w `server.cors_origins`.
* Domyślne `max_tokens` to 1024 — długie odpowiedzi wymagają ustawienia go
  w narzędziu.
