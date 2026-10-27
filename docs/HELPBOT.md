# Pomocnik — help bot, który ustawia aplikację za użytkownika

Mały model, który przychodzi z aplikacją i robi to, co jej ekrany: znajduje
modele w sklepie i na Hugging Face, pobiera je, ustawia sposób ładowania, ładuje,
wyładowuje i usuwa, zmienia **dowolne** ustawienie i czyta diagnostykę. Ktoś, kto
nie wie, czym jest kwantyzacja albo okno kontekstu, pisze „pobierz mi coś do
kodowania, co się zmieści" — i to się dzieje.

```bash
localcode pomoc                       # rozmowa w terminalu (wznawia ostatnią)
localcode pomoc "co mogę usunąć?"     # od razu z pytaniem
localcode pomoc --new                 # nowa rozmowa
localcode pomoc --install             # tylko pobierz model pomocnika
```

W aplikacji: przycisk **Pomocnik** w pasku narzędzi (⌘⇧H) otwiera panel obok
bieżącej sekcji — obok, a nie zamiast, bo cały sens to widzieć, jak zmiana ląduje
na swoim ekranie.

## Model

Domyślnie `mlx-community/Qwen3-4B-Instruct-2507-4bit`, ok. 2,3 GB
(`settings.helpbot.model`). Nie mniejszy, bo ten projekt to zmierzył
(`hub/metadata.py`, `AGENT_MIN_PARAMS_B`): poniżej ok. 3 mld parametrów model
wywołuje właściwe narzędzie i wypełnia je bzdurą — a to jedyna awaria, która
liczy się u bota, który pobiera i usuwa gigabajty.

„Przychodzi z aplikacją":

* `scripts/install.sh` pobiera go od razu (`--no-helpbot`, żeby pominąć),
* serwer przy starcie dociąga go, jeśli go nie ma (`helpbot.autoDownload`;
  zmienna `LOCALCODE_NO_HELPBOT_DOWNLOAD=1` wyłącza to w testach i CI),
* panel w aplikacji ma przycisk „Pobierz pomocnika" z postępem.

Każda z tych dróg **najpierw sprawdza miejsce na dysku** (model + 2 GB zapasu,
`helpbot.room_problem`) i odmawia z czytelnym komunikatem, zamiast zapełnić dysk.

Ładowany jest na żądanie, nie trzymany w pamięci. Kto chce, żeby odpowiadał
natychmiast, przypina go: `runtime.pinnedModels`.

## Jak to jest zbudowane

To **ta sama pętla co agent kodujący** (`agent/loop.py`), z innym zadaniem:

| | agent kodujący | pomocnik |
|---|---|---|
| prompt | `agent/prompts.py` | `helpbot/prompt.py` (`Agent(system_prompt=…)`) |
| narzędzia | pliki, `bash`, `grep`… | `helpbot/tools.py` — 10 narzędzi, bez plików i powłoki |
| fasada | — | `helpbot/ops.HelpbotOps` (`Agent(tool_state=…)`) |
| katalog roboczy | projekt | `<LOCALCODE_HOME>/helpbot` |

Z tego samego powodu terminal i aplikacja pokazują go widokami, które już mają —
transkrypcją, kartami zgody i polem wpisywania z sekcji Kod.

**Katalog roboczy decyduje, że sesja jest pomocnikiem.** Serwer
(`AgentHub._default_agent`) i terminal (`LocalBackend._ensure_agent`) budują
pomocnika dla każdej sesji w `helpbot_dir()`. Dzięki temu rozmowa przetrwa
restart i wznowienie bez nowego pola w formacie sesji, a rozmowa zaczęta w
terminalu da się kontynuować w aplikacji.

`HelpbotOps` stoi na `Core`, więc ustawienia zmienia tym samym kodem co ekran
ustawień — z tą samą walidacją, zapisem i aktualizacją tego, co już działa.

## Gdzie kończy się moc

Pełna moc na modelu 4B wymaga twardych granic:

* **Wszystko, co coś zmienia, pyta.** Pobranie, załadowanie, wyładowanie,
  usunięcie, zmiana ustawień, przerwanie pobierania. Czytanie — nigdy.
  Narzędzia, które umieją jedno i drugie (`settings`, `model_settings`,
  `downloads`), pytają tylko wtedy, gdy mają coś zmienić.
* **Klucz API nie wchodzi do rozmowy.** Zostałby w jej zapisie na zawsze.
  Pomocnik odsyła do Ustawienia → Modele zdalne.
* **Nie usunie własnego modelu** i **nie wyładuje się przez „wyładuj wszystko"**
  — trzyma model tylko podczas pisania, więc wyładowanie go w połowie tury
  kazałoby następnemu zdaniu czekać na ponowne ładowanie i mogłoby wypchnąć model,
  który właśnie miał załadować.
* **Ładując model dla użytkownika, zostaje pomocnikiem.** Narzędzie agenta
  `model_load` przekazuje rozmowę nowemu modelowi; tutaj `on_switch` jest pusty.
  Z tego samego powodu panel w aplikacji nie słucha wyboru modelu w nagłówku.
* **Model, który się nie zmieści, nie jest ładowany** — pomocnik dostaje
  szacunek pamięci i proponuje mniejszy.

## HTTP

```
GET  /api/helpbot                 → stan: ready | downloading | missing (+ model, postęp)
POST /api/helpbot/install         → zaczyna pobieranie (409, gdy brak miejsca)
POST /api/helpbot/session {new?}  → sesja pomocnika; rozmowa przez zwykły WS agenta
```
