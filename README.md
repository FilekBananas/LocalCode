# LocalCode

**Agent kodujący w stylu Claude Code i czat — w całości na Twoim Macu.**
Modele pobierasz z Hugging Face (GGUF przez `llama-server` albo MLX przez `mlx-lm`),
uruchamiasz lokalnie i używasz w terminalu (`localcode`) oraz w natywnej aplikacji
na Maca. Nic nie wychodzi do chmury — poza pobieraniem modeli.

```
localcode                     agent kodujący w bieżącym katalogu
localcode -p "co to robi?"    jedna odpowiedź na stdout (skrypty, CI)
localcode chat                czat bez narzędzi
localcode models curated      katalog modeli — polecane na ten komputer
localcode models install NAZWA  instalacja jednym poleceniem
localcode serve               serwer HTTP + API zgodne z OpenAI
```

Wymagania: **macOS na Apple Silicon** (M1–M5), **Python 3.11–3.13**, ok. 10 GB
wolnego miejsca na pierwszy model. LocalCode działa też na Macach z Intelem,
ale wtedy tylko z modelami GGUF (MLX wymaga Apple Silicon).

---

## Spis treści

1. [Instalacja](#instalacja)
2. [Aplikacja terminalowa](#aplikacja-terminalowa)
3. [Aplikacja na Maca](#aplikacja-na-maca)
4. [Modele z Hugging Face](#modele-z-hugging-face)
5. [Środowisko modeli (runtime)](#środowisko-modeli-runtime)
6. [Mały kontekst — jak LocalCode sobie z nim radzi](#mały-kontekst--jak-localcode-sobie-z-nim-radzi)
7. [Czat: projekty, pamięć, baza wiedzy](#czat-projekty-pamięć-baza-wiedzy)
8. [Generowanie obrazów](#generowanie-obrazów)
9. [Serwer zgodny z OpenAI](#serwer-zgodny-z-openai)
10. [Skróty klawiszowe](#skróty-klawiszowe)
11. [Ściąga: polecenia](#ściąga-polecenia)
12. [Rozwiązywanie problemów](#rozwiązywanie-problemów)
13. [Dla programistów](#dla-programistów) — w tym [jak tego używać](docs/UZYCIE.md) i [dokumentacja API](docs/API.md)

---

## Instalacja

```bash
git clone <adres-repozytorium> localcode && cd localcode
./scripts/install.sh
```

Skrypt:

1. tworzy środowisko Pythona w `~/.localcode/venv` i instaluje LocalCode
   (na Apple Silicon razem z `mlx-lm`),
2. instaluje `llama.cpp` przez Homebrew, jeśli `llama-server` nie jest dostępny,
3. tworzy komendy `localcode` i `lc` w `~/.local/bin`,
4. buduje aplikację `macos/dist/LocalCode.app` (potrzebny Xcode; pomiń przez `--no-app`),
5. na koniec uruchamia `localcode doctor`.

Opcje: `--no-app` (bez aplikacji na Maca), `--app-only` (tylko przebudowa aplikacji),
`--dev` (używa `.venv` w repozytorium zamiast `~/.localcode/venv`).

Jeśli `localcode: command not found`, dodaj katalog do `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh
```

Sprawdzenie instalacji:

```bash
localcode doctor
```

`doctor` wypisuje listę kontrolną (procesor i wersja macOS, pamięć, wolne miejsce,
Python, `llama-server`, `mlx_lm`, katalog modeli, działający serwer, dostęp do
Hugging Face, budżet RAM). Przy każdym problemie podaje konkretną komendę do
naprawy i kończy się kodem `1`, gdy brakuje czegoś niezbędnego.

### Gdzie LocalCode trzyma dane

```
~/.localcode/
  settings.json          ustawienia (uprawnienia pliku 600 — może zawierać token HF)
  server.json            port i token działającego serwera
  models/<org>/<repo>/   pobrane modele
  data/chat.db           rozmowy czatu, pamięć i baza wiedzy (SQLite)
  data/uploads/          załączniki i pliki bazy wiedzy
  sessions/<projekt>/    transkrypty sesji agenta (JSONL)
  logs/                  logi silników i serwera
  history/tui.txt        historia wpisanych poleceń
```

Biblioteka pokazuje **tylko modele pobrane przez LocalCode**. Jeśli chcesz widzieć
także modele leżące już na dysku (z **LM Studio** w `~/.lmstudio/models` albo z cache
Hugging Face), włącz to w Ustawieniach aplikacji lub w `~/.localcode/settings.json`
(`scan_lmstudio`, `scan_hf_cache`) — domyślnie obie opcje są wyłączone.

---

## Aplikacja terminalowa

```bash
cd ~/projekty/moj-projekt
localcode
```

Przy pierwszym uruchomieniu (gdy nie masz żadnego modelu) LocalCode pokaże listę
polecanych modeli mieszczących się w pamięci tego Maca i pobierze wybrany —
z paskiem postępu. Potem wita Cię ramka z modelem, katalogiem i trybem:

```
╭─ LocalCode ────────────────────────────────────────────────────────╮
│    model  Qwen3 4B Instruct 2507 · Q4_K_M · GGUF · 16,4k kontekstu │
│  katalog  ~/projekty/moj-projekt                                   │
│     tryb  domyślny (pyta o zmiany plików i polecenia)              │
│                                                                    │
│  /help — lista poleceń, /model — wybór modelu, /context — kontekst │
│  @plik.py dołącza plik, Shift+Tab zmienia tryb, Esc przerywa       │
╰────────────────────────────────────────────────────────────────────╯
```

Wpisujesz polecenie, model czyta pliki, uruchamia narzędzia i odpowiada:

```
› napraw failujący test w tests/test_api.py

● Odczyt tests/test_api.py (1–200)
  ⎿ 143 linie
✻ Myślenie… (412 tokenów)
● Edytuj tests/test_api.py
  ⎿ 1 zmiana
     @@ -87,7 +87,7 @@
     -    assert response.status == 200
     +    assert response.status == 201
● Uruchom: pytest tests/test_api.py -q
  ⎿ exit 0
     1 passed in 0,41s

Test przechodzi — asercja oczekiwała 200, a endpoint zwraca 201 przy tworzeniu.
  14,2 tok/s · 3 kroki · 18,4 s · 612 tokenów odpowiedzi
```

### Tryby uprawnień

Zmieniasz je klawiszem **Shift+Tab** albo poleceniem `/mode`:

| tryb | zachowanie |
|---|---|
| `default` | odczyt bez pytania, zmiany plików i polecenia — za zgodą |
| `accept_edits` | zmiany plików bez pytania, polecenia — za zgodą |
| `auto` | jak `default`, ale akcję wymagającą zgody oceniają dwa modele naraz: krytyczny **recenzent** z kontekstem (model agenta albo wybrany w Ustawieniach) i **strażnik lokalny** bez kontekstu (zawsze model lokalny, widzi tylko polecenie). Wykona się sama tylko przy dwóch TAK; inaczej pyta Ciebie z opinią „Co robi / Powód”. Polecenia niebezpieczne i wysyłkę danych do dostawcy zawsze decydujesz Ty — szczegóły w [docs/UZYCIE.md](docs/UZYCIE.md#tryb-auto) |
| `plan` | **tylko odczyt**: żadnych zmian ani poleceń, model ma przedstawić plan |
| `yolo` | wszystko bez pytania, poza poleceniami niebezpiecznymi (`rm -rf ~`, `curl … \| sh`) |

Gdy potrzebna jest zgoda, LocalCode pokazuje dokładnie to, co ma się wydarzyć
(różnicę w pliku albo polecenie) i pyta:

```
╭─ Uruchom: pytest -q ────────────────────────────────────────────────╮
│ pytest -q                                                           │
╰─────────────────────────────────────────────────────────────────────╯
Czy pozwolić na tę operację?
 ❯ 1. Tak
   2. Tak, i nie pytaj więcej o polecenia: pytest
   3. Nie (Esc)
```

Odpowiedź „2” pamięta zgodę **na czas tej sesji** — dla narzędzi plikowych na
całe narzędzie, dla powłoki na *prefiks polecenia* (`git commit`, `npm test`,
`python -m pytest`), więc `rm -rf /` nigdy nie przejdzie jako „pytest”.

### Tylko folder projektu

Agent pracuje w folderze, w którym uruchomiono `localcode` (w aplikacji — w
folderze wybranym dla rozmowy w sekcji Kod), i **poza niego nie wychodzi bez
pytania — w żadnym trybie, także w `yolo`**:

* narzędzia plikowe pytają o każdy odczyt i zapis poza projektem;
* polecenia powłoki działają w piaskownicy macOS: nie widzą reszty katalogu
  domowego (dokumentów, kluczy SSH, innych projektów, ustawień LocalCode) i nie
  mogą niczego zapisać poza projektem — wyjątki to pliki tymczasowe, pamięć
  podręczna narzędzi (`~/.npm`, `~/.cache`, `~/Library/Caches`…) i odczyt
  narzędzi programisty (`~/.gitconfig`, `~/.cargo`, `~/.nvm`…);
* polecenie, które wprost wskazuje miejsce poza projektem (`cat ~/.zshrc`,
  `ls ..`, `find /`), jest oznaczone **⚠ poza projektem** i czeka na Twoją
  zgodę; po zgodzie wykonuje się bez piaskownicy.

Wyłączenie (polecenia z pełnymi uprawnieniami, jak dawniej):
`"agent": {"confine_to_project": false}` w `~/.localcode/settings.json`.

### Dołączanie plików: `@`

Wpisz `@` i zacznij pisać ścieżkę — pojawią się podpowiedzi z projektu (bez
`node_modules`, `.git`, `.venv`). Po wysłaniu plik trafia do wiadomości
w bloku `<file path="…">`, **skrócony do części okna kontekstu** (domyślnie 25 %),
żeby jeden nieuważny `@` nie wysadził modelu z 4k kontekstu. `@katalog` wstawia
listę plików zamiast treści.

### Wznawianie pracy

```bash
localcode -c                       # ostatnia sesja w tym katalogu
localcode --resume                 # wybór z listy
localcode --resume 20260923-141503-a1b2c3
```

Transkrypty leżą w `~/.localcode/sessions/<projekt>/<id>.jsonl` i są dopisywane
w locie, więc nawet zamknięcie terminala w środku tury niczego nie gubi.
W trakcie pracy: `/resume`.

### Tryb bezinterfejsowy (skrypty, CI)

```bash
localcode -p "wypisz publiczne API tego modułu"            # tekst na stdout
localcode -p "napraw lint" --mode accept_edits             # z prawem do zmian
localcode -p "co tu jest nie tak?" --output-format json    # jeden obiekt JSON
localcode -p "zrób przegląd" --output-format stream-json   # zdarzenia linia po linii
echo "podsumuj README" | localcode -p                      # prompt ze stdin
```

Kod wyjścia to `0` przy powodzeniu i `1` przy błędzie, przerwaniu albo
przekroczeniu limitu kroków — nadaje się do `set -e`. W tym trybie nikt nie
odpowiada na pytania o zgodę, więc operacje wymagające zgody są **odrzucane**;
użyj `--mode accept_edits`, `--mode auto` albo `--mode yolo`, jeśli agent ma coś zmieniać.
Komunikaty pomocnicze idą na `stderr`, więc `localcode -p … > odpowiedz.md`
zapisuje samą odpowiedź.

### Polecenia ukośnikowe

| polecenie | działanie |
|---|---|
| `/help` | lista poleceń i skrótów |
| `/model [klucz]` | wybór modelu (lista z rozmiarem, kwantyzacją i tym, czy zmieści się w RAM) |
| `/models` | modele zainstalowane na dysku |
| `/search zapytanie` | szukanie na Hugging Face |
| `/download repo [KWANT]` | pobranie modelu z paskiem postępu |
| `/load [klucz]`, `/unload [id\|all]`, `/ps` | ręczne zarządzanie pamięcią |
| `/runtime [set klucz wartość]` | ustawienia środowiska modeli |
| `/mode [tryb]` | tryb uprawnień |
| `/think [on\|off]` | pokazuj albo zwijaj rozumowanie modelu |
| `/goal cel` | agent pracuje, aż cel zostanie osiągnięty i sprawdzony — po każdej odpowiedzi sam wraca do pracy (do 20 razy, `agent.goal_max_turns`); kończy narzędziem `goal_done`, Esc/Stop przerywa; `/goal` pokazuje stan, `/goal wyczyść` usuwa cel |
| `/compact` | skróć rozmowę teraz |
| `/clear` | wyczyść rozmowę i kontekst |
| `/context` | co dokładnie zajmuje okno kontekstu |
| `/resume [id]` | wznów wcześniejszą sesję |
| `/stats` | statystyki ostatniej tury |
| `/init` | agent pisze `LOCALCODE.md` dla tego projektu |
| `/chat` | przełączenie między agentem a czatem |
| `/doctor` | diagnostyka bez wychodzenia z aplikacji |
| `/cwd`, `/exit` | katalog roboczy · wyjście |

**Obrazki.** Jeśli aktywny model widzi obrazy (w `/model` ma dopisek „widzi obrazy” —
lokalnie np. Qwen3‑VL, zdalnie np. Kimi K2.6 albo Qwen3‑VL), dołącz obrazek przez
`@zrzut.png` albo po prostu przeciągnij plik na okno Terminala. Duże zdjęcia i HEIC
są zmniejszane do JPEG. Model bez wizji nie dostanie wiadomości — terminal od razu
podpowie, żeby się przełączyć.

`LOCALCODE.md` (a także `AGENTS.md` i `CLAUDE.md`) z katalogu projektu trafia do
promptu systemowego — to miejsce na zasady projektu: jak budować, jak testować,
czego nie ruszać.

### Lokalnie czy przez serwer

LocalCode sam wybiera:

* jeśli działa serwer (`localcode serve` albo aplikacja na Maca), terminal
  **dołącza się do niego** — modele już wczytane do pamięci są używane
  ponownie, zamiast ładować je drugi raz;
* jeśli serwera nie ma, wszystko dzieje się w procesie terminala.

Wymuszenie: `--local` (zawsze w tym procesie) albo `--attach` (wymagaj serwera;
błąd, gdy go nie ma). Dotyczy to też `localcode ps`, `load`, `unload`,
`runtime` i `models` — pokazują stan tej instancji, z którą rozmawiają.

---

## Aplikacja na Maca

```bash
./macos/scripts/build-app.sh      # albo ./scripts/install.sh
localcode app                     # otwiera LocalCode.app
```

Aplikacja (SwiftUI) ma zakładki **Czat**, **Kod** (ten sam agent co w terminalu),
**Odkrywaj** (wyszukiwanie i pobieranie modeli z Hugging Face), **Moje modele**
i **Runtime** (pamięć, instancje, limity). Startuje własny serwer Pythona i
zapisuje jego port oraz token w `~/.localcode/server.json`, dzięki czemu
terminal automatycznie się do niego dołącza. Aplikację można skopiować do
`/Applications`:

```bash
cp -R macos/dist/LocalCode.app /Applications/
```

Jedna pułapka macOS, którą instalator obchodzi za Ciebie: aplikacja z
`/Applications` nie ma dostępu do folderu **Dokumenty**. Gdyby jej środowisko
Pythona leżało w repozytorium trzymanym w `~/Documents`, serwer zawisłby przy
starcie (system czeka na zgodę, której nikt nie widzi). Dlatego `install.sh`
instaluje **kopię** LocalCode w `~/.localcode/venv` i to na nią wskazuje
aplikacja; tryb `--dev` zostaje przy `.venv` w repozytorium i jest przeznaczony
do uruchamiania z `macos/dist/`.

Ikona: wrzuć własny PNG 1024×1024 jako `macos/Resources/AppIcon.png` i przebuduj
aplikację — skrypt sam zrobi z niego `.icns`. Bez własnego pliku rysowana jest
prosta ikona zastępcza (`macos/scripts/make_icon.py`).

Czat w terminalu i czat w aplikacji to ta sama baza (`~/.localcode/data/chat.db`) —
rozmowa zaczęta w `localcode chat` jest widoczna w aplikacji i odwrotnie.

---

## Katalog modeli

Modele instalujesz po nazwie — LocalCode sam dobiera wersję pliku do pamięci
Twojego Maca:

```bash
localcode models curated                       # katalog: co jest i do czego służy
localcode models curated --category kodowanie  # kodowanie | czat | rozumowanie |
                                               # obrazy | wizja | embedding
localcode models search qwen coder             # szukanie po nazwie
localcode models install "Qwen3 Coder 30B"     # jedno polecenie — wariant automatycznie
localcode models list
```

Katalog pokazuje nazwę, do czego model służy, rozmiar i czy zmieści się
w pamięci („Zmieści się” / „Na styk” / „Za duży”). Jeśli model się nie mieści,
`install` zapyta o potwierdzenie. W środku jest **kilka tysięcy modeli wbudowanych
w aplikację**, więc lista działa też bez internetu — sieć jedynie dokłada nowości.

### Zaufanie i źródło

Przy każdym modelu stoi liczba 0–100 i jedno słowo o pochodzeniu:

| odznaka | znaczenie |
| --- | --- |
| `oficjalne` | repozytorium pracowni, która wytrenowała model (Qwen, Google, Mistral…) |
| `konwersja` | przygotowane przez znany zespół (unsloth, bartowski, mlx-community…) |
| `społeczność` | znane wagi, plik złożony przez niezależną osobę |
| `nieznane` | nie udało się ustalić, czyj to model |

Liczba bierze się z pochodzenia pliku, licencji, formatu wag, popularności
i daty ostatniej aktualizacji — **to nie jest skan antywirusowy**, tylko miara
tego, ile w ogóle wiadomo o pliku. Uzasadnienie widać w dymku przy odznace
i w całości w „Szczegółach”; szczegóły reguły opisuje
[docs/CATALOG.md](docs/CATALOG.md).

Dwie rzeczy, których popularność nie przebije: repozytorium z plikami `pickle`
(`.bin`, `.pt`, `.ckpt` — przy wczytaniu potrafią wykonać kod) i repozytorium
bez ustalonego pochodzenia wag nigdy nie dostaną zielonej odznaki.

```bash
localcode models search qwen --official        # tylko repozytoria wydawców
localcode models search qwen --min-trust 60    # pomiń słabo udokumentowane
localcode models search kod --category kodowanie
```

W aplikacji na Maca te same filtry są nad listą: „Tylko od wydawcy”
i „Tylko sprawdzone”.

Modele pobierane są z **Hugging Face**. Identyfikatory repozytoriów i etykiety
kwantyzacji są schowane — pokazuje je `--advanced`:

```bash
localcode models curated --advanced            # z kolumną „repozytorium”
localcode models search qwen --advanced
localcode models info unsloth/Qwen3-4B-Instruct-2507-GGUF
localcode models download unsloth/Qwen3-4B-Instruct-2507-GGUF Q4_K_M   # ręczny wybór
localcode models rm unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M
```

Bez podanej kwantyzacji `download` pokazuje listę wariantów z rozmiarem i
oszacowaniem pamięci, z zaznaczonym wariantem zalecanym. Pliki lądują
w `~/.localcode/models/<org>/<repo>/`, pobieranie jest wznawialne — przerwane
`Ctrl+C` i powtórzone zaczyna od miejsca, w którym stanęło.

### GGUF czy MLX?

| | **GGUF** (`llama-server`) | **MLX** (`mlx-lm`) |
|---|---|---|
| dostępność | wszystkie Maki, ogromny wybór modeli | tylko Apple Silicon |
| szybkość | bardzo dobra | zwykle szybsza na Apple Silicon |
| kwantyzacje | `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`, … | `4bit`, `6bit`, `8bit`, `bf16`, `MXFP4` |
| obrazy | przez plik `mmproj` (pobierany automatycznie) | zależnie od modelu |
| cache KV | `f16`, `q8_0`, `q4_0` | `f16`, `8bit`, `4bit` |

W praktyce: jeśli model jest dostępny w obu formatach, MLX jest zwykle
szybszy; GGUF ma więcej wariantów i lepiej znosi ciasną pamięć.

### Którą kwantyzację wybrać

* **`Q4_K_M` / `4bit`** — domyślny wybór. Najlepszy stosunek jakości do pamięci,
  model 4B zajmuje ok. 2,5 GB, 14B ok. 9 GB.
* **`Q5_K_M`, `Q6_K` / `6bit`** — gdy masz zapas pamięci i zależy Ci na jakości.
* **`Q8_0` / `8bit`** — praktycznie bez strat, dwa razy większy niż Q4.
* **`IQ`/`IQ4_XS`** — najmniejsze, kosztem jakości; dobre na 8 GB RAM.

Reguła kciuka: **model + cache KV + narzut ≈ rozmiar pliku × 1,25**. Musi się
zmieścić w budżecie RAM (domyślnie 70 % pamięci) razem ze wszystkim, co już
jest załadowane. `localcode models info <repo>` liczy to dokładnie dla każdego
wariantu.

### Token Hugging Face

Do modeli publicznych nie jest potrzebny. Do modeli „gated” (np. niektóre
Llamy) ustaw go w aplikacji (Ustawienia → Hugging Face) albo w
`~/.localcode/settings.json` (`"hf_token": "hf_…"`). Plik ma prawa `600`.

---

### Sprawdzanie modeli Cerberem

Jeśli masz Cerbera (antywirus dla macOS), LocalCode może poprosić go o
przeskanowanie każdego modelu **przed pobraniem** — w aplikacji, w terminalu i
przez Pomocnika:

1. W Cerberze: **Integracje → API (localhost)** — włącz API i skopiuj token.
2. W LocalCode: **Ustawienia → Cerber** — wklej token i włącz „Sprawdzaj modele
   Cerberem przed pobraniem”.

Cerber skanuje repozytorium w swojej piaskownicy (kod, konfiguracje, szablony
czatu, wagi pickle). Werdykt „podejrzany”, „potencjalnie niechciany” albo
„złośliwy” blokuje pobranie; „czysto” i „drobne uwagi” je przepuszczają. Gdy
sprawdzanie jest włączone, a Cerber jest zamknięty, pobranie **nie rusza** —
zamiast po cichu pominąć kontrolę, LocalCode prosi o otwarcie Cerbera.

## Środowisko modeli (runtime)

Jeden komponent (`RuntimeManager`) decyduje, co jest w pamięci: ładuje model
przy pierwszym żądaniu, współdzieli go między agentem, czatem i API,
a nieużywane instancje zwalnia.

```bash
localcode ps                      # co jest załadowane, ile pamięci, ile tok/s
localcode load qwen3-4b --ctx 16384 --parallel 2
localcode unload inst-1           # albo: localcode unload --all
localcode runtime                 # ustawienia + zajętość pamięci
localcode runtime set ram-gb 20
```

| ustawienie | `runtime set` | domyślnie | znaczenie |
|---|---|---|---|
| modeli naraz | `max-models` | 2 | ile **różnych** instancji może działać jednocześnie |
| instancji na model | `instances` | 1 | ile kopii tego samego modelu (prawdziwa równoległość) |
| równoległych żądań | `parallel` | 1 | sloty w jednej instancji (`-np` w llama.cpp, batch w MLX) |
| limit RAM | `ram-gb` | auto (70 % RAM) | twardy budżet dla wszystkich modeli |
| wyładuj po | `idle-ttl` | 30 | minut bezczynności do automatycznego zwolnienia (`0` = nigdy) |
| domyślny kontekst | `ctx` | auto | okno kontekstu przy ładowaniu (auto = tyle, ile się zmieści) |
| cache KV | `kv` | `f16` | `q8_0` zmniejsza cache o połowę niewielkim kosztem jakości |
| flash attention | `flash-attn` | `auto` | dla GGUF |
| polityka zwalniania | `evict` | `lru` | `none` = nigdy nie wyładowuj automatycznie, odmawiaj ładowania |

Jak to działa w praktyce:

* **Ładowanie na żądanie.** Model wczytuje się przy pierwszej wiadomości;
  w terminalu widzisz „Ładowanie modelu…”.
* **Współdzielenie.** Agent w terminalu, czat i API korzystają z tej samej
  instancji — dlatego terminal domyślnie dołącza się do serwera aplikacji.
* **Budżet i zwalnianie.** Przed każdym ładowaniem sprawdzany jest limit RAM;
  gdy brakuje miejsca, zwalniane są instancje bezczynne (najdawniej używane
  jako pierwsze). Zajęte nigdy nie są przerywane. Jeśli mimo to się nie mieści,
  dostajesz wyjaśnienie, co zmienić (mniejszy kontekst, mniej slotów, zwolnienie
  konkretnego modelu, wyższy limit).
* **Równoległość.** `parallel 2` pozwala na dwa żądania naraz w jednej
  instancji, ale okno kontekstu jest dzielone między sloty — na małych
  modelach zwykle lepiej zostawić `1`.

---

## Mały kontekst — jak LocalCode sobie z nim radzi

Lokalny model 4B ma zwykle 8–32k kontekstu zamiast setek tysięcy. Dlatego
oszczędzanie kontekstu jest wbudowane w każdą warstwę:

1. **Prompt systemowy skaluje się do okna** — przy ≤ 8k używany jest wariant
   skrócony i węższy zestaw narzędzi (opisy narzędzi płacisz w tokenach
   w *każdym* żądaniu).
2. **Stały prefiks promptu.** System i narzędzia nigdy się nie zmieniają
   w trakcie sesji, historia tylko rośnie — dzięki temu silnik ponownie używa
   cache KV i kolejny krok trwa sekundy zamiast dziesiątek sekund.
3. **Każdy wynik narzędzia jest przycinany przy wstawianiu** (początek + koniec,
   z jawnym `[… pominięto N linii …]`), a `read_file` czyta stronami po ~200 linii.
4. **Automatyczne kompaktowanie** po przekroczeniu progu (domyślnie 80 % okna),
   trzystopniowo: ukrycie starych wyników narzędzi → podsumowanie starszych tur
   przez model (zadanie, decyzje, dotknięte pliki, lista zadań) → usunięcie
   najstarszych wiadomości. Każdy krok widać w terminalu:
   `Kontekst skompaktowany: 12,4k → 5,1k tokenów`.
5. **Rozumowanie (`<think>`) nigdy nie wraca do modelu.** Widzisz je (`/think on`),
   ale w historii go nie ma.
6. **Wskaźnik na dole** pokazuje `ctx 5,1k/8,2k (63%)` — a `/context` rozbija to
   na prompt systemowy, opisy narzędzi, rozmowę i rezerwę na odpowiedź.

Gdy mimo wszystko zrobi się ciasno: `/compact` (skrót teraz), `/clear` (od zera),
albo większe okno — `localcode load <model> --ctx 16384`.

---

## Czat: projekty, pamięć, baza wiedzy

```bash
localcode chat                      # REPL bez narzędzi
localcode chat -m qwen3-4b "cześć"
```

Czat nie ma dostępu do plików ani powłoki — to zwykła rozmowa z modelem.
Rozmowy zapisują się w `~/.localcode/data/chat.db` i są widoczne w aplikacji
na Maca, gdzie dochodzą trzy rzeczy:

* **Projekty (foldery).** Każdy ma własne instrukcje i własną pamięć. Projekt
  „współdzielony” widzi też kontekst globalny, „izolowany” — wyłącznie swój
  (nic globalnego nie wycieka do rozmowy).
* **Pamięć długoterminowa.** Notatki o Tobie i o projekcie dołączane do promptu.
  Zdanie „zapamiętaj, że…” w rozmowie zapisuje wpis.
* **Baza wiedzy (RAG).** Wgrywasz pliki (txt, md, kod, csv, json, html, pdf, docx),
  są dzielone na fragmenty i wyszukiwane przy każdej wiadomości. Wyszukiwanie
  jest hybrydowe: pełnotekstowe (FTS5/BM25) plus — jeśli ustawisz model
  embeddingowy — semantyczne, łączone metodą reciprocal rank fusion.
  Odpowiedź dostaje przypisy do źródeł.

Model embeddingowy ustawiasz w aplikacji albo w `settings.json`
(`chat.embedding_model`); polecany:
`nomic-ai/nomic-embed-text-v1.5-GGUF` (`Q8_0`, ok. 150 MB).

Wszystko to mieści się w oknie kontekstu z twardym budżetem: historia jest
przycinana od najstarszych wiadomości, a pamięć i fragmenty z bazy wiedzy mają
z góry ustalony udział.

---

## Generowanie obrazów

LocalCode generuje obrazy lokalnie, tymi samymi modelami co aplikacje graficzne
na Maca: **FLUX.1 / FLUX.2 / Qwen-Image** przez [mflux](https://github.com/filipstrand/mflux)
(MLX, Apple Silicon) oraz **Stable Diffusion 1.5 / SDXL / SD 3.5** i FLUX w
formacie GGUF przez [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(Metal). Modele pobierasz tak samo jak językowe — z Hugging Face, do
`~/.localcode/models` — a każdy działa we własnym procesie roboczym, więc
zwolnienie pamięci naprawdę ją zwalnia, a awaria silnika nie kładzie aplikacji.

Obsługa obrazów jest **dodatkiem opcjonalnym**:

```bash
pip install "localcode[images]"       # mflux + stable-diffusion-cpp-python
```

> **Który `pip`?** Aplikacja na Maca uruchamia serwer z `~/.localcode/venv`,
> a `pip` w PATH to zwykle zupełnie inny Python. Dlatego komunikat w aplikacji
> podaje pełną ścieżkę interpretera (`localcode.config.install_command`) —
> to polecenie można wkleić i zadziała:
> `~/.localcode/venv/bin/python -m pip install "localcode[images]"`.

Bez niego reszta LocalCode działa normalnie, a przy próbie generowania
zobaczysz komunikat, co doinstalować. Galeria (przeglądanie tego, co już
powstało) działa zawsze.

### W terminalu

```bash
localcode image "kot w kapeluszu siedzi na dachu, akwarela"
localcode image "zamek nad jeziorem" -m schnell --size 832x1216 --steps 4 --seed 42
localcode image "trzy warianty tej samej sceny" -n 3 -o ~/Obrazy/scena.png --open

localcode images list                 # ostatnio wygenerowane
localcode images list --favorite
localcode images rm 1a2b3c4d          # wystarczy początek identyfikatora
localcode images open 1a2b3c4d
```

| opcja | znaczenie |
|---|---|
| `-m, --model KLUCZ` | model obrazu (fragment klucza wystarczy) |
| `--size SZERxWYS` | rozmiar, np. `1024x1024`, `832x1216` (wielokrotność 8) |
| `--steps N` | liczba kroków; domyślnie zależna od modelu (FLUX schnell 4, SD 20) |
| `--guidance X` | siła podpowiedzi |
| `--seed N` | ziarno — to samo ziarno i ten sam opis dają ten sam obraz |
| `--negative OPIS` | czego na obrazie ma nie być |
| `-n, --count ILE` | ile obrazów wygenerować |
| `-o, --output PLIK` | dodatkowa kopia pliku (przy `-n` numerowana) |
| `--open` | otwórz gotowy obraz w Podglądzie |

W trakcie generowania widać pasek postępu (krok po kroku), a po zakończeniu —
rozmiar, ziarno, czas i ścieżkę pliku. W iTerm2, WezTermie i kitty obraz
rysuje się od razu w terminalu (`--no-preview` to wyłącza).

W interfejsie agenta i czatu to samo robi polecenie `/obraz`:

```
› /obraz kot w kapeluszu, akwarela
```

### Gdzie to leży

```
~/.localcode/data/images.db          indeks (SQLite, WAL)
~/.localcode/data/images/2026-09/    pliki PNG, katalog na miesiąc
```

Przy każdym obrazie zapisane są wszystkie parametry generowania (opis, opis
negatywny, model, rozmiar, kroki, guidance, ziarno, czas), więc zawsze można go
powtórzyć albo użyć jako punktu wyjścia. Usunięcie obrazu kasuje też plik.
Katalog zmienisz ustawieniem `images.output_dir` w `~/.localcode/settings.json`
(razem z `images.default_model`, `images.default_size`, `images.default_steps`,
`images.default_guidance`, `images.max_loaded` i `images.idle_ttl_minutes`).

Modele obrazów dzielą budżet pamięci z modelami językowymi: zanim LocalCode
załaduje model obrazu, w razie potrzeby wyładuje bezczynny model językowy.
Widać je razem w `localcode ps` i w aplikacji na Maca.

### Przez HTTP

Przy działającym `localcode serve` (uruchamia go też aplikacja na Maca):

```
GET    /api/images/models                  modele obrazów + domyślne ustawienia
POST   /api/images/generate                strumień SSE: status, progress, image, done
GET    /api/images?limit&offset&favorite   galeria
GET    /api/images/stats                   ile obrazów, ile miejsca
GET    /api/images/{id}/file               plik PNG
PATCH  /api/images/{id} {"favorite": true}
DELETE /api/images/{id}
```

```bash
curl -N -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8421/api/images/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "kot w kapeluszu", "width": 1024, "height": 1024, "count": 1}'
```

Zerwanie połączenia przerywa generowanie — nic nie liczy się „w tle”.

---

## Serwer zgodny z OpenAI

```bash
localcode serve                        # port z ustawień (domyślnie 8421)
localcode serve --port 0               # wolny port wybrany przez system
localcode serve --token mój-token --verbose
```

Serwer wypisuje na stdout jedną linię `LOCALCODE_READY port=<p>` i zapisuje
`~/.localcode/server.json` (port, token, pid) z prawami `600`. `--parent-pid PID`
sprawia, że kończy się razem z procesem nadrzędnym.

Endpointy zgodne z OpenAI (jak lokalny serwer LM Studio):

```
GET  /v1/models
POST /v1/chat/completions      (strumieniowanie i narzędzia)
POST /v1/completions
POST /v1/embeddings
POST /v1/images/generations    (lokalne generowanie obrazów)
```

`/v1/images/generations` przyjmuje `{prompt, model?, n?, size?, seed?,
response_format?}`. Domyślnie odpowiada obrazem w `b64_json` — inaczej niż
api.openai.com, bo plik leży za tym samym tokenem co reszta `/api/*`.
`response_format: "url"` zwraca pełny adres `/api/images/{id}/file`; użyj go
tylko z klienta, który sam wyśle nagłówek `Authorization`.

Przykład:

```bash
curl http://127.0.0.1:8421/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M",
       "messages":[{"role":"user","content":"Cześć!"}],"stream":true}'
```

Nazwa modelu to jego klucz z `localcode models list`. Model ładuje się
automatycznie przy pierwszym żądaniu.

Domyślnie `/v1/*` **nie wymaga tokenu** (żeby działały narzędzia zakładające
lokalny serwer bez klucza), a żądania z nagłówkiem `Origin` — czyli ze stron
w przeglądarce — są odrzucane, o ile nie dopiszesz adresu do
`server.cors_origins`. Zmiana w `~/.localcode/settings.json`:

```json
{
  "server": {
    "port": 8421,
    "openai_api_enabled": true,
    "openai_api_requires_token": true,
    "cors_origins": ["http://localhost:3000"]
  }
}
```

Pozostałe API (`/api/*`, używane przez aplikację na Maca i przez terminal
w trybie `--attach`) zawsze wymaga nagłówka `Authorization: Bearer <token>`.

Pełna referencja (endpointy, uwierzytelnianie, WebSocket agenta, przykłady curl/Python/JavaScript):
[`docs/API.md`](docs/API.md).

---

## Skróty klawiszowe

| klawisz | działanie |
|---|---|
| `Enter` | wyślij wiadomość |
| `Alt+Enter`, `Ctrl+J` | nowa linia (albo zakończ linię znakiem `\`) |
| `Shift+Tab` | tryb: domyślny → akceptuj edycje → auto → plan → yolo |
| `Esc` | przerwij odpowiedź; przy pytaniu o zgodę = „Nie” |
| `Ctrl+C` | przerwij; dwa razy przy pustym polu — wyjście |
| `Ctrl+D` | wyjście |
| `Ctrl+R` | wyszukiwanie w historii poleceń |
| `Tab` | uzupełnianie (polecenia po `/`, ścieżki po `@`) |
| `↑` / `↓` | historia wpisanych wiadomości |
| `1` / `2` / `3` | odpowiedź na pytanie o zgodę |

---

## Ściąga: polecenia

```
localcode [POLECENIE]              agent w bieżącym katalogu
  -m, --model KLUCZ                model (wystarczy fragment klucza)
  --mode default|accept_edits|auto|plan|yolo
  --ctx N                          okno kontekstu przy ładowaniu
  -c, --continue                   wznów ostatnią sesję
  --resume [ID]                    wznów wskazaną sesję
  -p, --print [PROMPT]             jedna tura bez interfejsu
  --output-format text|json|stream-json
  --no-session                     nie zapisuj transkryptu
  --local | --attach               wymuś tryb lokalny / serwer
  -C, --cwd KATALOG                katalog projektu
  -V, --version

localcode chat [-m MODEL] [WIADOMOŚĆ]
localcode models list [--refresh]
localcode models search ZAPYTANIE [--format gguf|mlx] [--limit N] [--advanced]
                                  [--official] [--min-trust 0-100] [--category KATEGORIA]
localcode models install NAZWA [--variant WARIANT] [-y]
localcode models info REPO_LUB_KLUCZ
localcode models download REPO [KWANTYZACJA]
localcode models rm KLUCZ [-y]
localcode models curated [--category KATEGORIA] [--advanced] [--official]
localcode image "OPIS" [-m MODEL] [--size SZERxWYS] [--steps N] [--seed N]
                       [--guidance X] [--negative OPIS] [-n ILE] [-o PLIK] [--open]
localcode images [list [--favorite] | rm ID [-y] | open ID]
localcode ps
localcode load KLUCZ [--ctx N] [--parallel N] [--new]
localcode unload [ID | --all]
localcode runtime [show | set KLUCZ WARTOŚĆ]
localcode serve [--host H] [--port P] [--token T] [--parent-pid PID] [--verbose]
localcode app [--path /ścieżka/LocalCode.app]
localcode doctor [--json]
```

`lc` to krótszy alias `localcode`.

---

## Rozwiązywanie problemów

**`localcode: command not found`**
`~/.local/bin` nie jest w `PATH`:
`echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh`.

**`Brak jakiegokolwiek silnika` w `doctor`**
Zainstaluj co najmniej jeden backend: `brew install llama.cpp` (GGUF) albo
`pip install mlx-lm` (MLX, tylko Apple Silicon).

**Model nie chce się załadować: „nie mieści się w budżecie pamięci”**
Po kolei: `localcode unload --all`, mniejszy kontekst
(`localcode load <model> --ctx 8192`), oszczędniejszy cache
(`localcode runtime set kv q8_0`), mniejsza kwantyzacja (`Q4_K_M` zamiast `Q8_0`),
wyższy limit (`localcode runtime set ram-gb 24`). Ile czego potrzeba, pokaże
`localcode models info <klucz>`.

**Model odpowiada bardzo wolno**
Sprawdź `localcode ps` — jeśli `tok/s` spada z każdą turą, prawdopodobnie kontekst
jest pełny i cache KV jest przeliczany od nowa: `/compact` albo `/clear`.
Duży model przy prawie pełnej pamięci zaczyna korzystać ze swapu; wtedy pomaga
mniejszy wariant. `parallel 1` jest najszybsze dla pojedynczej rozmowy.

**„Model nie potrafi poprawnie sformułować wywołania narzędzia”**
Mały model nie trafia w format wywołań. Pomaga model z tagiem `tools`
(`localcode models list` pokazuje możliwości), np. Qwen3 Instruct, albo prostsze,
bardziej konkretne polecenie.

**Rozmowa ciągle się kompaktuje**
Okno jest za małe dla zadania. Załaduj model z większym oknem
(`localcode load <model> --ctx 16384`), pracuj nad mniejszą liczbą plików naraz
i używaj `@plik` zamiast wklejania całych plików.

**Terminal nie widzi modeli wczytanych przez aplikację**
Terminal dołącza się do serwera tylko wtedy, gdy `~/.localcode/server.json`
wskazuje **działający** serwer. Sprawdź `localcode doctor`; przy nieaktualnym
pliku: `rm ~/.localcode/server.json`. Wymuszenie: `localcode --attach`.

**Pobieranie modelu się zatrzymało**
Pobieranie jest wznawialne — powtórz `localcode models install "<nazwa>"`
(albo `localcode models download <repo> <kwant>`).
Przy modelach „gated” potrzebny jest token Hugging Face w `settings.json`.

**Zostawione procesy `llama-server` / `mlx_worker`**
`localcode unload --all` albo zamknięcie serwera. Serwer sam kończy silniki przy
wyjściu, a `--parent-pid` gwarantuje, że nie przeżyje aplikacji.

**Chcę zacząć od zera**
Usuń `~/.localcode/settings.json` (wrócą wartości domyślne). Rozmowy i pamięć
są w `~/.localcode/data/chat.db`, sesje agenta w `~/.localcode/sessions/`,
modele w `~/.localcode/models/` — każde z osobna można skasować.

---

## Dla programistów

```bash
.venv/bin/pytest tests -q -m "not real_model and not network"   # szybkie testy
.venv/bin/pytest tests/tui -q                                   # testy terminala
```

Kontrakt architektury i API HTTP: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Przewodnik użytkownika: [`docs/UZYCIE.md`](docs/UZYCIE.md). Lokalne API HTTP (OpenAI, `/api/*`, WebSocket agenta): [`docs/API.md`](docs/API.md).
Teksty dla użytkownika są po polsku, kod i prompty dla modelu po angielsku.

## Licencja

LocalCode jest udostępniany w modelu *source-available* na **LocalCode License 1.0**
(plik [LICENSE](LICENSE), wersja polska i angielska): bezpłatnie do użytku prywatnego
przez osoby fizyczne, bez prawa do użycia komercyjnego i przez organizacje oraz bez
rozpowszechniania modyfikacji. Licencje komercyjne — przez to repozytorium.
