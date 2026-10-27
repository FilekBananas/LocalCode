# Wideo: krótki klip z opisu tekstowego

Jedno zadanie, jeden model: opis tekstowy zamienia się w kilkusekundowy klip
`.mp4`. Inaczej niż w rozpoznawaniu/syntezie mowy (`docs/SPEECH.md`) to nie są
sekundy liczenia — to **minuty na sekundę materiału** na 32 GB Apple Silicon,
więc generowanie działa w osobnym procesie roboczym
(`python -m localcode.video.worker`), z paskiem postępu krok po kroku, żeby
crash albo zawieszony kernel Metala nie zabrał ze sobą serwera.

**Ten silnik nigdy nie wygenerował klipu na prawdziwym sprzęcie.** Cała logika
poniżej jest napisana i przetestowana bez pobierania wag modelu (zasada tej
pracy: nie ściągamy modeli) — sekcja „Co jest zweryfikowane, a co nie” na
końcu tego dokumentu mówi wprost, co jest pewne, a co jest tylko dobrze
uzasadnioną hipotezą.

## Model i biblioteka Pythona

* **Model:** wyłącznie **LTX-Video 2B** (`Lightricks/LTX-Video`) —
  najmniejszy model tekst-na-wideo w katalogu, jedyny z realną szansą na
  zmieszczenie się w 32 GB. Pozostałe ~90 pozycji z półki „Wideo” w sklepie
  (Wan, HunyuanVideo, MiniMax, linia LTX-2 13B+) są rozpoznawane po nazwie,
  ale świadomie odrzucane jako nieobsługiwane —
  `localcode/video/base.py:classify_video_model`.
* **Nie są modelami obrazów.** Plik GGUF z LTX-Video ma tensory
  `transformer_blocks.*` dokładnie tak jak FLUX, więc sonda w
  `localcode/hub/metadata.py` brała go za model dyfuzyjny do obrazów i
  wrzucała na półkę „Obrazy”; wybranie go tam kończyło się błędem
  `ValueError: NULL pointer access` z backendu obrazów. Dlatego
  `is_video_checkpoint()` (architektura GGUF `ltxv`/`wan`/`hunyuan-video`… albo
  nazwa pliku) wyklucza je z `ModelMeta.image_family`. Zmiana podbiła
  `CACHE_VERSION` w `localcode/hub/library.py`, bo stary `library-cache.json`
  trzymałby je na półce „Obrazy” mimo poprawki.
* **Format wag:** katalog ma dla wideo wyłącznie kwanty **GGUF** (community
  requanty pod ComfyUI-GGUF, np. `city96/LTX-Video-gguf`,
  `calcuis/ltxv-gguf`). MLX nie wchodzi w grę: istniejące porty MLX dla wideo
  celują w dużo większą rodzinę LTX-2 (13–19B), nie w ten model.
* **Biblioteka:** [`diffusers`](https://github.com/huggingface/diffusers)
  na PyTorchu, backend **MPS** (z awaryjnym przejściem na CPU, dużo wolniej,
  jeśli jakaś operacja nie ma pokrycia na MPS). `diffusers` wczytuje
  transformer GGUF wprost —
  `LTXVideoTransformer3DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(...))`
  — a resztę pipeline'u (VAE, enkoder tekstu T5, scheduler) bierze z
  podstawowego repo `Lightricks/LTX-Video`. **Te dodatkowe pliki `diffusers`
  ściąga sam, przy pierwszym użyciu, mimo cache'u Hugging Face — z pominięciem
  własnego menedżera pobierania LocalCode** (`/api/downloads` ich nie widzi,
  nie ma paska postępu ani budżetu RAM na nie).

## Instalacja

```bash
pip install "localcode[video]"     # diffusers, transformers, accelerate,
                                    # sentencepiece, torch, imageio[ffmpeg]
localcode doctor                   # sprawdzi, czego brakuje
```

> **Który `pip`?** Aplikacja na Maca uruchamia serwer z `~/.localcode/venv`,
> a `pip` w PATH to zwykle zupełnie inny Python. Dlatego komunikat w aplikacji
> podaje pełną ścieżkę interpretera (`localcode.config.install_command`) —
> to polecenie można wkleić i zadziała:
> `~/.localcode/venv/bin/python -m pip install "localcode[video]"`.

Bez dodatku wszystko inne działa dalej — `/api/video/models` i
`/api/video/estimate` nie dotykają `torch`/`diffusers` w ogóle, a
`localcode video`/`/api/video/generate` mówią po polsku, czego brakuje,
zamiast wysypać się `ImportError`-em (dokładnie ten komunikat, stałą
`INSTALL_HINT` w `localcode/video/base.py`):

> Brakuje dodatku do wideo (diffusers + torch — generowanie działa przez
> PyTorch/MPS, nie MLX). Zainstaluj:
> `~/.localcode/venv/bin/python -m pip install "localcode[video]"`

Model bierzesz ze sklepu, z kategorii **Wideo**:

```bash
localcode models search ltx-video
localcode models install "city96/LTX-Video-gguf"
```

## W terminalu

```bash
localcode video "kot w kapeluszu idzie ulicą"
localcode video "zachód słońca nad górami" --size 512x384 --frames 65 --steps 20
localcode video "taniec robota" --seed 42 -o klip.mp4 --force   # --force: więcej niż 81 klatek
```

Przed startem terminal zawsze pokazuje szacunkowy czas (patrz niżej) i dopiero
potem pasek postępu (ładowanie → odszumianie krok po kroku → dekodowanie).
Flagi: `-m/--model`, `--size SZERxWYS` (wielokrotność 32), `--frames`,
`--fps`, `--steps`, `--guidance`, `--seed`, `--negative`, `-o/--output`,
`--force`.

## W aplikacji

Sekcja **„Wideo"** w oknie LocalCode (`macos/Sources/VideoFeature/`). Ekran jest
zbudowany wokół czekania, bo sekunda materiału to około minuty liczenia:

* **wycena przed startem** — rozmiar, liczba klatek i kroków zmieniają zdanie
  „25 klatek (1,0 s materiału), 704×480, 8 kroków — szacunkowo ~1 min liczenia",
  zanim cokolwiek się zacznie (`POST /api/video/estimate`);
* **postęp w trakcie** — pasek z numerem kroku odszumiania i licznik czasu;
* **wynik z pomiarem** — odtwarzacz, opis, rozmiar, ziarno i ile to naprawdę
  trwało, plus „Zapisz jako…" i „Pokaż w Finderze";
* **modele, których nie uruchomimy**, wypisane z powodem — żeby pytanie „mam
  model wideo, czemu go nie widzę" miało odpowiedź na ekranie.

Czego **nie** sprawdzono: nikt nie przeszedł po tym ekranie klikaniem. To
środowisko nie ma uprawnienia do sterowania interfejsem (`osascript` dostaje
−25211), więc układ i odstępy są zweryfikowane tylko kompilacją, a kontrakt
danych — testami `tests/server/test_swift_contract.py`, które porównują
struktury Swift z prawdziwymi odpowiedziami serwera.

## Przez HTTP

```
GET  /api/video/models                    → modele wideo na dysku (obsługiwane i nie)
POST /api/video/estimate {width?,height?,numFrames?,fps?,steps?}
     → {seconds, message, width, height, numFrames, fps, steps}
POST /api/video/generate {prompt, modelKey?, negativePrompt?, width?, height?, numFrames?,
        fps?, steps?, guidance?, seed?, force?, output?}
     → SSE: progress (co krok odszumiania) / video (gotowy klip) / done / error
```

Strumień działa tak samo jak `/api/images/generate`: zamknięcie połączenia
przerywa zadanie i sprząta proces roboczy. Szczegóły pól — `docs/ARCHITECTURE.md`.

## Ograniczenia rozmiaru

LTX-Video pracuje na sztywnej siatce: szerokość/wysokość muszą być
wielokrotnością 32 (domyślnie 704×480, zalecany sufit producenta 768),
liczba klatek musi mieć postać `8k+1` (domyślnie 41 klatek ≈ 1,7 s przy
24 kl/s). Dwa limity liczby klatek, oba w `localcode/video/base.py`:

* **domyślny (81 klatek, ~3,4 s materiału)** — przekroczenie wymaga
  `--force`/`"force": true`; to już kilka minut liczenia;
* **bezwzględny (161 klatek, ~6,7 s materiału)** — **nigdy nieomijalny**,
  nawet przez `--force`, bo nikt nigdy nie zmierzył tego silnika powyżej tej
  granicy (to dosłownie "koniec sekwencji" z karty modelu, nic więcej).

## Pierwsze prawdziwe uruchomienie (29 września 2026)

Silnik **został przepuszczony przez realne wagi** na MacBooku z 32 GB:
`city96/LTX-Video-0.9.6-distilled-gguf` (Q4_K_M, 1,2 GB) + repozytorium bazowe
`Lightricks/LTX-Video` dociągnięte przez `diffusers`. Wynik: poprawny plik
H.264, 448×256, 25 klatek, 1,04 s materiału, 46 KB.

**Silnik generuje prawdziwe wideo.** Klip 704×480, 25 klatek, 8 kroków, model
`city96/LTX-Video-gguf` (Q4_K_M): rozpoznawalny banan na drewnianym blacie,
**86,7 s** liczenia.

Dojście do tego zajęło sześć przebiegów i wyjaśniło rzecz, która inaczej gryzłaby
użytkowników po cichu: **transformer w wersji 0.9.6 wstawiony do potoku 0.9.0
generuje sam szum.** Nie zgłasza błędu, nie przerywa się — po półtorej minuty
zapisuje poprawny plik H.264, w którym każda klatka jest kolorowym śmieciem.
Sprawdzone wykluczająco: dwie kwantyzacje (Q4_K_M i BF16), dwa typy obliczeń
(float16 i bfloat16), trzy rozdzielczości (448×256, 512×320, 704×480) — zawsze
szum. Zmiana jednej rzeczy, wersji checkpointu na zgodną z potokiem, dała obraz.

Dlatego `KNOWN_LTX_2B_REPOS` zawiera **wyłącznie** checkpointy 0.9.0, a warianty
0.9.6+ są odrzucane z komunikatem: Lightricks nie publikuje dla nich
repozytorium bazowego w formacie diffusers, więc nie ma ich z czym połączyć.

Przy okazji: modele „distilled" trenuje się do pracy **bez** classifier-free
guidance, więc `localcode/video/base.py` rozpoznaje je po nazwie (`is_distilled`)
i domyślnie podaje im `guidance = 1.0` (`DISTILLED_GUIDANCE`).

Czasu z tego pierwszego przebiegu **nie da się użyć jako pomiaru** — 1447 s
zawierało pobieranie kilku gigabajtów wag. Kolejne przebiegi, z wagami w cache,
to już właściwy pomiar: **74 s** (512×320, 25 klatek, 8 kroków) i **76 s**
(704×480, 25 klatek, 8 kroków) — czyli rzędu **minuty na sekundę materiału**
przy niskiej liczbie kroków, nie „minut”, jak zgadywała stała w kodzie.

### Czego szukanie przyczyny szumu nauczyło o samym sklepie

Trzy wady wyszły dopiero przy próbie uruchomienia i wszystkie są już naprawione:

* **instalator nie umiał pobrać innej kwantyzacji** modelu, który już był na dysku
  („już zainstalowany” kończyło sprawę) — czyli wybór jakości był nieosiągalny;
* **limit rozmiaru dla „LTX 2B” stał na 3,5 GB**, a własny plik BF16 tego modelu
  waży 3,6 GB — „najwyższa jakość” pobierała plik, którego silnik potem nie chciał
  uruchomić;
* **dokładny identyfikator repozytorium przegrywał z dopasowaniem rozmytym**:
  `city96/LTX-Video-gguf` instalowało `city96/LTX-Video-0.9.6-distilled-gguf`,
  bo każde słowo zapytania mieści się w tej dłuższej nazwie.

## Ile to naprawdę kosztuje czasu

Stała w kodzie jest wciąż **zgadnięta**, nie zmierzona:
`localcode/video/base.py` ma do tego jedną stałą,
`_SECONDS_PER_STEP_PER_FRAME_REFERENCE = 0.55` (sekundy na krok odszumiania,
na klatkę, przy 704×480) — **zgadniętą**, nie zmierzoną, opisaną w kodzie
wprost jako do podmiany, gdy ktoś w końcu przepuści to przez realny model.
Na tej podstawie kalkulator (`estimate_seconds`) liczy: stały narzut
wczytania modelu (~90 s: transformer GGUF + enkoder T5) + krok × klatki ×
stała + stały narzut dekodowania (~20 s).

Dla domyślnych ustawień (41 klatek, 30 kroków, 704×480, czyli 1,7 s
materiału) wychodzi z tego **~13 minut liczenia na 1,7 s klipu — czyli rzędu
7–8 minut liczenia na każdą sekundę materiału.** Przy limicie domyślnym
(81 klatek, 3,4 s materiału) to już ~24 minuty. To są liczby z niezmierzonej
stałej, nie pomiar — traktuj je jako "to będzie długo, licz w minutach", nie
jako obietnicę.

## Co jest zweryfikowane, a co nie

**Sprawdzone (bez wag, samym kodem i testami w `tests/video/`):**

* walidacja parametrów, zaokrąglanie do siatki 32 px / `8k+1` klatek, oba
  limity klatek (w tym że bezwzględny nigdy nie jest omijalny) —
  `test_video_base.py`;
* rozpoznawanie modeli po nazwie/repo/rozmiarze
  (`classify_video_model`) — `test_video_base.py`;
* wybór modelu, budowa ścieżki wyjściowej, mapowanie wyniku procesu
  roboczego na `VideoResult` (łącznie z przypadkami brzegowymi, gdy proces
  nie zwróci wszystkich pól), obsługa błędów (`ImportError` →
  komunikat po polsku, wyjątek ogólny → `generate_failed`) —
  `test_video_runtime.py`, na zaślepionym `runner`;
* **realny protokół JSON po stdin/stdout** (`VideoRuntime._pump`): prawdziwy
  proces potomny, prawdziwe `asyncio.create_subprocess_exec`, ale ze
  skryptem-zaślepką zamiast `worker.py` — `test_video_runtime.py`
  (`worker_cmd=`);
* walidacja argumentów i **dokładnie ten sam kod**, który w `worker.py`
  łapie `ImportError` i buduje polski komunikat o brakującym dodatku —
  wywołane bezpośrednio, z `import diffusers` wymuszonym na porażkę
  (`sys.modules["diffusers"] = None`), niezależnie od tego, czy `diffusers`
  akurat jest zainstalowany w danym środowisku — `test_video_worker.py`;
* trzy trasy HTTP (`/api/video/models|estimate|generate`), walidacja ciała
  żądania, kształt ramek SSE, autoryzacja, przerwanie połączenia przez
  klienta — `test_routes_video.py`, na zaślepionym `VideoRuntime`;
* CLI (`localcode video`): parsowanie flag, kolejność „najpierw szacunek,
  potem generowanie”, pasek postępu, komunikaty błędów — `test_cli_video.py`
  (zaślepiony `VideoRuntime`, bez tego pliku silnik nigdy się nie uruchamia).

**Sprawdzone uruchomieniem na prawdziwych wagach (29 września 2026):**

* **prawdziwa generacja** — `LTXPipeline` z transformerem GGUF wczytał się,
  odszumił i zapisał plik; API `diffusers` 0.40 (`GGUFQuantizationConfig`,
  `LTXVideoTransformer3DModel.from_single_file`, `callback_on_step_end`,
  `export_to_video`) zgadza się z tym, co robi `worker.py`;
* zależności `diffusers`/`accelerate`/`imageio[ffmpeg]` zainstalowane i użyte;
* zapis pliku, kodek H.264, liczba klatek i długość materiału — potwierdzone
  przez `ffprobe`.

**Nadal nie sprawdzone:**
* wybór urządzenia **MPS vs CPU** i awaryjne przejście na CPU po
  nieudanej operacji na MPS (`worker.py`) — kod istnieje, nigdy nie
  wykonany;
* dobór `dtype` (float16 na MPS, float32 na CPU) — nieprzetestowana decyzja,
  opisana w kodzie jako niesprawdzona;
* to, że `export_to_video` faktycznie produkuje odtwarzalny plik `.mp4`
  (zależy od `imageio-ffmpeg` w systemie);
* samo **domniemane pobieranie komponentów pipeline'u przez `diffusers`**
  (VAE/T5/scheduler z `Lightricks/LTX-Video`) z pominięciem menedżera
  pobierania LocalCode — zachowanie opisane w dokumentacji `diffusers`,
  nigdy nie zaobserwowane w tym repo;
* rzeczywisty czas i zużycie RAM na 32 GB Apple Silicon — patrz sekcja
  wyżej: liczby tam to niezmierzona stała, nie pomiar;
* to, czy model w ogóle mieści się w pamięci obok reszty LocalCode (nie ma
  tu book-keepingu budżetu RAM jak w `localcode/images/runtime.py` —
  świadomie, bo model wideo nie zostaje w pamięci między turami, patrz
  moduł `localcode/video/runtime.py`).

Krótko: kontrakt (parametry, limity, błędy, protokół, trasy HTTP, CLI) jest
przetestowany od deski do deski. To, co dzieje się wewnątrz `pipe(...)` na
prawdziwym LTX-Video, jest napisane najlepiej jak się dało z dokumentacji
`diffusers` — i czeka na pierwsze prawdziwe uruchomienie.
