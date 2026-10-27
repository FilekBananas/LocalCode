# Sklep z modelami: indeks i ocena zaufania

LocalCode pokazuje tysiące modeli, których nikt tutaj nie uruchomił. Ten dokument
opisuje, skąd się biorą, co dokładnie znaczy liczba przy każdym z nich i jak
odtworzyć katalog od zera.

## Co się składa na listę

Sklep łączy trzy źródła, w tej kolejności ważności:

1. **Polecane** (`localcode/hub/curated.py`) — kilkadziesiąt modeli sprawdzonych
   ręcznie: uruchomionych, zmierzonych, z opisem napisanym pod konkretny sprzęt.
2. **Indeks** (`localcode/hub/data/catalog.json.gz`) — popularna część Hugging
   Face przefiltrowana do tego, co silniki LocalCode potrafią załadować.
   Działa bez sieci, bo jest wbudowany w pakiet.
3. **Wyszukiwanie na żywo** — to, co zwróci Hugging Face w danej chwili;
   dokłada się do listy, kiedy jest połączenie.

Do tego zawsze dochodzi wszystko, co leży na dysku — model zainstalowany spoza
katalogu też ma się pojawiać w „Moje modele".

## Jak powstaje indeks

```bash
python scripts/hub_fetch.py --out .cache/hub-raw.jsonl        # 1. fakty z API
python scripts/hub_fetch.py --extra --skip .cache/hub-raw.jsonl --out .cache/hub-extra.jsonl
#    1b. wszystkie repozytoria laboratoriów + duże modele zaufanych pakujących
python scripts/hub_fetch.py --authors                         # 1c. informacje o autorach
python scripts/build_catalog.py select                        # 2. co da się uruchomić
python scripts/build_catalog.py batches --out .cache/batches   # 3. praca dla opisów
#    (--protected: opisy dla wydań laboratoriów i dużych modeli, bez względu na pozycję)
#    (opisy pisze flota agentów; pliki lądują w .cache/enriched/)
python scripts/check_enrichment.py                            # 4. kontrola opisów
python scripts/build_catalog.py build                         # 5. catalog.json.gz
```

**Krok 1** to jedno przejście po `/api/models` dla każdej pary (format, sortowanie).
Zapisywane są tylko potrzebne pola; szablony czatu nigdy nie trafiają na dysk
(mają po kilkadziesiąt kilobajtów każdy), zamiast nich zapisujemy wyliczone z nich
flagi (narzędzia / myślenie / obrazy / effort).

**Krok 1b** istnieje, bo listy popularności ucinają dwie rzeczy, których sklep
potrzebuje: wydania laboratoriów, które nie trafiły na wykresy, i duże modele
(30B, 70B, MoE na setki miliardów parametrów) — mniej osób je pobiera, bo mniej
osób ma Maca, który je uniesie. Przegląd idzie po organizacjach z
`OFFICIAL_PUBLISHERS` i po katalogach głównych pakujących (`BIG_MODEL_PACKAGERS`
w `hub_fetch.py`). Anonimowe API Hugging Face po kilkuset listach odpowiada 429 —
skrypt wtedy czeka i ponawia (z `HF_TOKEN` limit jest dużo wyższy).

Z tego przeglądu `select` przyjmuje **tylko** wiersze chronione (`protected` w
`build_catalog.py`): wydanie laboratorium (≥ 20 pobrań/mies.) albo duży model
(≥ 24 mld parametrów lub ≥ 14 GB) od laboratorium lub w jego wiernej konwersji
(≥ 200 pobrań/mies., bez „abliterated/uncensored/merge” w nazwie). Wiersze
chronione nie wypadają przez limit 6000, a te z przeglądu dochodzą ponad niego —
reszta sklepu się nie zmienia.

**Krok 1c** zbiera informacje o autorach z Hugging Face. Dla każdego autora z
indeksu pobieramy dane z `/api/organizations/{author}/overview` albo
`/api/users/{author}/overview`: czy to organizacja, czy ma weryfikację,
liczbę followerów i plan abonamentowy. Legityność twórcy to 40% oceny zaufania, 
więc zweryfikowane organizacje i popularne konta wpływają na wynik.

**Krok 2** odrzuca to, czego nie ma jak uruchomić: modele audio i wideo, adaptery
LoRA, modele szkicujące do spekulatywnego dekodowania, repozytoria z samymi
plikami pomocniczymi (mmproj, clip, t5, vae), modele obrazowe bez enkoderów
tekstu, repozytoria oznaczone jako treści dla dorosłych. Z pozostałych zostaje
jeden wpis na (model, format) — najbardziej zaufana i najczęściej pobierana wersja
danego opakowania; liczba pozostałych opakowań ląduje w polu `packagings`.

**Krok 3 i 4** to opisy po polsku. Pisze je model językowy na podstawie faktów
z kroku 1 — i dlatego każdy opis przechodzi potem przez
`scripts/check_enrichment.py`, który odrzuca m.in. **każdą liczbę, której nie ma
w danych wejściowych** (wymyślony rozmiar w GB, wymyślona liczba parametrów),
żargon, nick autora konwersji i zmianę kategorii bez podstawy w nazwie modelu.
Odrzucone wiersze wracają do ponownego napisania; kategoria zmieniona bez podstawy
jest po prostu ignorowana przy składaniu pliku.

**Krok 5** zapisuje `catalog.json.gz` (`mtime=0`, więc identyczne dane dają
identyczny plik). Plik zawiera **wyłącznie fakty** — nie zawiera ocen. Ocena
zaufania liczona jest z tych faktów przy każdym wyświetleniu, więc regułę można
zmienić, przetestować i przedyskutować bez regenerowania czegokolwiek.

## Ocena zaufania (0–100)

`localcode/hub/trust.py`. To **nie jest skan antywirusowy** i aplikacja nigdzie
tak tego nie nazywa. Wagi ustalił właściciel projektu (6.10.2026):

| sygnał | maks. | co znaczy |
| --- | --- | --- |
| pobrania w tym miesiącu | 15 | ≥1 mln → 15, ≥100 tys. → 12, ≥10 tys. → 9, ≥1 tys. → 6, ≥100 → 3 |
| pobrania ogółem | 15 | ≥10 mln → 15, ≥1 mln → 12, ≥100 tys. → 9, ≥10 tys. → 6, ≥1 tys. → 3 (bez danych: liczone z miesięcznych) |
| serduszka | 30 | ≥1000 → 30, ≥500 → 26, ≥200 → 22, ≥100 → 18, ≥50 → 14, ≥20 → 10, ≥5 → 5, ≥1 → 2 |
| twórca | 40 | najwyższy pasujący próg — niżej |

Twórca: 40 — repozytorium samego laboratorium; 34 — wierna konwersja modelu
laboratorium (zaufany autor konwersji) albo **zweryfikowana organizacja** na
Hugging Face; 26 — zaufany autor konwersji, wagi nieznanego pochodzenia;
22 — twórca z ≥1000 obserwujących; 16 — niezależna konwersja modelu
laboratorium; 10 — twórca z ≥100 obserwujących. Dane o weryfikacji i
obserwujących pobiera `hub_fetch.py --authors`; trafiają do indeksu (pole
`authors`), więc wyniki wyszukiwania na żywo też są z nimi porównywane.

Licencja i aktualność nie dają już punktów, ale ostrzeżenia o braku licencji i
starym repozytorium nadal się wyświetlają. Model polecany przez LocalCode nie
spada poniżej 60.

**Odrzucone** (wynik 0, „odrzucony", sklep ich nie pokazuje, instalacja ze
sklepu odmawia — zainstalowane wcześniej zostają widoczne z czerwoną odznaką):

* podejrzane pliki: pickle (`.bin`, `.pt`, `.ckpt`, `.pkl`…) albo pliki
  wykonywalne i skrypty (`.exe`, `.dll`, `.so`, `.dylib`, `.sh`, `.bat`, `.ps1`,
  `.pyc`, `.jar`, `.msi`). Samo `.py` nie odrzuca — LocalCode nigdy nie
  uruchamia kodu z repozytorium — zostaje ostrzeżenie;
* twórca bez żadnego sygnału legalności (0 punktów za twórcę). Nie dotyczy
  modeli polecanych przez LocalCode.

Każdy składnik dokłada jedno zdanie po polsku, więc na pytanie „dlaczego 62?"
zawsze jest odpowiedź — widać ją w dymku przy odznace i w zakładce „Szczegóły".

## Skąd wiadomo, czy źródło jest oficjalne

`localcode/hub/publishers.py`. Kolejność rozstrzygania:

1. **`baseModels` z Hugging Face** — repozytorium konwersji wskazuje model
   źródłowy. Jeśli autor konwersji to ten sam podmiot co autor wag → `official`.
2. **Tagi `base_model:<repo>`** — to samo, kiedy pole `baseModels` jest puste.
3. **Rodzina modeli w nazwie** — `Qwen3-…` → Qwen, `gemma-…` → Google. Używane
   tylko wtedy, gdy Hugging Face nie podaje modelu źródłowego.

Cztery możliwe odpowiedzi:

* `official` — repozytorium pracowni, która wytrenowała model („źródło oficjalne");
* `trusted` — konwersja znanego zespołu (unsloth, bartowski, mlx-community,
  lmstudio-community, ggml-org…) ze znanych wag („sprawdzona konwersja");
* `community` — znane wagi, niezależny autor pliku;
* `unknown` — nie wiadomo, czyj to model.

Ważny przypadek szczególny: znany zespół konwertujący **nieznane** wagi (typowe
repozytorium `mradermacher/CzyjśMerge-13B-i1-GGUF`) dostaje mniej punktów niż ten
sam zespół konwertujący model z pracowni — konwersja jest w dobrych rękach, ale
nikt nie powie, czyj model do niej wszedł.

## Odświeżanie

Indeks starzeje się jak każda migawka — nowe modele wychodzą co tydzień. Żeby go
odświeżyć, wystarczy powtórzyć pięć kroków powyżej; krok 3–4 można pominąć, wtedy
nowe wpisy dostaną opis wyliczony z kategorii i liczby parametrów zamiast
napisanego zdania.

Testy pilnują zarówno kodu, jak i danych: `tests/hub/test_store_data.py` sprawdza
wysyłany plik (każdy wiersz ma format, kategorię i sensowną ocenę, nie ma
duplikatów, większość pierwszego tysiąca ma opis), a `tests/hub/test_trust.py`
i `tests/hub/test_publishers.py` — regułę oceny i rozpoznawanie wydawcy.
