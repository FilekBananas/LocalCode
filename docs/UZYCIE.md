# LocalCode — jak tego używać

Przewodnik po aplikacji na Maca i po terminalu, zadanie po zadaniu. To nie jest
pełna dokumentacja: instalacja, tryby uprawnień, katalog modeli i skróty
klawiszowe są opisane w [README](../README.md) — tu odsyłam, zamiast powtarzać.
Dla programistów jest osobna [dokumentacja API](API.md).

**Spis treści**

1. [Pierwsze uruchomienie](#pierwsze-uruchomienie)
2. [Pobranie modelu](#pobranie-modelu)
3. [Czat](#czat)
4. [Kod: agent w Twoim projekcie](#kod-agent-w-twoim-projekcie)
5. [Poziom myślenia](#poziom-myślenia)
6. [Ulubione modele](#ulubione-modele)
7. [Modele zdalne (klucze API)](#modele-zdalne-klucze-api)
8. [Ustawienia](#ustawienia)
9. [Obrazy, wideo, mowa](#obrazy-wideo-mowa)
10. [Terminal](#terminal)
11. [Gdy coś nie działa](#gdy-coś-nie-działa)

---

> **Gdzie jest ta dokumentacja w aplikacji?** Ikona **Pomoc** (książka) w lewym
> pasku, nad „Ustawienia”, albo menu **Help → Dokumentacja LocalCode** (⌘?). Tam też
> **Pobierz…** zapisuje oba pliki na dysku. Ta sama strona jest w **Ustawienia →
> Dokumentacja**.

## Pierwsze uruchomienie

1. Zainstaluj LocalCode i zbuduj aplikację — [README → Instalacja](../README.md#instalacja)
   (`./scripts/install.sh`), potem `localcode app` albo skopiuj
   `macos/dist/LocalCode.app` do `/Applications`.
2. Aplikacja sama uruchamia lokalny serwer (`localcode serve`) i zamyka go
   razem z oknem. Podczas startu widzisz ekran łączenia; jeśli serwer nie
   wstanie, ekran pokazuje koniec logu i pozwala wskazać program `localcode` ręcznie.
3. Na świeżej instalacji, gdy nie masz jeszcze żadnego modelu, aplikacja **sama
   pobiera jeden mały model** (domyślnie `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`,
   ok. 1 GB), żeby od razu było z kim rozmawiać. Pobieranie widać na pasku
   pobrań i można je przerwać. Nie chcesz tego — wyłącz w `settings.json`:
   `"store": {"auto_install_starter": false}`.
4. Lewy pasek ma zakładki: **Czat**, **Kod**, **Obrazy**, **Mowa**, **Wideo**,
   **Odkrywaj** (sklep z modelami), **Moje modele**, **Runtime** i **Ustawienia**.

Terminal i aplikacja dzielą ten sam serwer i te same modele w pamięci:
[README → Lokalnie czy przez serwer](../README.md#lokalnie-czy-przez-serwer).
Przy pierwszym `localcode` bez modelu terminal sam pokaże listę polecanych
i pobierze wybrany.

---

## Pobranie modelu

**Odkrywaj** to sklep: wbudowany katalog kilku tysięcy modeli działa także bez
internetu (sieć tylko dokłada nowości).

1. Wybierz kategorię (np. kodowanie, czat, rozumowanie, wizja) albo wpisz
   nazwę w polu szukania.
2. Przy modelu zobacz, czy **zmieści się** w pamięci Twojego Maca
   („Zmieści się” / „Na styk” / „Za duży”), jak jest szybki i ile waży.
3. Kliknij **Zainstaluj**. LocalCode sam dobiera wersję pliku do Twojej pamięci
   (domyślnie 4-bitową, w MLX, gdy jest). Inne wersje wybierzesz w **Szczegóły**.
4. Postęp pobierania widać na pasku pobrań; pobieranie jest wznawialne.
   Gotowy model pojawia się w **Moje modele** i w menu modeli czatu i agenta.

**Zaufanie i źródło.** Przy każdym modelu stoi liczba 0–100 i odznaka
(`oficjalne`, `konwersja`, `społeczność`, `nieznane`). To miara tego, ile
wiadomo o pochodzeniu pliku — **nie skan antywirusowy**. Nad listą są filtry
„Tylko od wydawcy” i „Tylko sprawdzone”. Szczegóły:
[README → Zaufanie i źródło](../README.md#zaufanie-i-źródło).

**Sprawdzanie Cerberem.** Jeśli masz Cerbera, LocalCode może go poprosić o
przeskanowanie modelu przed pobraniem: wklej token w **Ustawienia → Serwer i
modele → Cerber** i włącz „Sprawdzaj modele Cerberem przed pobraniem”. Gdy
sprawdzanie jest włączone, a Cerber zamknięty, pobranie nie rusza — program
prosi o jego otwarcie. Kroki: [README → Sprawdzanie modeli Cerberem](../README.md#sprawdzanie-modeli-cerberem).

Przydatne:

* **Własne repozytorium** — wklej `autor/nazwa` albo adres huggingface.co w polu
  szukania; model spoza katalogu instaluje się tak samo (bez naszego opisu).
* **„Zaawansowane”** — przegląd całego Hugging Face z wyborem kwantyzacji i pojedynczych plików.
* **Model już na dysku** — w `localcode models add ŚCIEŻKA` (plik `.gguf` albo
  katalog MLX); LM Studio i cache Hugging Face włączysz w `settings.json`
  (`scan_lmstudio`, `scan_hf_cache`).
* **Modele z ograniczeniami (gated)** — potrzebny token Hugging Face
  ([README → Token Hugging Face](../README.md#token-hugging-face)).
* **Którą kwantyzację?** [README → Którą kwantyzację wybrać](../README.md#którą-kwantyzację-wybrać).

**Moje modele** to biblioteka z gwiazdkami „Ulubione” i ustawieniami ładowania,
a **Runtime** — pamięć, załadowane instancje i limity
([README → Środowisko modeli](../README.md#środowisko-modeli-runtime)). Model ładuje się
sam przy pierwszej wiadomości; pasek nad czatem i kodem pozwala wczytać go ręcznie.

---

## Czat

Rozmowa z modelem bez dostępu do plików i powłoki. Ten sam zapis rozmów widzi
`localcode chat` w terminalu.

### Rozmowy

* **Nowa rozmowa** — przycisk u góry paska bocznego. Model wybierasz w polu
  wpisywania lub nagłówku (domyślny ustawisz w **Ustawienia → Generowanie**).
* **Enter** wysyła, **Shift+Enter** to nowa linia. Przycisk wysyłania zmienia się
  w **Zatrzymaj** — przerwana odpowiedź zostaje zapisana jako zatrzymana.
* Menu rozmowy (prawy przycisk): zmiana nazwy (własna nazwa nie jest nadpisywana
  przez automatyczny tytuł), przypięcie, **przeniesienie do projektu**,
  **eksport do Markdown albo JSON**, archiwizacja, usunięcie.
* **Szukaj w rozmowach…** przeszukuje treść wszystkich wiadomości.
* Długa rozmowa jest automatycznie skracana (starsze tury trafiają do streszczenia),
  gdy zbliża się do końca okna kontekstu —
  [README → Mały kontekst](../README.md#mały-kontekst--jak-localcode-sobie-z-nim-radzi).

### Projekty, pamięć, baza wiedzy

Szczegóły i różnice między trybami „współdzielony” i „izolowany”:
[README → Czat: projekty, pamięć, baza wiedzy](../README.md#czat-projekty-pamięć-baza-wiedzy).
W aplikacji:

* **Projekt** (sekcja „Projekty” na pasku bocznym → *Nowy projekt*) zbiera rozmowy,
  instrukcje dla modelu, pamięć i pliki. Instrukcje trzymaj krótko przy małym
  oknie kontekstu.
* **Pamięć** — notatki o Tobie i o projekcie dołączane do promptu. Globalne zarządzasz
  w **Ustawienia → Pamięć**, projektowe w edycji projektu. Zdanie
  „zapamiętaj, że…” w rozmowie dodaje wpis samo (gdy pamięć jest włączona).
* **Baza wiedzy** — pliki (txt, md, kod, csv, json, html, pdf, docx) dodajesz w
  **Ustawienia → Wiedza** (globalnie) albo w plikach projektu. Przy każdej
  wiadomości LocalCode dokleja najlepiej pasujące fragmenty i podaje przypisy do
  źródeł. Ile fragmentów — suwak **Fragmenty z plików w prompcie** w ustawieniach
  Generowania (0 wyłącza wyszukiwanie).

### Załączniki i obrazy

Spinacz w polu wpisywania (**Dołącz plik**) dodaje pliki do wiadomości; tekst z
nich jest wyciągany i trafia do modelu.

> **Obrazy w Czacie nie są wysyłane do modelu** — dostaje on tylko notatkę, że
> obraz był dołączony. Obrazy rozumie **agent w zakładce Kod** (i terminal) z
> modelem, który widzi obrazy (oznaczenie „widzi obrazy” w menu modeli, np. Qwen3‑VL).

### Rozumowanie

Przełącznik **Pokazuj rozumowanie modelu** (Ustawienia → Generowanie) pokazuje
zwijany blok z tokiem myślenia, jeśli model je zwraca. Rozumowanie nigdy nie
wraca do modelu w kolejnej turze. Poziom myślenia: [niżej](#poziom-myślenia).

---

## Kod: agent w Twoim projekcie

Ten sam agent co w terminalu: czyta pliki, edytuje je i uruchamia polecenia —
w jednym wskazanym folderze. Poza folder nie wychodzi bez pytania w żadnym trybie
([README → Tylko folder projektu](../README.md#tylko-folder-projektu)).

### Start

1. Zakładka **Kod** → **Otwórz folder…** i wskaż katalog projektu. Folder trafia
   na listę „Projekty”; prawym przyciskiem możesz go pokazać w Finderze albo usunąć z listy.
2. **Nowa sesja** zaczyna rozmowę w tym folderze. Poprzednie sesje projektu są na
   liście pod nim — kliknięcie wznawia sesję z całą historią; sesję usuniesz z jej menu.
3. Napisz, co ma zrobić: *„napraw failujący test w tests/test_api.py”*.
   **Enter** wysyła, **Shift+Enter** to nowa linia, **Esc** (lub **Zatrzymaj**)
   przerywa. Wiadomość wysłana w trakcie pracy agent dostaje przy **następnym
   kroku** (nie dopiero po całej turze) — możesz więc poprawić kurs w locie.
   Wiadomość z obrazkiem czeka na koniec tury.

Pasek nad rozmową pokazuje folder, **model**, **tryb uprawnień**, poziom myślenia,
miernik kontekstu oraz przyciski **Kompaktuj** (skraca starsze tury) i **Wyczyść**.

W polu wpisywania:

* **`/`** — polecenia (`/model`, `/mode`, `/think`, `/goal`, `/compact`, `/clear`,
  `/context`, `/resume`, `/stats`, `/init`, `/cwd`, `/help`). `/init` każe agentowi
  napisać `LOCALCODE.md` — zasady projektu, które trafiają do jego promptu.
  `/goal cel` – agent pracuje, aż cel zostanie osiągnięty i sprawdzony.
* **`@ścieżka`** — dołącza plik z projektu do wiadomości (skrócony, żeby nie zapchać
  kontekstu); `@katalog` wstawia listę plików. Zasady jak w terminalu:
  [README → Dołączanie plików](../README.md#dołączanie-plików-).
* **Obrazy** — przycisk **Dodaj obrazek**, przeciągnięcie na okno albo `⌘V`. Działa
  tylko z modelem, który widzi obrazy; inaczej pole pokaże podpowiedź, żeby go zmienić.

### Tryby uprawnień

Przełączasz je w nagłówku sesji (menu trybu) — w terminalu `Shift+Tab` lub `/mode`.
Pełna tabela: [README → Tryby uprawnień](../README.md#tryby-uprawnień).

| w aplikacji | co robi |
|---|---|
| **Pytaj** | odczyt bez pytania; zapis plików i polecenia — za Twoją zgodą |
| **Akceptuj edycje** | zmiany w plikach bez pytania, polecenia nadal za zgodą |
| **Auto** | zamiast Ciebie ocenia model-recenzent (niżej) |
| **YOLO** | wszystko bez pytania, poza poleceniami niebezpiecznymi |
| **Plan** | tylko odczyt: agent przedstawia plan, niczego nie zmienia |

**Prośba o zgodę.** Karta „Zgoda wymagana” pokazuje dokładnie, co ma się stać (różnicę
w pliku albo polecenie): **Zezwól** (`⏎`), **Zawsze** (`⌥⏎` — nie pytaj o to
ponownie w tej sesji; dla poleceń chodzi o ich prefiks, np. `git commit`) albo
**Odmów** (`Esc`).

### Tryb Auto

W trybie **Auto** zapis i edycja plików w projekcie przechodzą od razu (jak w
„Akceptuj edycje”), a każde inne działanie, o które w trybie „Pytaj” zapytałby Cię
agent — polecenia powłoki, akcje poza projektem — trafia do **krytycznego recenzenta** — modelu, który ma być podejrzliwy: zatwierdza tylko
to, co wyraźnie służy Twojemu poleceniu, zostaje w projekcie, nie niszczy danych,
niczego nie wysyła na zewnątrz i nie instaluje paczek bez prośby. W razie
wątpliwości odrzuca. Odpowiada trzema liniami: **TAK** albo **NIE**, potem **Co robi**
(jedno zdanie po ludzku) i **Powód**.

* Wynik widać pod kartą narzędzia, np. `Auto: TAK — Co robi: … Powód: …`
  (zielony znak przy zgodzie, czerwony przy odmowie).
* Gdy recenzent powie **NIE**, akcja nie przepada po cichu: dostajesz kartę zgody z
  jego opinią („Recenzent: NIE — Co robi: … Powód: …”) i sam decydujesz.
* Gdy recenzent nie da użytecznej odpowiedzi (przekroczony czas, błąd, brak modelu),
  karta zgody pokaże „Recenzent nie zdecydował” z powodem.
* Polecenia katastrofalne (`rm -rf ~`, `curl … | sh`) i przekazanie rozmowy zdalnemu
  dostawcy zawsze pytają Ciebie; recenzent i strażnik dają przy nich tylko opinię.

**Strażnik lokalny (bez kontekstu).** Najpierw sam kod LocalCode sprawdza polecenie i
szuka tylko tego, co mogłoby zaszkodzić Macowi albo wynieść z niego dane: sieci (`curl`,
`ssh`… i kod „inline” sięgający do sieci), pushu na adres wpisany w poleceniu,
prywatnych plików (`~/.ssh`, klucze, tokeny), Twoich danych poza projektem (`/tmp` się
nie liczy), kasowania poza projektem, zmian w systemie i `sudo`, instalowania paczek.
Zwykła praca w projekcie — testy, buildy, skrypty czytające logi, sprzątanie
`build/` czy `.debug/`, `git push` do własnego remote'a, `git reset` — tego nie ma, więc
strażnik w ogóle nie jest pytany i decyduje sam recenzent (pod kartą: „Strażnik lokalny:
TAK — reguły LocalCode”). Gdy coś groźnego jest, ocenia to równolegle drugi model — **zawsze lokalny** i **bez
rozmowy**: widzi tylko samo polecenie i ścieżkę projektu, więc treść plików, czatu
ani odpowiedzi modelu zdalnego nie może go do niczego namówić. Akcja wykona się sama
tylko wtedy, gdy recenzent **i** strażnik powiedzą TAK; w każdym innym przypadku
(NIE, brak odpowiedzi, brak modelu lokalnego) decydujesz Ty. Dzięki temu nawet
przejęty dostawca zdalny nie uruchomi niczego na Twoim Macu bez Twojej zgody. Żeby
rzadziej odmawiał bez potrzeby, strażnik dostaje też **Twoje polecenie** z tej tury
(sam tekst, bez dołączonych plików) i krótkie **„po co”** od agenta — ale tylko gdy
polecenie nie dotyka sieci, prywatnych plików, systemu ani kodu „inline”, i tylko gdy
opis nie próbuje go przekonywać („bezpieczne”, „odpowiedz TAK”…); inaczej ocenia samo
polecenie.
Domyślnie strażnikiem jest model agenta, jeśli jest lokalny, a przy modelu zdalnym
model pomocnika aplikacji albo najmniejszy zainstalowany model lokalny. Zmienisz to w
**Ustawienia → Agent → Strażnik lokalny (bez kontekstu)** (tylko modele lokalne).

Kolejność w trybie Auto: najpierw **reguły** (bez modelu) same przepuszczają odczyty i
polecenia tylko do odczytu (`ls`, `git status` …) i od razu oddają Tobie polecenia
katastrofalne; resztę oceniają recenzent i strażnik.

**Uczenie się z Twoich decyzji.** Gdy odpowiadasz na kartę zgody przy poleceniu
powłoki albo akcji poza folderem projektu (w każdym trybie), LocalCode zapamiętuje
Twoją decyzję dla tego projektu — tylko krótki klucz wyliczony z polecenia, np.
`git push` albo `npm run build` (przy plikach: narzędzie i folder, np.
`write_file ~/notatki`), nigdy tekst od modelu. W trybie Auto recenzent dostaje potem
pasujące wcześniejsze decyzje, a strażnik — zamiast rozmowy — krótką listę tego, na
co się zgadzasz i czego odmawiasz (najpierw pasujące, łącznie do 8 pozycji): po
Twojej zgodzie są łagodniejsi dla podobnych poleceń, po odmowie ostrzejsi. Sama
zapamiętana decyzja niczego nie uruchamia — nadal potrzebne jest TAK obu, a polecenia
katastrofalne zawsze pytają Ciebie. Decyzje leżą tylko na tym Macu
(`~/.localcode/auto_precedents.json`, osobno dla każdego projektu, do 100 na projekt),
nigdy w folderze projektu. Wyłączysz to przełącznikiem **Ustawienia → Agent → Ucz się
z moich decyzji**, a wyczyścisz przyciskiem **Wyczyść wyuczone decyzje**; w terminalu
`/forget` czyści decyzje bieżącego projektu.

**Kto jest recenzentem?** Domyślnie ten sam model co agent (korzysta wtedy z jego
pamięci podręcznej, więc jest najtańszy). Zmienisz to w **Ustawienia → Agent →
Recenzent w trybie Auto**: lista zawiera modele lokalne i zdalne.

* Inny model **lokalny** musi zmieścić się w pamięci obok modelu agenta; gdy nie da
  się go uruchomić, ocenia model agenta.
* Model **zdalny** dostaje rozmowę i każdą ocenianą akcję, a każda ocena to osobne,
  **płatne** zapytanie do dostawcy.

### Limit kroków

**Ustawienia → Agent → Maks. liczba kroków agenta** (suwak 0–200,
co 5) to liczba wywołań narzędzi na jedną wiadomość. Wartość **0 to „bez limitu”**:
agent pracuje, aż skończy albo go zatrzymasz. Domyślnie 40.

---

## Poziom myślenia

Modele z trybem rozumowania mogą „myśleć” dłużej albo krócej. W nagłówku sesji
agenta i rozmowy jest przełącznik poziomu: **Wyłączone / Niskie / Średnie / Wysokie**
(modele z prostym przełącznikiem — jak Qwen3 — mają tylko **Wyłączone / Włączone**).
Model bez żadnej kontroli myślenia ignoruje to ustawienie, a przełącznik się nie
pokazuje. Poziom startowy dla nowych rozmów i sesji ustawisz w **Ustawienia →
Serwer i modele → Myślenie**. W terminalu: `/think [off|low|medium|high]` i `--effort`.

### Ultracode

W zakładce **Kod** (i w terminalu) po poziomach myślenia jest jeszcze **Ultracode**
— najwyższy poziom agenta. Model myśli wtedy na poziomie „Wysokie”, a agent
pracuje jak koordynator zespołu pomocników: już w pierwszym kroku rozdziela
niezależne części zadania (kilka meczów albo konfiguracji do uruchomienia, kilka
logów do przeanalizowania, kilka modułów do zrozumienia) na **pomocników w tle**
i sam pracuje dalej; raport każdego pomocnika dostaje, gdy ten skończy. Na koniec
uruchamia testy i zleca pomocnikom krytyczny przegląd zmienionych plików.
Wybierasz go po nazwie (nie ma go w cyklicznym przełączaniu `/think`):
`/think ultracode`, `--effort ultracode` albo segment „Ultracode” w nagłówku sesji.

* Jest **wolniejszy i droższy**: każdy pomocnik to osobna tura modelu, a przy modelu
  zdalnym — płatne zapytania.
* Naraz pracuje do 6 pomocników (`agent.ultracodeParallel`), a w jednej turze agent
  może zlecić ich najwyżej 20. Pomocnicy dostają dwa razy więcej kroków i też myślą na „Wysokie”.
* Gdy agent przez kilka kroków z rzędu pracuje sam, dostaje krótkie przypomnienie, że może
  zlecić niezależną pracę pomocnikom (nie częściej niż co 4 kroki).
* Pomocnicy są włączani automatycznie, nawet gdy przełącznik „Agent może zlecać
  pracę pomocnikowi” jest wyłączony — ale nie na modelu z oknem kontekstu poniżej
  16 384 tokenów (wtedy Ultracode działa bez nich).
* To poziom agenta: w Czacie jest niedostępny (terminal w trybie czatu go odrzuca).

### Pomocnicy (w tle)

Pomocnik to osobny agent z własnym, czystym kontekstem, któremu agent zleca jedno
zadanie (narzędzie `delegate`). Pomocnik **czyta pliki i uruchamia polecenia**, ale
nie edytuje plików i nie zleca pracy dalej. Każde jego polecenie przechodzi przez te
same zasady co polecenia agenta: w trybie Auto ocenia je recenzent i lokalny strażnik,
w trybie Domyślnym pytasz Ty, w trybie planu pomocnik w ogóle nie ma poleceń. Karta
zgody mówi, kto pyta: „Pomocnik h2: …”.

* **W tle** (domyślnie w Ultracode): wywołanie wraca od razu („Pomocnik h2 pracuje w
  tle”), agent pracuje dalej, a gdy pomocnik skończy, jego raport trafia do agenta przed
  następnym krokiem. Karta pomocnika ma znacznik „w tle”, a w nagłówku sesji widać
  „Pomocnicy w tle: N”.
* Jeśli agent skończy odpowiedź, a pomocnicy jeszcze pracują, tura czeka na nich i
  idzie dalej z ich raportami — nic nie przepada. Stop przerywa też wszystkich pomocników.
* Poza Ultracode agent zwykle czeka na raport (może jednak poprosić o pracę w tle).


---

## Ulubione modele

Gwiazdka oznacza model jako ulubiony. Ulubione są **na początku każdej listy
modeli** — w menu czatu i agenta (sekcja „Ulubione”), w pasku ładowania modeli i
w **Moje modele** — razem lokalne i zdalne, w kolejności dodania.

* Gwiazdkę klikasz przy modelu w **Moje modele** i w pasku ładowania; w menu modeli
  czatu i agenta służy do tego pozycja **Dodaj do ulubionych / Usuń z ulubionych**
  (dla modelu, którego używasz).
* Terminal pokazuje te same ulubione na początku listy `/model`; dodajesz je w
  aplikacji albo wpisem `favorite_models` w `~/.localcode/settings.json`.

---

## Modele zdalne (klucze API)

LocalCode jest lokalny, ale możesz **pożyczyć** model od dostawcy, jeśli wkleisz jego
klucz. Dostępni: **OpenAI, Anthropic, Google Gemini, DeepInfra, Unlid**.

1. **Ustawienia → Serwer i modele → Modele zdalne.**
2. Przy dostawcy wklej **Klucz API** i kliknij **Sprawdź i zapisz**. Klucz jest
   weryfikowany u dostawcy **przed** zapisem; link „Skąd wziąć klucz?” prowadzi do
   jego panelu. Klucza nigdy nie odczytasz z powrotem — widać tylko „Klucz zapisany”.
3. Modele dostawcy pojawiają się na listach obok lokalnych, z ikoną chmury i nazwą
   dostawcy w nawiasie (np. „Llama 3.3 70B (DeepInfra)”), pod kluczem
   `remote:<dostawca>:<model>`. Przełącznik przy dostawcy wyłącza go bez kasowania klucza.

Co warto wiedzieć:

* **Prywatność.** Wybierając model zdalny, wysyłasz do dostawcy rozmowę, a w Kodzie
  także fragmenty plików, które agent czyta. Dopóki go nie wybierzesz, nic nie
  wychodzi z Maca. Zdalne modele działają w Czacie i w Kodzie, ale **nie** przez
  lokalne API zgodne z OpenAI (`/v1`).
* **Koszty.** Zapytania rozlicza dostawca, nie LocalCode — licznik kosztu w
  aplikacji zostaje na zero. Agent robi wiele kroków na jedną wiadomość, więc kontrola
  kosztów leży po stronie dostawcy (limity, panel rozliczeń). Pomocnicy agenta i recenzent
  Auto, jeśli wskażesz dla nich model zdalny, też są płatnymi zapytaniami.
* **Cache.** W Anthropic LocalCode oznacza prompt systemowy, narzędzia i ostatnią
  wiadomość do cachowania, żeby kolejne kroki agenta nie płaciły pełnej ceny za całą
  rozmowę od nowa; OpenAI cachuje automatycznie. Ile promptu przyszło z cache,
  zobaczysz w statystykach tury w terminalu (`z cache`, polecenie `/stats`).
* **Limity dostawcy** (429/503) LocalCode ponawia po kilku sekundach zamiast
  przerywać turę.
* **Tryb: Standard / Flex / Priority** (DeepInfra, Unlid, OpenAI) — przełącznik „Tryb”
  przy dostawcy z zapisanym kluczem. **Standard** to domyślne zachowanie. **Flex** jest
  tańszy (DeepInfra: ok. 0,8× ceny), ale gdy brakuje mocy, zapytanie czeka w kolejce
  dostawcy do 10 minut — LocalCode tyle czeka, a jeśli dostawca w końcu odmówi (429,
  bez opłaty), pokazuje komunikat zamiast ponawiać. **Priority** odpowiada szybciej przy
  dużym obciążeniu, ale drożej (DeepInfra: ok. 1,5× ceny). Nie każdy model obsługuje
  tryby — jeśli odmówi, LocalCode powie, żeby przełączyć na Standard. Unlid przekazuje
  ustawienie do dostawcy modelu; czy zadziała, zależy od modelu. Anthropic i Gemini
  trybów nie mają.
* Dostęp programistyczny: [API.md → Modele zdalne](API.md#modele-zdalne-routes_remotepy).

---

## Ustawienia

Strony: **Serwer i modele**, **Agent**, **O mnie**, **Prompt systemowy**, **Generowanie**,
**Pamięć**, **Wiedza**, **Dokumentacja**.

### Generowanie

* **Domyślny model** — dla nowych rozmów.
* **Temperatura** — niżej: odpowiedzi powtarzalne; wyżej: twórcze (dla małych modeli 0,3–0,7).
* **Maksymalna długość odpowiedzi** — suwak od 256 tokenów; **skrajne prawe położenie
  to „bez limitu”**: model pisze, aż skończy albo zabraknie miejsca w oknie kontekstu.
  Aplikacja ostrzega, gdy limit zajmuje ponad połowę okna.
* **Fragmenty z plików w prompcie** (0 = wyłączone) i **Limit historii wysyłanej do
  modelu** (0 = bez limitu).
* **Automatyczne tytuły rozmów** i **Pokazuj rozumowanie modelu**.

### Agent

**Maks. liczba kroków agenta** (0 = bez limitu), **Recenzent w trybie Auto**,
**Strażnik lokalny (bez kontekstu)**, **Ucz się z moich decyzji** ([wyżej](#tryb-auto)).

### Serwer i modele

* **Serwer** — informacja o lokalnym procesie `localcode serve` i możliwość wskazania
  innego programu `localcode`.
* **Modele zdalne** — klucze API ([wyżej](#modele-zdalne-klucze-api)).
* **Myślenie** — domyślny poziom dla Czatu i Agenta oraz dwa przełączniki:
  *Asystent zarządza modelami* (pozwala poprosić „pobierz mi dobry model do
  kodowania”; w Kodzie każde pobranie pyta o zgodę) i *Agent może zlecać pracę
  pomocnikowi* (szukanie, przebiegi, analiza logów w osobnym kontekście, także w tle —
  [Pomocnicy](#pomocnicy-w-tle)).
* **Cerber** — sprawdzanie modeli przed pobraniem.

Gdzie to leży w plikach (`~/.localcode/settings.json`, sesje, baza czatu):
[README → Gdzie LocalCode trzyma dane](../README.md#gdzie-localcode-trzyma-dane).

### Serwer dla innych aplikacji

**Runtime → Serwer zgodny z OpenAI** pokazuje adres do skopiowania i przełącznik
„Włączony”. Jak podłączyć do niego własne narzędzia i skrypty — w
[API.md](API.md#localcode-jako-backend-innych-narzędzi). Uwaga: aplikacja uruchamia
serwer na losowym porcie; jeśli potrzebujesz stałego adresu, uruchom przed nią
`localcode serve` (domyślnie port 8421), a aplikacja do niego dołączy.

---

## Obrazy, wideo, mowa

Te zakładki działają na osobnych modelach i wymagają dodatków Pythona
(`localcode[images]`, `localcode[video]`, `localcode[speech]`). Gdy dodatku brakuje,
program wyjaśnia to po polsku i podaje, co doinstalować.

| zakładka | co robi | szczegóły |
|---|---|---|
| **Obrazy** | obraz z opisu (FLUX, Qwen‑Image, Stable Diffusion…), galeria z ulubionymi | [IMAGES.md](IMAGES.md), [README → Generowanie obrazów](../README.md#generowanie-obrazów) |
| **Wideo** | krótki klip z opisu (LTX‑Video 2B); to **minuty liczenia na sekundę materiału**, więc zakładka najpierw szacuje czas | [VIDEO.md](VIDEO.md) |
| **Mowa** | dyktowanie (nagranie → tekst) i czytanie na głos | [SPEECH.md](SPEECH.md) |

---

## Terminal

Wszystko z aplikacji da się też zrobić w terminalu:

```bash
cd ~/projekty/moj-projekt && localcode        # agent w katalogu
localcode -c                                  # wznów ostatnią sesję
localcode -p "co robi ten projekt?"           # jedna odpowiedź na stdout (skrypty, CI)
localcode -m qwen3-4b --mode auto --effort high
localcode chat                                # rozmowa bez narzędzi
localcode models install "Qwen3 Coder 30B"    # instalacja po nazwie
localcode serve                               # serwer HTTP + API zgodne z OpenAI
```

Tryb `auto` w terminalu pokazuje decyzję recenzenta jako `✓ Auto: TAK — …` lub
`✗ Auto: NIE — …`. W trybie bezinterfejsowym (`-p`) nikt nie odpowiada na pytania o
zgodę, więc użyj `--mode accept_edits`, `auto` albo `yolo`, jeśli agent ma coś zmieniać.
Pełny opis: [README → Aplikacja terminalowa](../README.md#aplikacja-terminalowa),
[Skróty klawiszowe](../README.md#skróty-klawiszowe) i [Ściąga: polecenia](../README.md#ściąga-polecenia).

---

## Gdy coś nie działa

Najczęstsze sytuacje i pełna lista: [README → Rozwiązywanie problemów](../README.md#rozwiązywanie-problemów).
Na początek:

* `localcode doctor` sprawdza instalację i podaje polecenie naprawy przy każdym problemie.
* **Aplikacja nie łączy się z serwerem** — ekran startowy pokazuje koniec logu;
  całość to `~/.localcode/logs/server.log`.
* **Terminal nie widzi modeli z aplikacji** — `~/.localcode/server.json` musi wskazywać
  działający serwer; przy nieaktualnym pliku usuń go albo użyj `localcode --attach`.
* **Model nie mieści się w pamięci** — mniejsza kwantyzacja, mniejszy kontekst albo
  zwolnienie innych modeli w **Runtime**.
* **Model odpowiada wolno albo kontekst jest pełny** — w Kodzie **Kompaktuj** lub
  **Wyczyść**.
* **Czat nie widzi obrazu** — w Czacie obrazy nie trafiają do modelu; użyj zakładki Kod
  z modelem „widzi obrazy”.
* **Agent „utknął”** — czeka na zgodę (karta „Zgoda wymagana”) albo trwa kolejka;
  **Zatrzymaj** przerywa turę.
* **Pomocnik** w aplikacji odpowiada na pytania o samą aplikację
  ([HELPBOT.md](HELPBOT.md)).
