# LocalCode — instrukcje dla agenta pracującego w tym repo

Projekt: lokalna aplikacja w stylu LM Studio dla Apple Silicon — agent kodujący (jak Claude Code)
w terminalu i w natywnej aplikacji na Maca, plus chatbot. Modele: GGUF (llama-server) i MLX (mlx-lm),
pobierane z Hugging Face. Modele mają mały kontekst, więc oszczędność kontekstu jest funkcją, nie detalem.

## Zasady

- Kontrakt architektury: `docs/ARCHITECTURE.md`. Pliki współdzielone (`localcode/config.py`,
  `localcode/hub/types.py`, `localcode/runtime/base.py`, `localcode/events.py`,
  `localcode/agent/tools/base.py`) zmieniamy tylko świadomie — zależy od nich cała reszta.
- Python: `.venv/bin/python`, testy `.venv/bin/pytest tests -q -m "not real_model and not network"`.
  Testy z prawdziwymi modelami: `-m real_model` (modele w `~/.localcode/models`, helper `lc_testing.py`).
- Teksty dla użytkownika po polsku; kod, komentarze i prompty dla modelu po angielsku.
- Procesy silników chodzą w piaskownicy macOS (`localcode/security/sandbox.py`): bez internetu,
  bez dostępu do `~/Documents`, `~/.ssh`, Keychaina. Silnik deklaruje, co mu wolno, przez
  `_sandbox_plan()`. Dwie pułapki, obie już kosztowały dzień: (1) katalog cache Metala musi być
  zapisywalny, inaczej llama.cpp kompiluje szadery przy każdym starcie (0,8 s → 30 s), a
  `os.confstr` **nie zna** `CS_DARWIN_USER_CACHE_DIR` na macOS — bierzemy to z `getconf`;
  (2) ścieżka do czytania potrzebuje też metadanych katalogów nadrzędnych, inaczej `realpath`
  pada na „Operation not permitted”. Tryb w `settings.security.sandbox` (`auto`/`strict`/`off`).
- Pomocnik (`localcode/helpbot/`, `docs/HELPBOT.md`) to pętla agenta z innym promptem i narzędziami;
  sesja jest pomocnikiem, bo pracuje w `<LOCALCODE_HOME>/helpbot`. Wszystko, co zmienia, pyta;
  klucze API nigdy nie wchodzą do rozmowy.
- Jedna tabela komend ukośnikowych na oba interfejsy: `localcode/tui/commands.py` (pole `in_app`),
  aplikacja czyta ją z `GET /api/agent/commands`. Drugi egzemplarz listy to dokładnie mechanizm,
  który rozjechał aplikację do 4 komend wobec 23 w terminalu — nie rób go ponownie.
- Skille mają dwa warianty (`docs/SKILLS.md`): `SKILL.md` i opcjonalny `SKILL-mini.md`. Treść
  pisana przez model **musi** przejść `tests/agent/test_skills_content.py` — walidator sprawdza
  m.in., czy skill nie odsyła do nieistniejącej ścieżki. Jeden już to robił.
- Nigdy nie zostawiaj osieroconych procesów silników (`llama-server`, `mlx_worker`) — testy sprzątają w `finally`.
- Układ ekranu sprawdzasz bez klikania: `cd macos && swift run ScreenShots <katalog>` renderuje
  sekcje do plików PNG. Rysuje przez **okno poza ekranem** (`NSHostingView` w `NSWindow` +
  `cacheDisplay`), nie przez `ImageRenderer` — i to jest cała różnica: `ImageRenderer` nie
  układa `ScrollView` i zamienia kontrolki AppKit (`TextEditor`, `Picker`, `Stepper`) w żółte
  zastępniki, bo nie mają okna, w którym mogłyby żyć. Okno poza ekranem daje im je, niczego nie
  pokazuje na ekranie i nie potrzebuje żadnych uprawnień. Stan do renderowania dają fabryki
  `preview(api:)` w magazynach, a widoki mają wariant bez chrome (`VideoView(showsChrome:)`,
  `SpeechScreenPreview`) — wszystko tylko w kompilacji debug.
- Aplikacja na Maca: `macos/` (SwiftPM), budowanie `macos/scripts/build-app.sh`, styl 1:1 z referencją
  `the reference Apple client`.
- Sklep z modelami: `localcode/hub/data/catalog.json.gz` (indeks tysięcy modeli) jest **generowany**
  — nie edytuj go ręcznie, procedura w `docs/CATALOG.md`. Plik zawiera wyłącznie fakty; ocena
  zaufania liczy się z nich przy wyświetlaniu (`localcode/hub/trust.py`), więc regułę zmienia się
  w kodzie i w testach, bez regeneracji danych.
- Ikona aplikacji: `macos/Resources/AppIcon.png` (własny plik wygrywa; brak → rysuje ją
  `macos/scripts/make_icon.py`). Skrypt budowania sam robi `.icns`.
- Aplikacja z `/Applications` **nie ma dostępu do `~/Documents`** (TCC). Jej `LCServerPath` musi
  wskazywać na `~/.localcode/venv/bin/localcode` (kopia, nie `-e`), inaczej serwer zawiesza się
  w `open()` na `pyvenv.cfg` i nigdy nie startuje. `build-app.sh` i `install.sh` już tak robią;
  `.venv` w repo zostaje do pracy z `macos/dist/`.
