# Skille — instrukcje, które agent wczytuje dopiero wtedy, gdy są potrzebne

Skill to katalog z plikiem `SKILL.md`. Agent **nie** dostaje jego treści w
prompcie — dostaje tylko nazwę i jedno zdanie opisu, a po treść sięga narzędziem
`skill` w momencie, w którym uzna, że skill pasuje. Dzięki temu skill na dziesięć
stron nie kosztuje nic, dopóki nie zostanie użyty.

Dwa miejsca:

```
~/.localcode/skills/<nazwa>/SKILL.md        # Twoje, w każdym projekcie
<projekt>/.localcode/skills/<nazwa>/SKILL.md  # tego projektu, idzie do gita
```

Skill projektowy przesłania osobisty o tej samej nazwie — repozytorium wie
lepiej niż maszyna, czego to repozytorium potrzebuje.

## Dwa warianty: mini i pełny

Model 4B z oknem 8k i model zza klucza API nie mogą dostać tej samej strony
instrukcji: to, co dla jednego jest briefingiem, dla drugiego jest jedną trzecią
okna. Dlatego skill może mieć dwa warianty:

```
<nazwa>/SKILL.md        # wariant PEŁNY (pro) — wymagany
<nazwa>/SKILL-mini.md   # wariant MINI (lite) — opcjonalny, sama treść bez nagłówka
```

Który wariant dostanie model, decyduje `localcode/agent/skills.py:variant_for`:

| model | wariant |
|---|---|
| zza klucza API (zdalny) | pełny |
| lokalny, okno > 16 384 | pełny |
| lokalny, okno ≤ 16 384 | **mini** |
| lokalny, okno nieznane (`n_ctx = 0`) | **mini** |

Nieznane okno dostaje mini celowo: dłuższy tekst trafiłby wtedy do modelu, o
którym wiemy najmniej.

**Skill bez wariantu mini nie jest odrzucany.** Mały model dostaje początek
pełnej treści, ucięty na akapicie, z dopiskiem, gdzie leży całość — może
doczytać resztę zwykłym `read_file`. Dlatego w pełnym wariancie **najważniejsze
idzie na początek**.

## Opisy też mają wariant

Opis siedzi w prompcie przy **każdym** zapytaniu, więc przy kilkudziesięciu
skillach sam spis treści zaczyna kosztować. Czterdzieści skilli po 200 znaków to
około 2300 tokenów — jedna trzecia okna 8k, zanim model cokolwiek zrobi.

Stąd pole `description-mini` w nagłówku (do 80 znaków). Gdy go nie ma, mini
powstaje przez skrócenie pełnego opisu. A gdy całe mini-menu i tak się nie
mieści, spada do samych nazw — **dlatego nazwa skilla musi mówić, co to jest**
(`uruchamianie-testow`, nie `testy2`).

## Limity

| co | limit | cel |
|---|---|---|
| `SKILL-mini.md` | 1500 znaków | ≤ 1200, ≤ 15 niepustych linii |
| `SKILL.md` | 20 000 znaków | 2500–8000 |
| `description` | 200 znaków | jedno zdanie „Użyj, gdy…” |
| `description-mini` | 80 znaków | tryb rozkazujący, np. „Uruchom testy repo” |
| nazwa | 48 znaków | małe litery, cyfry, `-` i `_`; opisowa, min. 4 znaki, bez cyfry na końcu |

Skąd 1500 znaków: w oknie 8k rezerwa na odpowiedź to 1024 tokeny, próg
kompaktowania 0,80, więc na pracę zostaje około 5700 tokenów. Prompt systemowy,
instrukcje projektu, pierwsze zadanie i menu skilli zabierają z tego około 3500.
Zostaje około 2200 tokenów, a skill może wziąć z tego piątą część: około 500
tokenów. Polskie znaki diakrytyczne kosztują po tokenie, więc liczymy 3 znaki na
token, nie 3,5 — stąd 1500 znaków.

## Szablon

### `SKILL.md` (pełny)

```markdown
---
name: uruchamianie-testow
description: Użyj, gdy trzeba uruchomić, zawęzić albo napisać testy w tym repo.
description-mini: Uruchom testy repo (pytest)
---

# Uruchamianie testów

## Kiedy użyć
Przed uznaniem zmiany za gotową i po każdej poprawce błędu.

## Polecenia

```bash
.venv/bin/pytest tests -q -m "not real_model and not network"
```

## Kroki
1. Zacznij od obszaru, którego dotyczy zmiana, potem puść całość.
2. Gdy test pada, uruchom sam ten plik z `-x` i popraw przyczynę, nie test.

## Pułapki
- Testy `-m real_model` wymagają modeli na dysku i są wolne.

## Weryfikacja
Wynik kończy się `N passed` bez `failed` i `error`.
```

Kolejność sekcji: `# Tytuł` → `## Kiedy użyć` → `## Polecenia` → `## Kroki` →
`## Pułapki` → `## Weryfikacja`. Pełne zdania są tu w porządku; uzasadnienia,
wyjątki i przykłady też.

### `SKILL-mini.md` (mini)

```markdown
# Testy

```bash
.venv/bin/pytest tests -q -m "not real_model and not network"
```

Gdy test pada → uruchom sam ten plik z `-x`, popraw przyczynę, nie test.

Nie rób: nie puszczaj `-m real_model` domyślnie — to minuty i modele z dysku.
```

Bez nagłówka `---`, bez tabel, bez sekcji. Tryb rozkazujący, krótkie zdania,
żadnych uzasadnień i żadnych „możesz też”. Układ: tytuł → blok poleceń →
do pięciu linii „Gdy X → zrób Y” → do trzech linii „Nie rób: …”.

## Zasady dla autora

1. **Oba warianty są samodzielne.** Żaden nie odsyła do drugiego. Mini to
   podzbiór pełnego, nie inna wersja: te same polecenia, te same ścieżki, te
   same flagi.
2. **Sprawdź, zanim napiszesz.** Każde polecenie uruchom, każdą ścieżkę otwórz.
   Skill, który każe uruchomić nieistniejący skrypt, jest gorszy niż jego brak —
   model zmarnuje na niego turę i straci zaufanie do całej półki.
3. **Treść po polsku, kod i komentarze w kodzie po angielsku** — jak w całym
   projekcie.
4. **Najważniejsze na początku pełnego wariantu**, bo przy braku mini to właśnie
   początek zobaczy mały model.

## Sprawdzenie przed oddaniem

```bash
.venv/bin/pytest tests/agent/test_skills_content.py -q
```

Test przechodzi po wszystkich skillach w repo i pilnuje limitów, nagłówków i
tego, że polecenia z mini występują też w pełnym wariancie.
