#!/bin/bash
# LocalCode — instalator dla macOS (Apple Silicon).
#
#   ./scripts/install.sh              # wszystko: venv + zależności + llama.cpp + komendy + aplikacja
#   ./scripts/install.sh --no-app     # bez budowania aplikacji na Maca
#   ./scripts/install.sh --dev        # używa .venv w repo zamiast ~/.localcode/venv
#   ./scripts/install.sh --app-only   # tylko przebuduj aplikację
#   ./scripts/install.sh --no-images  # bez dodatku do obrazów
#   ./scripts/install.sh --no-speech  # bez dodatku do mowy
#   ./scripts/install.sh --no-video   # bez dodatku do wideo
#   ./scripts/install.sh --no-helpbot # bez pobierania modelu pomocnika (ok. 2,3 GB)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOME_DIR="${LOCALCODE_HOME:-$HOME/.localcode}"
VENV="$HOME_DIR/venv"
BIN_DIR="$HOME/.local/bin"
BUILD_APP=1
APP_ONLY=0
DEV=0
NO_IMAGES=0
NO_SPEECH=0
NO_VIDEO=0
NO_HELPBOT=0

for arg in "$@"; do
  case "$arg" in
    --no-app) BUILD_APP=0 ;;
    --no-images) NO_IMAGES=1 ;;
    --no-speech) NO_SPEECH=1 ;;
    --no-video) NO_VIDEO=1 ;;
    --no-helpbot) NO_HELPBOT=1 ;;
    --app-only) APP_ONLY=1 ;;
    --dev) DEV=1; VENV="$REPO/.venv" ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "Nieznana opcja: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31mBŁĄD:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "LocalCode działa tylko na macOS."
[ "$(uname -m)" = "arm64" ] || warn "Wykryto procesor Intel — MLX nie zadziała, zostanie tylko GGUF."

# --------------------------------------------------------------------------- #
# 1. Python + środowisko wirtualne
# --------------------------------------------------------------------------- #
if [ "$APP_ONLY" = 0 ]; then
  PY=""
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      ver=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
      case "$ver" in 3.11|3.12|3.13) PY="$candidate"; break ;; esac
    fi
  done
  [ -n "$PY" ] || die "Potrzebny Python 3.11–3.13. Zainstaluj: brew install python@3.12"
  say "Python: $($PY --version) ($PY)"

  if [ ! -x "$VENV/bin/python" ]; then
    say "Tworzę środowisko: $VENV"
    mkdir -p "$(dirname "$VENV")"
    "$PY" -m venv "$VENV"
  fi
  "$VENV/bin/pip" install -q --upgrade pip
  say "Instaluję LocalCode i zależności (to potrwa chwilę)…"
  # Poza trybem --dev instalujemy KOPIĘ, nie dowiązanie do repozytorium.
  # Powód jest praktyczny: jeśli repo leży w ~/Documents, to aplikacja z
  # /Applications dostaje od macOS blokadę na ten folder (pyvenv.cfg i kod
  # pakietu są wtedy w Dokumentach) i serwer zawiesza się przy starcie Pythona,
  # czekając na zgodę, której nikt nie widzi. Kopia w ~/.localcode/venv tego
  # problemu nie ma.
  EDITABLE=""
  [ "$DEV" = 1 ] && EDITABLE="-e"
  if [ "$(uname -m)" = "arm64" ]; then
    "$VENV/bin/pip" install -q $EDITABLE "$REPO[mlx]"
  else
    "$VENV/bin/pip" install -q $EDITABLE "$REPO"
  fi

  # ----------------------------------------------------------------------- #
  # 1b. Dodatki: obrazy, mowa, wideo
  # ----------------------------------------------------------------------- #
  # Każdy pakiet instalujemy OSOBNO, a nie jako jedno `pip install
  # "$REPO[images]"`. W jednym poleceniu pip działa na zasadzie
  # wszystko-albo-nic: stable-diffusion-cpp-python kompiluje się ze źródeł i
  # potrafi nie wejść (brak narzędzi wiersza poleceń, brak miejsca na dysku) — a
  # wtedy zabiera ze sobą mflux, przez który działają modele MLX. Użytkownik
  # widzi w aplikacji „brakuje pakietu mflux”, choć nie udało się zupełnie co
  # innego. Osobne polecenia zostawiają to, co weszło, i mówią, czego brakuje.
  #
  # Lista pakietów nie jest tu przepisana z pyproject.toml — czytamy ją z
  # metadanych zainstalowanego LocalCode, więc nie rozjedzie się przy kolejnej
  # zmianie w [project.optional-dependencies].
  extra_packages() {
    "$VENV/bin/python" - "$1" 2>/dev/null <<'LCPY'
import sys
from importlib.metadata import metadata
from packaging.requirements import Requirement

extra = sys.argv[1]
for raw in metadata("localcode").get_all("Requires-Dist") or []:
    req = Requirement(raw)
    if req.marker is not None and req.marker.evaluate({"extra": extra}):
        names = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
        print(f"{req.name}{names}{req.specifier}")
LCPY
  }

  install_extra() {  # install_extra <nazwa-dodatku> <opis dla człowieka>
    local extra="$1" label="$2" pkgs failed="" retry=""
    pkgs="$(extra_packages "$extra" || true)"
    say "Instaluję dodatek: ${label}…"
    if [ -z "$pkgs" ]; then
      # Nie udało się odczytać metadanych — wracamy do starego sposobu.
      "$VENV/bin/pip" install -q $EDITABLE "$REPO[$extra]" \
        || warn "Nie udało się zainstalować dodatku „${label}” — reszta LocalCode działa normalnie."
      return 0
    fi
    while IFS= read -r pkg; do
      [ -n "$pkg" ] || continue
      if ! "$VENV/bin/pip" install -q "$pkg"; then
        failed="$failed $pkg"
        retry="$retry \"$pkg\""
      fi
    done <<LCPKGS
$pkgs
LCPKGS
    if [ -n "$failed" ]; then
      warn "Dodatek „${label}” wszedł tylko częściowo — brakuje:$failed"
      warn "Reszta LocalCode działa normalnie. Spróbuj później: $VENV/bin/pip install$retry"
    fi
  }

  if [ "$(uname -m)" = "arm64" ]; then
    # Obrazy (docs/IMAGES.md): mflux dla modeli MLX, stable-diffusion-cpp-python
    # dla GGUF — ten drugi się kompiluje, więc trwa kilka minut.
    if [ "$NO_IMAGES" = 0 ]; then
      install_extra images "generowanie obrazów (kompilacja, kilka minut)"
    fi
    # Mowa (docs/SPEECH.md): dyktowanie i czytanie na głos.
    if [ "$NO_SPEECH" = 0 ]; then
      install_extra speech "dyktowanie i czytanie na głos"
    fi
    # Wideo (docs/VIDEO.md): diffusers na PyTorchu — same pliki to ok. 1 GB.
    if [ "$NO_VIDEO" = 0 ]; then
      install_extra video "generowanie wideo (ok. 1 GB plików)"
    fi
  fi

  # ----------------------------------------------------------------------- #
  # 1c. Pomocnik — mały model, który za użytkownika ustawia resztę aplikacji
  # ----------------------------------------------------------------------- #
  # "Przychodzi z aplikacją": pobieramy go od razu, żeby był gotowy, zanim
  # ktokolwiek o niego poprosi. `localcode pomoc --install` sam sprawdza miejsce
  # na dysku i odmawia zamiast zapełnić dysk; porażka nie przerywa instalacji —
  # serwer spróbuje ponownie przy starcie, a aplikacja ma przycisk "Pobierz".
  if [ "$NO_HELPBOT" = 0 ] && [ "$(uname -m)" = "arm64" ]; then
    say "Pobieram pomocnika (ok. 2,3 GB, jednorazowo)…"
    "$VENV/bin/localcode" pomoc --install \
      || warn "Nie udało się pobrać pomocnika — aplikacja zaproponuje to sama. Ręcznie: localcode pomoc --install"
  fi

  # ----------------------------------------------------------------------- #
  # 2. llama.cpp (backend GGUF)
  # ----------------------------------------------------------------------- #
  if ! command -v llama-server >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/llama-server ]; then
    if command -v brew >/dev/null 2>&1; then
      say "Instaluję llama.cpp (llama-server) przez Homebrew…"
      brew install llama.cpp || warn "Nie udało się zainstalować llama.cpp — modele GGUF nie będą działać."
    else
      warn "Brak Homebrew — zainstaluj llama.cpp ręcznie, inaczej modele GGUF nie będą działać."
    fi
  else
    say "llama-server: $(command -v llama-server || echo /opt/homebrew/bin/llama-server)"
  fi

  # ----------------------------------------------------------------------- #
  # 3. Komendy `localcode` i `lc`
  # ----------------------------------------------------------------------- #
  mkdir -p "$BIN_DIR"
  ln -sf "$VENV/bin/localcode" "$BIN_DIR/localcode"
  ln -sf "$VENV/bin/localcode" "$BIN_DIR/lc"
  say "Komendy: $BIN_DIR/localcode, $BIN_DIR/lc"
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "Dodaj do PATH:  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.zshrc" ;;
  esac

  mkdir -p "$HOME_DIR/models" "$HOME_DIR/data" "$HOME_DIR/sessions" "$HOME_DIR/logs"
fi

# --------------------------------------------------------------------------- #
# 4. Aplikacja na Maca
# --------------------------------------------------------------------------- #
if [ "$BUILD_APP" = 1 ] || [ "$APP_ONLY" = 1 ]; then
  if [ -f "$REPO/macos/scripts/build-app.sh" ]; then
    if command -v swift >/dev/null 2>&1; then
      say "Buduję aplikację LocalCode.app (pierwszy raz trwa kilka minut)…"
      LC_SERVER_PATH="$VENV/bin/localcode" bash "$REPO/macos/scripts/build-app.sh"
      if [ -d "$REPO/macos/dist/LocalCode.app" ]; then
        say "Aplikacja: $REPO/macos/dist/LocalCode.app"
        say "Skopiuj do /Applications:  cp -R '$REPO/macos/dist/LocalCode.app' /Applications/"
      fi
    else
      warn "Brak kompilatora Swift (zainstaluj Xcode) — pomijam aplikację na Maca."
    fi
  else
    warn "Brak macos/scripts/build-app.sh — pomijam aplikację na Maca."
  fi
fi

# --------------------------------------------------------------------------- #
# 5. Podsumowanie
# --------------------------------------------------------------------------- #
if [ "$APP_ONLY" = 0 ]; then
  say "Sprawdzam instalację…"
  "$VENV/bin/localcode" doctor || warn "localcode doctor zgłosił problemy (patrz wyżej)."
  cat <<EOF

Gotowe. Co dalej:

  localcode                      agent kodujący w bieżącym katalogu
  localcode models curated       polecane modele do pobrania
  localcode models download <repo> [kwantyzacja]
  localcode chat                 czat w terminalu
  localcode serve                serwer (API zgodne z OpenAI na /v1)
  localcode doctor               diagnostyka

EOF
fi
