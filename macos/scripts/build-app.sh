#!/bin/bash
#
# Builds LocalCode.app from the SwiftPM package.
#
# There is no Xcode project on purpose: `swift build` produces a plain
# executable, and this script wraps it in the bundle macOS needs for a Dock
# icon, a menu bar and an Info.plist the app can read at runtime (that is where
# LCServerPath lives — the fallback location of the Python server).
#
#   ./scripts/build-app.sh            release build into dist/LocalCode.app
#   ./scripts/build-app.sh --debug    the same, from the debug build
#
# The bundle is signed ad hoc (`-s -`) and is NOT sandboxed: LocalCode spawns a
# Python process, reads model files anywhere on disk and opens a local port.

set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PACKAGE_DIR}/.." && pwd)"
CONFIGURATION="release"
APP_NAME="LocalCode"
BUNDLE_ID="com.localcode.app"
VERSION="0.1.0"
BUILD="1"

# Where the app looks for the Python server when nothing else is configured.
# Preference: what install.sh exported, then the shared environment in
# ~/.localcode (a copy in /Applications keeps working when the checkout moves),
# then this repository's .venv for development builds.
if [[ -n "${LC_SERVER_PATH:-}" ]]; then
    DEFAULT_SERVER_PATH="${LC_SERVER_PATH}"
elif [[ -x "${HOME}/.localcode/venv/bin/localcode" ]]; then
    DEFAULT_SERVER_PATH="${HOME}/.localcode/venv/bin/localcode"
else
    DEFAULT_SERVER_PATH="${REPO_DIR}/.venv/bin/localcode"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) CONFIGURATION="debug"; shift ;;
        --server-path) DEFAULT_SERVER_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Nieznany argument: $1" >&2; exit 2 ;;
    esac
done

# The in-app documentation (Help → Dokumentacja LocalCode, Settings →
# Dokumentacja) ships inside the bundle. A build without it would show an
# empty viewer, so a missing file stops the build before anything is compiled.
DOC_FILES=(UZYCIE.md API.md)
for doc in "${DOC_FILES[@]}"; do
    if [[ ! -f "${REPO_DIR}/docs/${doc}" ]]; then
        echo "BŁĄD: brak pliku dokumentacji ${REPO_DIR}/docs/${doc} — aplikacja nie zostanie zbudowana bez niego." >&2
        exit 1
    fi
done

cd "${PACKAGE_DIR}"

echo "==> swift build -c ${CONFIGURATION}"
swift build -c "${CONFIGURATION}"

BIN_DIR="$(swift build -c "${CONFIGURATION}" --show-bin-path)"
EXECUTABLE="${BIN_DIR}/${APP_NAME}"

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Nie znaleziono pliku wykonywalnego: ${EXECUTABLE}" >&2
    exit 1
fi

DIST_DIR="${PACKAGE_DIR}/dist"
APP_DIR="${DIST_DIR}/${APP_NAME}.app"
CONTENTS="${APP_DIR}/Contents"

echo "==> assembling ${APP_DIR}"
rm -rf "${APP_DIR}"
mkdir -p "${CONTENTS}/MacOS" "${CONTENTS}/Resources"

cp "${EXECUTABLE}" "${CONTENTS}/MacOS/${APP_NAME}"
chmod +x "${CONTENTS}/MacOS/${APP_NAME}"

# Swift runtime libraries the package may have bundled next to the binary.
for bundle in "${BIN_DIR}"/*.bundle; do
    [[ -e "${bundle}" ]] || continue
    cp -R "${bundle}" "${CONTENTS}/Resources/"
done

# Documentation: read by the app through Bundle.main (subdirectory "Docs").
echo "==> docs (${DOC_FILES[*]})"
mkdir -p "${CONTENTS}/Resources/Docs"
for doc in "${DOC_FILES[@]}"; do
    cp "${REPO_DIR}/docs/${doc}" "${CONTENTS}/Resources/Docs/${doc}"
done

# --------------------------------------------------------------------------- #
# Icon: macos/Resources/AppIcon.png (drop your own there), or the generated
# placeholder.  sips + iconutil are part of macOS, so there is nothing to install.
# --------------------------------------------------------------------------- #
ICON_PNG="${PACKAGE_DIR}/Resources/AppIcon.png"
ICON_NAME=""
if [[ ! -f "${ICON_PNG}" ]]; then
    echo "==> rysuję zastępczą ikonę (macos/Resources/AppIcon.png)"
    python3 "${PACKAGE_DIR}/scripts/make_icon.py" "${ICON_PNG}" >/dev/null \
        || echo "!!  nie udało się narysować ikony — aplikacja zostanie bez niej" >&2
fi
if [[ -f "${ICON_PNG}" ]]; then
    echo "==> icon"
    ICONSET="$(mktemp -d)/AppIcon.iconset"
    mkdir -p "${ICONSET}"
    for spec in "16:16x16" "32:16x16@2x" "32:32x32" "64:32x32@2x" "128:128x128" \
                "256:128x128@2x" "256:256x256" "512:256x256@2x" "512:512x512" "1024:512x512@2x"; do
        px="${spec%%:*}"
        label="${spec##*:}"
        sips -z "${px}" "${px}" "${ICON_PNG}" --out "${ICONSET}/icon_${label}.png" >/dev/null 2>&1
    done
    if iconutil -c icns "${ICONSET}" -o "${CONTENTS}/Resources/AppIcon.icns" 2>/dev/null; then
        ICON_NAME="AppIcon"
    else
        echo "!!  iconutil nie zbudował .icns — aplikacja zostanie bez ikony" >&2
    fi
    rm -rf "$(dirname "${ICONSET}")"
fi

cat > "${CONTENTS}/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${BUILD}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticTermination</key>
    <false/>
    <key>NSSupportsSuddenTermination</key>
    <false/>
    <key>NSHumanReadableCopyright</key>
    <string>LocalCode</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>LocalCode potrzebuje mikrofonu do nagrywania dyktowanego tekstu w sekcji „Mowa”. Nagranie zostaje na tym komputerze.</string>
    <key>LCServerPath</key>
    <string>${DEFAULT_SERVER_PATH}</string>${ICON_NAME:+
    <key>CFBundleIconFile</key>
    <string>${ICON_NAME}</string>
    <key>CFBundleIconName</key>
    <string>${ICON_NAME}</string>}
</dict>
</plist>
PLIST

cat > "${CONTENTS}/PkgInfo" <<'PKGINFO'
APPL????
PKGINFO

echo "==> codesign (ad hoc)"
codesign --force --deep --sign - "${APP_DIR}"
codesign --verify --verbose=1 "${APP_DIR}" >/dev/null

echo
echo "Gotowe: ${APP_DIR}"
echo "Serwer: ${DEFAULT_SERVER_PATH}"
echo "Uruchom:  open \"${APP_DIR}\""
