# Mowa: dyktowanie i czytanie na głos

Dwa zadania, które w sklepie stoją na jednej półce („Mowa"), a w środku nie mają
ze sobą nic wspólnego:

* **rozpoznawanie mowy** (`stt`) — nagranie zamienia się w tekst; na Apple
  Silicon robi to `mlx-whisper` (modele Whisper i pokrewne),
* **synteza** (`tts`) — tekst zamienia się w plik dźwiękowy; robi to
  `mlx-audio` (Kokoro i zgodne).

Oba to zadania na kilka sekund, więc — inaczej niż modele językowe — nie
zostają w pamięci: model jest wczytywany, używany i zwalniany w obrębie jednego
wywołania, w wątku roboczym. Dzięki temu dyktowanie nigdy nie wyrzuca z pamięci
modelu, z którym ktoś właśnie rozmawia.

## Instalacja

> **Kokoro potrzebuje `misaki[en]`.** Bez tej paczki mlx-audio wczytuje model,
> wypisuje „requires the optional 'misaki' package” i **nie tworzy pliku** — dlatego
> `localcode[speech]` wymaga jej wprost, a `SpeechRuntime.speak` sprawdza po wygenerowaniu,
> czy plik naprawdę powstał, i inaczej zgłasza błąd, zamiast udawać sukces.

```bash
pip install "localcode[speech]"     # mlx-whisper + mlx-audio
localcode doctor                    # sprawdzi, czego brakuje
```

> **Który `pip`?** Aplikacja na Maca uruchamia serwer z `~/.localcode/venv`,
> a `pip` w PATH to zwykle zupełnie inny Python. Dlatego komunikat w aplikacji
> podaje pełną ścieżkę interpretera (`localcode.config.install_command`) —
> to polecenie można wkleić i zadziała:
> `~/.localcode/venv/bin/python -m pip install "localcode[speech]"`.

Bez dodatku wszystko działa dalej — każde wejście mówi po polsku, co doinstalować,
zamiast wysypać się na `ImportError`.

Model bierzesz ze sklepu, z kategorii **Mowa**:

```bash
localcode models search whisper
localcode models install "mlx-community/whisper-small-mlx"
```

## W terminalu

```bash
localcode transcribe nagranie.m4a                    # tekst na ekran
localcode transcribe wykład.mp4 --srt napisy.srt     # i napisy do pliku
localcode transcribe rozmowa.wav --language pl --out tekst.txt
localcode say "Dzień dobry, tu LocalCode." --play    # czytanie na głos
localcode say "Tekst do lektora" --voice af_bella --out lektor.wav
```

## Przez HTTP

```
POST /api/speech/transcribe {path, modelKey?, language?, segments?, srt?}
POST /api/speech/speak      {text, modelKey?, voice?, speed?, output?}
GET  /api/speech/voices?modelKey=
```

Nagranie podaje się **ścieżką na tym komputerze**, nie jako upload: aplikacja i
terminal działają obok serwera, pliki bywają setkami megabajtów, a przesyłanie
ich przez HTTP tylko po to, żeby serwer zapisał je z powrotem na dysk, byłoby
objazdem donikąd.

## Skąd się biorą modele w katalogu

`scripts/build_catalog.py` rozpoznaje je po `pipeline_tag` z Hugging Face
(`automatic-speech-recognition`, `text-to-speech`, `text-to-audio`) i nadaje im
kategorię „mowa" z odpowiednią zdolnością (`stt` albo `tts`). Wideo
(`text-to-video`, `image-to-video`) dostaje kategorię „wideo" — jest w katalogu
i da się je pobrać, ale **silnika do wideo jeszcze nie ma**; lokalne generowanie
wideo na 32 GB to minuty na sekundę materiału i kilkadziesiąt gigabajtów pamięci.

Dwie rzeczy, które trzeba było naprawić, żeby te modele w ogóle dało się
zainstalować i wczytać — warto o nich pamiętać przy kolejnych formatach:

* `hub/search.py` odmawiał repozytoriom bez GGUF i bez `safetensors`, a porty
  MLX Whispera trzymają wagi w `weights.npz`; teraz takie repozytorium instaluje
  się w całości (`_whole_repo_variant`);
* `hub/library.py` uznawał katalog MLX za kompletny tylko po `safetensors`,
  więc pobrany model nie pojawiał się na liście; `*.npz` liczy się tak samo,
  a `hub/metadata.py` rozpoznaje model mowy po kształcie configu
  (`n_mels` + `n_audio_layer`) albo po nazwie rodziny.
