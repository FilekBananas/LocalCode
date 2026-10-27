"""``localcode image`` / ``localcode images`` / ``/obraz`` and the image client.

The CLI tests drive the real command functions with a fake image client; the two
integration tests at the bottom run the whole path with a fake engine — once
in-process (``LocalImageClient`` on a real ``Core``) and once over HTTP against a
live ``localcode serve`` (``RemoteImageClient``), which is what proves the
terminal works through both backends.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import pytest
from rich.console import Console

from localcode import cli
from localcode.tui import images_ui
from localcode.tui.app import LocalCodeTui
from localcode.tui.backends import BackendError
from localcode.tui.images_ui import ImageClient

_IMAGES_TESTS = Path(__file__).resolve().parents[1] / "images"
if str(_IMAGES_TESTS) not in sys.path:
    sys.path.insert(0, str(_IMAGES_TESTS))

aiohttp = pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestServer  # noqa: E402

from localcode.config import Settings  # noqa: E402
from localcode.images.gallery import ImageRecord  # noqa: E402
from localcode.server.app import create_app  # noqa: E402
from localcode.server.core import Core  # noqa: E402
from localcode.tui.backends import ServerInfo  # noqa: E402

from images_fakes import (  # noqa: E402
    IMAGE_KEY,
    PNG_1X1,
    FakeImageRuntime,
    make_image_model,
    make_result,
)
from server_fakes import CHAT_KEY, FakeEngine, FakeHub, FakeLibrary, FakeRuntime  # noqa: E402

TOKEN = "obrazy-test-token"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def make_dto(tmp_path: Path, index: int = 0, **over: Any) -> dict[str, Any]:
    path = tmp_path / f"obraz-{index}.png"
    path.write_bytes(PNG_1X1)
    dto = {
        "id": f"{index:032x}", "url": f"/api/images/{index:032x}/file", "path": str(path),
        "prompt": f"kot numer {index}", "negativePrompt": "", "modelKey": IMAGE_KEY,
        "modelName": "FLUX.1 schnell", "width": 1024, "height": 1024, "seed": 100 + index,
        "steps": 4, "guidance": None, "favorite": False, "sizeBytes": len(PNG_1X1),
        "createdAt": 1_700_000_000_000 + index, "elapsedMs": 2500,
    }
    dto.update(over)
    return dto


def flat(text: str) -> str:
    """Console output without the line wrapping, so long paths can be matched."""
    return "".join(text.split())


class FakeImageClient(ImageClient):
    """An :class:`ImageClient` with a scripted stream and an in-memory gallery."""

    def __init__(self, images: Iterable[dict[str, Any]] = (), *, error: str | None = None,
                 steps: int = 2) -> None:
        self.gallery = list(images)
        self.error = error
        self.steps = steps
        self.prompts: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.closed = False

    async def models(self) -> list[dict[str, Any]]:
        return [{"key": IMAGE_KEY, "imageFamily": "flux", "defaults": {"steps": 4}}]

    async def _stream(self, prompt: str, kwargs: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.prompts.append((prompt, kwargs))
        yield {"type": "status", "message": "Przygotowuję model…"}
        if self.error:
            yield {"type": "error", "message": self.error}
            return
        count = int(kwargs.get("count") or 1)
        for index in range(count):
            for step in range(1, self.steps + 1):
                yield {"type": "progress", "index": index, "step": step, "steps": self.steps}
            image = make_dto(self._tmp, index=len(self.gallery), prompt=prompt)
            self.gallery.append(image)
            yield {"type": "image", "image": image}
        yield {"type": "done"}

    def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        return self._stream(prompt, kwargs)

    async def list(self, *, limit: int = 20, offset: int = 0,
                   favorite: bool = False) -> list[dict[str, Any]]:
        rows = [i for i in self.gallery if i["favorite"]] if favorite else list(self.gallery)
        return rows[offset:offset + limit]

    async def get(self, image_id: str) -> dict[str, Any] | None:
        return next((i for i in self.gallery if i["id"] == image_id), None)

    async def delete(self, image_id: str) -> None:
        self.deleted.append(image_id)
        self.gallery = [i for i in self.gallery if i["id"] != image_id]

    async def set_favorite(self, image_id: str, favorite: bool) -> dict[str, Any]:
        image = await self.resolve(image_id)
        image["favorite"] = favorite
        return image

    async def aclose(self) -> None:
        self.closed = True


class StubBackend:
    """Just enough of a backend for the command functions."""

    kind = "local"

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def image_cli(monkeypatch: pytest.MonkeyPatch, console: Console, tmp_path: Path):
    """``cli._cmd_image`` / ``cli._cmd_images`` wired to a fake client."""
    client = FakeImageClient()
    client._tmp = tmp_path / "galeria"                      # type: ignore[attr-defined]
    client._tmp.mkdir(parents=True, exist_ok=True)          # type: ignore[attr-defined]
    backend = StubBackend()

    async def fake_open(options: Any, **kwargs: Any) -> Any:
        return backend

    monkeypatch.setattr(cli, "_open", fake_open)
    monkeypatch.setattr(cli, "_console", lambda: console)
    monkeypatch.setattr(images_ui, "image_client", lambda _backend: client)
    return client, backend


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_image_parser_reads_every_flag() -> None:
    options = cli.build_image_parser().parse_args(
        ["kot", "w", "kapeluszu", "-m", "schnell", "--size", "832x1216", "--steps", "6",
         "--seed", "42", "-n", "2", "-o", "plik.png", "--open"])
    assert options.prompt == ["kot", "w", "kapeluszu"]
    assert (options.model, options.size, options.steps) == ("schnell", "832x1216", 6)
    assert (options.seed, options.count, options.output) == (42, 2, "plik.png")
    assert options.open_file is True


def test_image_parser_defaults() -> None:
    options = cli.build_image_parser().parse_args(["kot"])
    assert (options.model, options.size, options.steps, options.seed) == (None, None, None, None)
    assert (options.count, options.output, options.open_file) == (1, None, False)


def test_images_parser_actions() -> None:
    parser = cli.build_images_parser()
    assert parser.parse_args([]).action is None
    assert parser.parse_args(["list", "--favorite"]).favorite is True
    assert parser.parse_args(["rm", "abc", "-y"]).yes is True
    assert parser.parse_args(["open", "abc"]).id == "abc"


@pytest.mark.parametrize("text, expected", [("1024x1024", (1024, 1024)), ("832X1216", (832, 1216)),
                                            ("1024×768", (1024, 768)), (None, (None, None))])
def test_parse_size(text: str | None, expected: tuple[int | None, int | None]) -> None:
    assert cli._parse_size(text) == expected


def test_parse_size_rejects_nonsense() -> None:
    with pytest.raises(ValueError) as excinfo:
        cli._parse_size("ogromny")
    assert "Nieprawidłowy rozmiar" in str(excinfo.value)


def test_image_is_a_known_subcommand() -> None:
    assert "image" in cli.SUBCOMMANDS and "images" in cli.SUBCOMMANDS


# --------------------------------------------------------------------------- #
# localcode image
# --------------------------------------------------------------------------- #


async def test_image_command_generates_and_reports(image_cli, console: Console) -> None:
    client, backend = image_cli
    code = await cli._cmd_image(["kot", "w", "kapeluszu", "--size", "512x512", "--seed", "7",
                                 "--no-preview"])
    assert code == 0

    prompt, kwargs = client.prompts[0]
    assert prompt == "kot w kapeluszu"
    assert (kwargs["width"], kwargs["height"], kwargs["seed"]) == (512, 512, 7)
    out = console.export_text()
    assert "1024×1024" in out                       # the summary line of the saved picture
    assert flat(client.gallery[0]["path"]) in flat(out)
    assert client.closed and backend.closed


async def test_image_command_writes_a_copy(image_cli, console: Console, tmp_path: Path) -> None:
    client, _ = image_cli
    target = tmp_path / "wyniki" / "kot.png"
    code = await cli._cmd_image(["kot", "-o", str(target), "--no-preview"])
    assert code == 0
    assert target.read_bytes() == PNG_1X1
    assert "Zapisano" in console.export_text()


async def test_image_command_numbers_a_series(image_cli, tmp_path: Path) -> None:
    code = await cli._cmd_image(["koty", "-n", "2", "-o", str(tmp_path / "kot.png"),
                                 "--no-preview"])
    assert code == 0
    assert (tmp_path / "kot-1.png").exists() and (tmp_path / "kot-2.png").exists()


async def test_image_command_can_open_the_result(image_cli, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(images_ui, "open_in_finder", lambda path: opened.append(str(path)) or True)
    assert await cli._cmd_image(["kot", "--open", "--no-preview"]) == 0
    assert opened and opened[0].endswith(".png")


async def test_image_command_reports_a_failure(image_cli, capsys: pytest.CaptureFixture) -> None:
    client, backend = image_cli
    client.error = "Nie udało się załadować modelu FLUX."
    code = await cli._run_subcommand("image", ["kot", "--no-preview"])
    assert code == 1
    assert "Nie udało się załadować modelu FLUX." in capsys.readouterr().err
    assert backend.closed


async def test_image_command_needs_a_prompt(image_cli, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):                 # argparse: prompt is required
        await cli._cmd_image([])


async def test_image_command_rejects_a_bad_size(image_cli, capsys: pytest.CaptureFixture) -> None:
    assert await cli._cmd_image(["kot", "--size", "ogromny"]) == 2
    assert "Nieprawidłowy rozmiar" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# localcode images
# --------------------------------------------------------------------------- #


async def test_images_list(image_cli, console: Console, tmp_path: Path) -> None:
    client, _ = image_cli
    client.gallery = [make_dto(tmp_path, 1, prompt="zamek nad jeziorem", favorite=True)]
    assert await cli._cmd_images(["list"]) == 0

    out = console.export_text()
    assert "zamek nad jeziorem" in out
    assert "★" in out
    assert "FLUX.1 schnell" in out


async def test_images_list_when_empty(image_cli, console: Console) -> None:
    assert await cli._cmd_images(["list"]) == 0
    assert "Galeria jest pusta" in console.export_text()


async def test_images_rm_accepts_a_prefix(image_cli, console: Console, tmp_path: Path) -> None:
    client, _ = image_cli
    client.gallery = [make_dto(tmp_path, 3)]
    image_id = client.gallery[0]["id"]

    assert await cli._cmd_images(["rm", image_id[:8], "-y"]) == 0
    assert client.deleted == [image_id]
    assert "Usunięto obraz" in console.export_text()


async def test_images_rm_reports_an_unknown_id(image_cli) -> None:
    with pytest.raises(BackendError) as excinfo:
        await cli._cmd_images(["rm", "nie-ma", "-y"])
    assert "Nie znaleziono obrazu" in str(excinfo.value)


async def test_images_open(image_cli, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, _ = image_cli
    client.gallery = [make_dto(tmp_path, 4)]
    opened: list[str] = []
    monkeypatch.setattr(images_ui, "open_in_finder", lambda path: opened.append(str(path)) or True)

    assert await cli._cmd_images(["open", client.gallery[0]["id"]]) == 0
    assert opened == [client.gallery[0]["path"]]


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def test_images_table(console: Console, tmp_path: Path) -> None:
    console.print(images_ui.images_table([make_dto(tmp_path, 0, prompt="las we mgle")]))
    out = console.export_text()
    assert "las we mgle" in out
    assert "1024×1024" in out
    assert "Wygenerowane obrazy" in out


def test_summary_line(console: Console, tmp_path: Path) -> None:
    console.print(images_ui.summary_line(make_dto(tmp_path, 0, seed=77, steps=4, elapsedMs=2500)))
    out = console.export_text()
    assert "ziarno 77" in out
    assert "4 kroków" in out
    assert "2,5 s" in out


@pytest.mark.parametrize("env, expected", [
    ({"TERM_PROGRAM": "iTerm.app"}, "iterm2"),
    ({"LC_TERMINAL": "iTerm2"}, "iterm2"),
    ({"TERM_PROGRAM": "WezTerm"}, "iterm2"),
    ({"KITTY_WINDOW_ID": "1"}, "kitty"),
    ({"TERM": "xterm-kitty"}, "kitty"),
    ({"TERM_PROGRAM": "Apple_Terminal"}, None),
    ({}, None),
    ({"TERM_PROGRAM": "iTerm.app", "TMUX": "/tmp/tmux"}, None),
])
def test_preview_protocol(env: dict[str, str], expected: str | None) -> None:
    assert images_ui.preview_protocol(env) == expected


def test_inline_preview_writes_an_iterm2_sequence(tmp_path: Path) -> None:
    path = tmp_path / "obraz.png"
    path.write_bytes(PNG_1X1)
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=80)

    assert images_ui.inline_preview(console, path, env={"TERM_PROGRAM": "iTerm.app"}) is True
    written = buffer.getvalue()
    assert written.startswith("\033]1337;File=inline=1;")
    assert f"size={len(PNG_1X1)}" in written


def test_inline_preview_writes_a_kitty_sequence(tmp_path: Path) -> None:
    path = tmp_path / "obraz.png"
    path.write_bytes(PNG_1X1)
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=80)

    assert images_ui.inline_preview(console, path, env={"TERM": "xterm-kitty"}) is True
    assert buffer.getvalue().startswith("\033_Ga=T,f=100,")


def test_inline_preview_is_silent_in_a_plain_terminal(console: Console, tmp_path: Path) -> None:
    path = tmp_path / "obraz.png"
    path.write_bytes(PNG_1X1)
    assert images_ui.inline_preview(console, path, env={}) is False
    assert images_ui.inline_preview(console, tmp_path / "nie-ma.png",
                                    env={"TERM_PROGRAM": "iTerm.app"}) is False


def test_save_copy_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(BackendError):
        images_ui.save_copy({"path": str(tmp_path / "nie-ma.png")}, tmp_path / "kopia.png")


# --------------------------------------------------------------------------- #
# generate_with_progress
# --------------------------------------------------------------------------- #


async def test_generate_with_progress_collects_the_images(console: Console,
                                                          tmp_path: Path) -> None:
    client = FakeImageClient()
    client._tmp = tmp_path                                  # type: ignore[attr-defined]
    images = await images_ui.generate_with_progress(client, console, "kot", count=2)
    assert len(images) == 2
    assert images[0]["prompt"] == "kot"


async def test_generate_with_progress_raises_the_server_error(console: Console,
                                                              tmp_path: Path) -> None:
    client = FakeImageClient(error="Model obrazów nie jest zainstalowany.")
    client._tmp = tmp_path                                  # type: ignore[attr-defined]
    with pytest.raises(BackendError) as excinfo:
        await images_ui.generate_with_progress(client, console, "kot")
    assert "nie jest zainstalowany" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# /obraz
# --------------------------------------------------------------------------- #


async def test_obraz_slash_command(backend: Any, console: Console, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeImageClient()
    client._tmp = tmp_path                                  # type: ignore[attr-defined]
    monkeypatch.setattr(images_ui, "image_client", lambda _backend: client)
    tui = LocalCodeTui(backend, console, history_path=tmp_path / "history.txt")

    await tui.dispatch("/obraz kot w kapeluszu")

    assert client.prompts[0][0] == "kot w kapeluszu"
    assert client.closed
    assert flat(client.gallery[0]["path"]) in flat(console.export_text())


async def test_obraz_without_a_prompt_explains_itself(backend: Any, console: Console,
                                                      tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeImageClient()
    client._tmp = tmp_path                                  # type: ignore[attr-defined]
    monkeypatch.setattr(images_ui, "image_client", lambda _backend: client)
    tui = LocalCodeTui(backend, console, history_path=tmp_path / "history.txt")

    await tui.dispatch("/obraz")
    assert "Użycie: /obraz" in console.export_text()
    assert client.prompts == []


async def test_obraz_works_in_chat_mode(backend: Any, console: Console, tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeImageClient()
    client._tmp = tmp_path                                  # type: ignore[attr-defined]
    monkeypatch.setattr(images_ui, "image_client", lambda _backend: client)
    tui = LocalCodeTui(backend, console, chat_mode=True, history_path=tmp_path / "h.txt")

    await tui.dispatch("/obraz pies")
    assert client.prompts[0][0] == "pies"


# --------------------------------------------------------------------------- #
# Both backends, end to end with a fake engine
# --------------------------------------------------------------------------- #


@pytest.fixture
def image_library() -> FakeLibrary:
    return FakeLibrary([make_image_model(IMAGE_KEY, name="FLUX.1 schnell")])


@pytest.fixture
async def image_core(tmp_path: Path, isolated_home: Path,
                     image_library: FakeLibrary) -> AsyncIterator[Core]:
    runtime = FakeImageRuntime(tmp_path / "engine", steps=2)
    core = Core(Settings(models_dir=str(tmp_path / "models")), db_path=tmp_path / "chat.db",
                library=image_library, hub=FakeHub(),
                runtime=FakeRuntime({CHAT_KEY: FakeEngine()}), images=runtime)
    await core.start()
    try:
        yield core
    finally:
        await core.shutdown()


async def test_local_client_runs_the_whole_path(image_core: Core, console: Console) -> None:
    """In-process: prompt → engine → gallery file → listing → deletion."""
    client = images_ui.LocalImageClient(image_core)
    models = await client.models()
    assert [m["key"] for m in models] == [IMAGE_KEY]
    assert models[0]["defaults"]["steps"] == 4

    images = await images_ui.generate_with_progress(client, console, "kot w kapeluszu",
                                                    width=512, height=512, steps=2, seed=3)
    assert len(images) == 1
    image = images[0]
    assert Path(image["path"]).read_bytes() == PNG_1X1
    assert Path(image["path"]).parent.parent == image_core.gallery.root
    assert image["seed"] == 3

    assert [r["id"] for r in await client.list()] == [image["id"]]
    assert (await client.set_favorite(image["id"], True))["favorite"] is True
    assert [r["id"] for r in await client.list(favorite=True)] == [image["id"]]

    await client.delete(image["id"])
    assert await client.list() == []
    assert not Path(image["path"]).exists()


@pytest.fixture
async def image_server(image_core: Core) -> AsyncIterator[ServerInfo]:
    server = TestServer(create_app(image_core, TOKEN))
    await server.start_server()
    try:
        yield ServerInfo("127.0.0.1", server.port, TOKEN, pid=1, version="0.1.0")
    finally:
        await server.close()


async def test_remote_client_runs_the_whole_path(image_server: ServerInfo, image_core: Core,
                                                 console: Console) -> None:
    """The same thing over the HTTP API of a running ``localcode serve``."""
    client = images_ui.RemoteImageClient(image_server)
    try:
        assert [m["key"] for m in await client.models()] == [IMAGE_KEY]

        images = await images_ui.generate_with_progress(client, console, "pies w berecie",
                                                        steps=2, count=2)
        assert len(images) == 2
        assert all(Path(i["path"]).is_file() for i in images)
        assert image_core.gallery.count() == 2

        listed = await client.list()
        assert [r["id"] for r in listed] == [images[1]["id"], images[0]["id"]]
        assert (await client.set_favorite(listed[0]["id"], True))["favorite"] is True
        assert [r["id"] for r in await client.list(favorite=True)] == [listed[0]["id"]]

        await client.delete(listed[0]["id"])
        assert [r["id"] for r in await client.list()] == [listed[1]["id"]]
    finally:
        await client.aclose()


async def test_remote_client_reports_a_server_error(image_server: ServerInfo,
                                                    console: Console) -> None:
    client = images_ui.RemoteImageClient(ServerInfo(image_server.host, image_server.port, "zly"))
    try:
        with pytest.raises(BackendError) as excinfo:
            await client.models()
        assert "token" in str(excinfo.value).lower()
    finally:
        await client.aclose()


def test_image_client_picks_the_right_implementation(image_core: Core) -> None:
    class Remote:
        kind = "remote"
        info = ServerInfo("127.0.0.1", 1234, "t")

    class Local:
        kind = "local"
        core = image_core

    assert isinstance(images_ui.image_client(Remote()), images_ui.RemoteImageClient)
    assert isinstance(images_ui.image_client(Local()), images_ui.LocalImageClient)
    with pytest.raises(BackendError):
        images_ui.image_client(StubBackend())


def test_record_dto_and_client_dto_agree(image_core: Core, tmp_path: Path) -> None:
    """What the terminal reads is exactly what the gallery stores."""
    record: ImageRecord = image_core.gallery.add(make_result(tmp_path / "e"))
    dto = record.to_dict()
    assert dto["path"] == record.path
    assert dto["url"].endswith("/file")


async def test_local_client_turns_server_errors_into_backend_errors(image_core: Core) -> None:
    """``ApiError``/``ImageGalleryError`` must reach the terminal as ``BackendError``."""
    client = images_ui.LocalImageClient(image_core)
    with pytest.raises(BackendError) as excinfo:
        await client.set_favorite("nie-ma", True)
    assert "Nie znaleziono obrazu" in str(excinfo.value)

    with pytest.raises(BackendError) as excinfo:
        async for _frame in client.generate("kot", model_key="nie-ma-takiego"):
            pass
    assert "Nie znaleziono modelu obrazów" in str(excinfo.value)

    image_core._images = None
    image_core._images_built = True
    with pytest.raises(BackendError) as excinfo:
        async for _frame in client.generate("kot"):
            pass
    assert "localcode[images]" in str(excinfo.value)
