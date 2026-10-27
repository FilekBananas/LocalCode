**Environment & deps.** Never call global `python`/`pip`. Use the project's venv (`.venv/bin/python`, `.venv/bin/pip`) or the tool it already uses (uv, Poetry, Hatch, Pipenv, conda). Dependencies and metadata live in `pyproject.toml` (PEP 621) or `requirements.txt`; add one only if it is already declared or the user asks. Pin nothing new without reason.

**Layout.** Common shapes: `src/<pkg>/` + `tests/`, or a flat package dir. Each package dir has `__init__.py`. Entry points go under `[project.scripts]`.

**Style.** PEP 8: 4-space indent, `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants, lines ~88–100 chars. Format with `ruff format` or `black`; lint with `ruff check` (or flake8). Prefer f-strings, `pathlib`, comprehensions, `enumerate`/`zip`, `dataclasses`, `enum.Enum`, context managers.

**Typing.** Add annotations to public APIs; check with `mypy` or `pyright` if configured. Use `list[int]`/`dict[str, int]` (3.9+), `X | None` (3.10+), `typing.Optional` only for older targets. Avoid `Any` where a real type fits.

**Errors.** Catch the narrowest exception; re-raise with `raise ... from exc` to keep the chain. Never swallow with bare `except`. Use `logging`, not `print`, in libraries.

**Testing.** pytest is standard: `test_*.py`, functions `test_*`, bare `assert`, `@pytest.fixture`, `@pytest.mark.parametrize`. Run `.venv/bin/pytest -q` (add `-x` to stop at the first failure, `-k name` to select). unittest is also seen. After any change run the suite and fix the cause of the first failure, not the symptom.

**Pitfalls.** Mutable default arguments (`def f(x=[])`) — use `None`; late-binding closures in loops; `==` vs `is` (only `is None`/`is True`); integer/float division; forgetting to `await` a coroutine; modifying a list while iterating; circular imports. Prefer `subprocess.run([...], check=True)` over `shell=True`.
