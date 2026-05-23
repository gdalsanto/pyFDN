"""Smoke-test that every example script runs to completion."""

import runpy
import shutil
from pathlib import Path

import pytest


plt = pytest.importorskip("matplotlib.pyplot")
pio = pytest.importorskip("plotly.io")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def _example_scripts() -> list[Path]:
    return sorted(EXAMPLES_DIR.rglob("example_*.py"))


@pytest.fixture(autouse=True)
def _headless_rendering():
    """Render off-screen (no GUI window / browser tab) and close figures."""
    plt.switch_backend("Agg")
    pio.renderers.default = "json"
    yield
    plt.close("all")


def test_examples_dir_is_populated() -> None:
    assert _example_scripts(), f"No example scripts found under {EXAMPLES_DIR}"


@pytest.mark.parametrize(
    "script",
    _example_scripts(),
    ids=lambda p: str(p.relative_to(EXAMPLES_DIR)),
)
def test_example_runs(script: Path) -> None:
    # draw_flamo_graph shells out to the system Graphviz "dot" binary, which is
    # not pip-installable. Skip (don't fail) where it isn't available.
    if "draw_flamo_graph" in script.read_text() and shutil.which("dot") is None:
        pytest.skip("system Graphviz 'dot' binary not installed")
    runpy.run_path(str(script), run_name="__main__")
