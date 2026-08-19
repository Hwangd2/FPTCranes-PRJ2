from __future__ import annotations

from io import StringIO

import matplotlib
from rich.console import Console

from src.pipeline import configure_logging


def test_offline_plotting_forces_non_interactive_agg_backend() -> None:
    import src.utils.pipeline_plots  # noqa: F401

    assert matplotlib.get_backend().lower() == "agg"


def test_rich_logging_honors_info_and_debug_levels() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    logger = configure_logging("DEBUG", console=console)

    logger.info("high-level stage")
    logger.debug("fold-level detail")

    rendered = output.getvalue()
    assert "INFO" in rendered
    assert "high-level stage" in rendered
    assert "DEBUG" in rendered
    assert "fold-level detail" in rendered


def test_info_logging_hides_debug_details() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)
    logger = configure_logging("INFO", console=console)

    logger.info("high-level stage")
    logger.debug("fold-level detail")

    rendered = output.getvalue()
    assert "high-level stage" in rendered
    assert "fold-level detail" not in rendered
