"""Tests for viz.plots.plot_protocol_timeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from sdg_leakage_audit.viz.plots import plot_protocol_timeline


@pytest.mark.parametrize("protocol_id", ["P0", "P1", "P2", "P3", "P4", "P5", "P6"])
def test_plot_protocol_timeline_saves_file_for_every_protocol(tmp_path: Path, protocol_id: str):
    out = tmp_path / f"{protocol_id}.png"
    plot_protocol_timeline(protocol_id, n=200, save_path=out)
    assert out.is_file() and out.stat().st_size > 0


def test_plot_protocol_timeline_rejects_unknown_protocol(tmp_path: Path):
    with pytest.raises(KeyError):
        plot_protocol_timeline("P9", n=100, save_path=tmp_path / "bad.png")
