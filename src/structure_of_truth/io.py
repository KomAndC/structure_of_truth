"""Results I/O. Every analysis writes a table (CSV or JSON) and a figure to results/."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _out_dir(name: str, root: Path | None) -> Path:
    d = (root or RESULTS_DIR) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(
    name: str, data: Mapping[str, Any], *, seed: int, root: Path | None = None, stem: str | None = None
) -> Path:
    """Write results/<name>/<stem or name>.json; the seed is always recorded."""
    path = _out_dir(name, root) / f"{stem or name}.json"
    payload = {"analysis": name, "seed": int(seed), **data}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable))
    return path


def write_csv(
    name: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    root: Path | None = None,
    stem: str | None = None,
) -> Path:
    """Write results/<name>/<stem or name>.csv; a `seed` column is added to every row."""
    rows = [{**r, "seed": int(seed)} for r in rows]
    if not rows:
        raise ValueError("refusing to write an empty results table")
    path = _out_dir(name, root) / f"{stem or name}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def save_figure(
    name: str, fig: plt.Figure, *, root: Path | None = None, stem: str | None = None
) -> Path:
    """Save results/<name>/<stem or name>.png and close the figure."""
    path = _out_dir(name, root) / f"{stem or name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _jsonable(x: Any) -> Any:
    import numpy as np

    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")
