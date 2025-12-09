from __future__ import annotations

from pathlib import Path
from typing import Tuple


def save_plot_with_fallback(fig, target_path, fallback_html: bool = True) -> Tuple[Path, bool, Exception | None]:
    """
    Try to save a Plotly figure as a static image; fall back to HTML if the image
    backend (e.g., Kaleido/Chrome) is unavailable on the host.

    Returns (saved_path, used_fallback, error)
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_image(path)
        return path, False, None
    except Exception as exc:
        if not fallback_html:
            raise

        html_path = path.with_suffix(".html")
        fig.write_html(html_path, include_plotlyjs=True)
        return html_path, True, exc
