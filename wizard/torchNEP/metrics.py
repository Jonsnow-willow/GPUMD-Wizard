from __future__ import annotations

import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict) -> None:
        row = {key: _format_value(value) for key, value in row.items()}
        fieldnames = list(row)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        if self._fieldnames is not None and self._fieldnames != fieldnames:
            write_header = True
        self._fieldnames = fieldnames
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.10g}"
    return value
