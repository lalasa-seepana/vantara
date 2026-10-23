"""Configuration, reproducible logging and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


class JsonFormatter(logging.Formatter):
    """Write machine-readable records without leaking request contents."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a single log record."""
        return json.dumps(
            {
                "time": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def setup_logging() -> None:
    """Configure structured console logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), handlers=[handler], force=True
    )


def config() -> dict[str, Any]:
    """Read YAML configuration and resolve environment path overrides."""
    path = Path(os.getenv("CONFIG_PATH", str(ROOT / "config/config.yaml")))
    cfg = yaml.safe_load(path.read_text())
    for name, value in cfg["paths"].items():
        value = os.getenv(f"VANTARA_{name.upper()}_PATH", value)
        cfg["paths"][name] = str((ROOT / value).resolve())
    return cfg


def digest(path: Path) -> str:
    """Calculate SHA-256 without changing the source."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write valid JSON with NumPy numbers converted to native values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
            allow_nan=False,
        )
    )


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text())
