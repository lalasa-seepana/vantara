"""Generate local database settings without committing credentials."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from src.utils.common import ROOT, setup_logging
import logging


def main() -> None:
    """Create an owner-readable .env only if it does not already exist."""
    setup_logging()
    path: Path = ROOT / ".env"
    if path.exists():
        logging.info("Existing .env preserved")
        return
    password = secrets.token_hex(24)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        output.write(
            f"POSTGRES_USER=vantara\nPOSTGRES_DB=vantara\nPOSTGRES_PASSWORD={password}\nLOG_LEVEL=INFO\n"
        )
    logging.info("Created local .env; keep it private and outside version control")


if __name__ == "__main__":
    main()
