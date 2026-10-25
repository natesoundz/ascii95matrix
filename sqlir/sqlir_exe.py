"""PyInstaller dispatch entry point for the conical SQL-IR commands."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {
        "serve",
        "verify",
        "compile",
        "chat",
        "weights",
    }:
        raise SystemExit("Usage: conical serve|verify|compile|chat|weights ...")
    command = sys.argv.pop(1)
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if command == "serve":
        from sqlir.serve import main as run
    elif command == "verify":
        from sqlir.verify import main as run
    elif command == "compile":
        from sqlir.compile import main as run
    elif command == "chat":
        from sqlir.chat import main as run
    else:
        from sqlir.weights import main as run
    run()


if __name__ == "__main__":
    main()
