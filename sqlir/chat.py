"""Interactive REPL for the compiled SQL-derived adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from .sql_adapter import SQLAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--weights", choices=("direct", "rubik"), default="direct")
    parser.add_argument("--rubik-store", type=Path)
    parser.add_argument("--rubik-module", type=Path)
    args = parser.parse_args()
    if args.weights == "rubik" and (
        args.rubik_store is None or args.rubik_module is None
    ):
        parser.error("rubik mode requires --rubik-store and --rubik-module")
    adapter = SQLAdapter(
        str(args.model),
        weight_mode=args.weights,
        rubik_store=args.rubik_store,
        rubik_module=args.rubik_module,
    )
    adapter.load()
    try:
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                break
            if not prompt.strip():
                continue
            result = adapter.run(prompt)
            print(f"{result['procedure']}: {result['text']}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
