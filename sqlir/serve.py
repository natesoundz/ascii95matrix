"""JSON-lines and local HTTP server for a compiled SQL-IR artifact."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .runtime import CompiledModel
from .sql_adapter import SQLAdapter
from .weights import DirectWeights, RubikWeights


def dispatch(model: CompiledModel, request: dict, state: dict | None = None) -> dict:
    request_id = request.get("id")
    state = state if state is not None else {}
    try:
        operation = request["op"]
        args = request.get("args") or {}
        weights = state.get("weights")
        if weights is None:
            weights = DirectWeights(model)
            state["weights"] = weights
        if operation == "forward":
            result = model.forward(args["text"], weights=weights)
        elif operation in {"generate", "scan"}:
            adapter = state.get("adapter")
            if adapter is None:
                adapter = SQLAdapter(str(model.directory), weights=weights)
                adapter.load()
                state["adapter"] = adapter
            if operation == "generate":
                result = adapter.run(
                    args["prompt"],
                    max_tokens=args.get("max_tokens", 40),
                    temperature=args.get("temperature", 0.0),
                )
            else:
                result = adapter.scan(args["text"])
        elif operation == "lookup":
            result = model.lookup(
                args["query_id"],
                args.get("key"),
                offset=args.get("offset", 0),
                limit=args.get("limit"),
            )
        elif operation == "count":
            result = model.lookup_count(args["query_id"], args.get("key"))
        elif operation == "row":
            result = model.row(args["table"], args["rowid"])
        elif operation == "list_queries":
            result = model.list_queries()
        elif operation == "manifest":
            result = dict(model.manifest)
            result["active_weights"] = weights.describe()
        else:
            raise ValueError(f"Unknown operation: {operation}")
        return {"id": request_id, "ok": True, "result": result}
    except Exception as exc:
        return {"id": request_id, "ok": False, "error": str(exc)}


def run_json_lines(model: CompiledModel, state: dict | None = None) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = dispatch(model, request, state)
        except Exception as exc:
            response = {"id": None, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
        sys.stdout.write("\n")
        sys.stdout.flush()


def run_http(model: CompiledModel, port: int, state: dict | None = None) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/call":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                response = dispatch(model, request, state)
            except Exception as exc:
                response = {"id": None, "ok": False, "error": str(exc)}
            payload = json.dumps(
                response, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            print(format % args, file=sys.stderr)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--http", type=int)
    parser.add_argument("--weights", choices=("direct", "rubik"), default="direct")
    parser.add_argument("--rubik-store", type=Path)
    parser.add_argument("--rubik-module", type=Path)
    args = parser.parse_args()
    model = CompiledModel(args.model)
    if args.weights == "rubik":
        if args.rubik_store is None or args.rubik_module is None:
            parser.error("rubik mode requires --rubik-store and --rubik-module")
        weights = RubikWeights(args.rubik_store, args.rubik_module)
    else:
        weights = DirectWeights(model)
    state = {"weights": weights, "adapter": None}
    if args.http is None:
        run_json_lines(model, state)
    else:
        run_http(model, args.http, state)


if __name__ == "__main__":
    main()
