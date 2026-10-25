"""JSON-lines and local HTTP server for a compiled SQL-IR artifact."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .runtime import CompiledModel


def dispatch(model: CompiledModel, request: dict) -> dict:
    request_id = request.get("id")
    try:
        operation = request["op"]
        args = request.get("args") or {}
        if operation == "forward":
            result = model.forward(args["text"])
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
            result = model.manifest
        else:
            raise ValueError(f"Unknown operation: {operation}")
        return {"id": request_id, "ok": True, "result": result}
    except Exception as exc:
        return {"id": request_id, "ok": False, "error": str(exc)}


def run_json_lines(model: CompiledModel) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = dispatch(model, request)
        except Exception as exc:
            response = {"id": None, "ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
        sys.stdout.write("\n")
        sys.stdout.flush()


def run_http(model: CompiledModel, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/call":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                response = dispatch(model, request)
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
    args = parser.parse_args()
    model = CompiledModel(args.model)
    if args.http is None:
        run_json_lines(model)
    else:
        run_http(model, args.http)


if __name__ == "__main__":
    main()
