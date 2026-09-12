"""Person 1 stream API (stdlib only, no new deps).

  python api_server.py [--port 8000] [--no-camera]

Endpoints (latest-packet semantics, never queued):
  GET /health  -> {"ok": true, "backend": ...}
  GET /light   -> latest LightPacket JSON (Person 4 field names)
  GET /stream  -> text/event-stream SSE, one event per new packet

Example consumer (renderer / teammate / phone on same Wi-Fi):
  curl http://<PC_IP>:8000/light
  curl -N http://<PC_IP>:8000/stream
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pipeline import TorchPipeline


class Handler(BaseHTTPRequestHandler):
    pipe: TorchPipeline | None = None
    server_version = "TorchAPI/1.0"

    def log_message(self, *args) -> None:  # quieter logs
        pass

    def _send_json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        pipe = Handler.pipe
        if self.path == "/health":
            lat = pipe.latest if pipe else None
            self._send_json({"ok": True,
                             "backend": lat.backend if lat else None})
        elif self.path == "/light":
            self._send_json(pipe.to_dict() if pipe else {"active": False})
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                for pkt in pipe.packets():
                    data = json.dumps(pipe.to_dict(pkt))
                    self.wfile.write(f"data: {data}\n\n".encode())
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send_json({"error": "use /health, /light, /stream"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-camera", action="store_true")
    args = ap.parse_args()
    pipe = TorchPipeline(no_camera=args.no_camera).start()
    Handler.pipe = pipe
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[api] serving on :{args.port}  (/health /light /stream)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
