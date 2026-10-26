"""Minimal OpenAI-compatible HTTP server on top of the Python binding (stdlib only; one request at a time).

    PYTHONPATH=build python scripts/serve.py --model models/bitnet-2b4t.ternix [--gpu] [--port 8080]

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions (both with "stream": true/false).
Binds to 127.0.0.1 by default; there is no authentication, so do not expose it on an untrusted network.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import ternix_engine as tx

EOS_IDS = {128001, 128009}
LOCK = threading.Lock()   # the engine holds one KV cache: serialize requests


def render_chat(messages: list[dict]) -> str:
    """BitNet b1.58 2B4T template: 'Role: content<|eot_id|>' per message, then 'Assistant: '."""
    return "".join(f"{m['role'].capitalize()}: {str(m['content']).strip()}<|eot_id|>" for m in messages) + "Assistant: "


def sample(logits: np.ndarray, temperature: float, top_k: int, top_p: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    idx = np.argsort(-logits)[: top_k if top_k > 0 else 4096]
    p = np.exp((logits[idx] - logits[idx[0]]) / temperature)
    p /= p.sum()
    if top_p < 1.0:
        keep = max(1, int(np.searchsorted(np.cumsum(p), top_p) + 1))
        idx, p = idx[:keep], p[:keep] / p[:keep].sum()
    return int(rng.choice(idx, p=p))


def generate(engine, prompt: str, max_tokens: int, temperature: float, top_k: int, top_p: float, seed: int):
    """Yields (text_piece, finish_reason|None, n_prompt_tokens)."""
    ids = engine.encode(prompt, True, True)
    rng = np.random.default_rng(seed)
    logits = engine.prefill(ids)
    pending = b""
    for i in range(max_tokens):
        t = sample(logits, temperature, top_k, top_p, rng)
        if t in EOS_IDS:
            yield pending.decode("utf-8", "replace"), "stop", len(ids)
            return
        pending += engine.decode([t])
        cut = len(pending)
        while cut > 0:
            try:
                pending[:cut].decode("utf-8")
                break
            except UnicodeDecodeError:
                cut -= 1
        if cut:
            yield pending[:cut].decode("utf-8"), None, len(ids)
            pending = pending[cut:]
        if len(ids) + i + 2 >= engine.max_ctx:
            yield pending.decode("utf-8", "replace"), "length", len(ids)
            return
        logits = engine.step(t)
    yield pending.decode("utf-8", "replace"), "length", len(ids)


class Handler(BaseHTTPRequestHandler):
    engine = None
    model_name = "bitnet-b1.58-2B-4T"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "gpu": self.engine.is_gpu})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.model_name, "object": "model", "owned_by": "local"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            chat = self.path.endswith("chat/completions")
            prompt = render_chat(req["messages"]) if chat else req["prompt"]
        except (KeyError, ValueError, TypeError) as e:
            return self._json(400, {"error": f"bad request: {e}"})
        args = (int(req.get("max_tokens", 256)), float(req.get("temperature", 0.0)), int(req.get("top_k", 0)),
                float(req.get("top_p", 1.0)), int(req.get("seed", int(time.time()))))
        rid, created, stream = "cmpl-" + uuid.uuid4().hex[:24], int(time.time()), bool(req.get("stream", False))
        obj = "chat.completion" if chat else "text_completion"

        def chunk(text, finish):
            choice = {"index": 0, "finish_reason": finish}
            choice.update({"delta": {"content": text}} if chat else {"text": text})
            return {"id": rid, "object": obj + (".chunk" if chat else ""), "created": created, "model": self.model_name, "choices": [choice]}

        with LOCK:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    for text, finish, _ in generate(self.engine, prompt, *args):
                        self.wfile.write(b"data: " + json.dumps(chunk(text, finish)).encode() + b"\n\n")
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            text, finish, n_prompt = "", "length", 0
            n_out = 0
            for piece, f, n_prompt in generate(self.engine, prompt, *args):
                text += piece
                n_out += 1
                finish = f or finish
        msg = {"message": {"role": "assistant", "content": text}} if chat else {"text": text}
        self._json(200, {"id": rid, "object": obj, "created": created, "model": self.model_name,
                         "choices": [{"index": 0, "finish_reason": finish, **msg}],
                         "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--head", default="int8", choices=["bf16", "int8", "int4"], help="output head: int8 is certified-exact for greedy")
    ap.add_argument("--sparse-down", action="store_true")
    ap.add_argument("--kv-int8", action="store_true", help="CPU only")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    Handler.engine = tx.Engine(a.model, gpu=a.gpu, threads=a.threads, ctx=a.ctx, head=a.head, sparse_down=a.sparse_down, kv_int8=a.kv_int8)
    print(f"serving on http://{a.host}:{a.port} ({'GPU' if a.gpu else 'CPU'}, ctx {a.ctx})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
