import json, os , ssl
import time
from pathlib import Path
from urllib.request import Request, urlopen

payload = Path("/tmp/inference-stream.json").read_bytes()
"""
request = Request(
    #"http://127.0.0.1:8000/v1/chat/completions",
    "http://10.192.10.244:8080/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
"""
token = os.getenv("API_TOKEN")  # rename to whatever env var you use

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

headers = {"Content-Type": "application/json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

request = Request(
    "https://inference.example.internal/v1/chat/completions",
    data=payload,
    headers=headers,
)

start = time.perf_counter()
first_text = None
finish_reason = None
first_answer = None
thinking_ended = False
marker_tail = ""

with urlopen(request, context=ctx, timeout=120) as response:
    for line in response:
        if not line.startswith(b"data:"):
            continue

        elapsed = time.perf_counter() - start
        data = line[len(b"data:"):].strip()

        if data == b"[DONE]":
            print(f"Stream completed: {elapsed:.3f} s")
            print(f"Finish reason:    {finish_reason}")
            break

        for choice in json.loads(data).get("choices", []):
            if choice["index"] != 0:
                continue
            delta = choice.get("delta", {})
            text = delta.get("reasoning") or delta.get("content") or ""
            if first_text is None and text.strip():
                first_text = elapsed
                print(f"TTFT: {first_text:.3f} s | first text: {text!r}", flush=True)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            content = delta.get("content") or ""
            if first_answer is None and content:
                if not thinking_ended:
                    before, marker, content = (marker_tail + content).partition("</think>")
                    thinking_ended = bool(marker)
                    # Keep enough trailing text to detect a tag split across chunks.
                    marker_tail = before[-7:] if not thinking_ended else ""

                if thinking_ended and content.strip():
                    first_answer = elapsed
                    print(f"Time to first answer: {first_answer:.3f} s", flush=True)
    else:
        raise SystemExit("Stream ended without [DONE]")
if first_answer is None:
                print("Time to first answer: not observed")
