# Phase 2: Backend failure and recovery

**Date:** 2026-11-19  
**Setup:** Same single vLLM backend and NGINX path as
[Phase 1](01-vllm-nginx-streaming.md). NGINX stayed running while the vLLM
container was stopped.

| Test | Observation |
|---|---|
| Stop vLLM before sending a fresh request | The user reported the request failed as expected. The exact HTTP status, body, and failure latency were not captured in the report. |
| Restart vLLM and send a new request without restarting NGINX | The stream reached `[DONE]` with `stop`: first text at **0.660 s**, first answer at **19.415 s**, completion at **21.474 s**. These are request timings after recovery, not outage duration. |
| Stop vLLM after the stream starts | First text (`<think>`) arrived at **0.571 s**. The client reported `Stream ended without [DONE]`; it did not report a completed answer. |

## What this tells us

A fresh request can succeed after vLLM restarts without restarting NGINX. That
is recovery, not failover: there was no other backend to serve requests during
the outage. The successful recovery request was observed in the first test;
recovery after the later mid-stream stop was not separately recorded.

Once response headers and partial text have been sent, NGINX cannot replace
that response with an HTTP error. The
[manual timing script](../../tests/manual/test_ttft.py) treats a stream that
ends without `[DONE]` as incomplete. Receiving text is not proof that the
generation finished.

## Client cancellation

The client received `<think>` at **0.642 s** and was interrupted with `Ctrl+C`
while reading the stream. vLLM stayed up. Its next reported samples showed
zero running and waiting requests; the first sample showed nonzero generation
throughput averaged over the reporting interval, and the next showed zero.
Those coarse samples did not show when generation stopped.

On the follow-up run, direct vLLM `/metrics` samples showed one running request
and an increasing generation-token counter. Between `16:44:43` and `16:44:44`
on the host, the running count fell from `1` to `0` and the cumulative token
counter went from `982` to `1000`. The counter stayed at `1000` in the
remaining samples. This shows generation stopped on the backend, not just
at the client. The excerpt does not contain the exact `Ctrl+C` timestamp,
so it cannot establish cancellation latency or rule out natural completion
on its own.

The one-request admission limit was tested in
[Phase 3](03-nginx-admission-control.md). Capacity under load is still untested.
