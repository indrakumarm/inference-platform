# Phase 1: Direct vLLM and NGINX streaming

**Date:** 2026-11-19  
**Status:** Manual streaming smoke test passed on both paths. Proxy overhead was
not isolated. Failure behavior is covered in [Phase 2](02-backend-failure-recovery.md).

## Setup

- One warm vLLM backend serving `Qwen/Qwen3-8B`, with a 4,096-token context
  limit. The request asks for three short bullets on prefill versus decode.
- **Direct:** Mac -> SSH tunnel -> vLLM on GPU-host loopback port `8000`.
- **Via NGINX:** Mac -> private network -> host NGINX on port `8080` -> the same
  vLLM backend on loopback. NGINX was configured for streaming with response
  buffering disabled.
- The [manual timing script](../../tests/manual/test_ttft.py) sends the request
  from `/tmp/inference-stream.json` to `POST /v1/chat/completions`. The two
  timing series below used the same streaming payload with `max_tokens: 1024`,
  one request at a time.

## Direct functional checks

With the initial 256-token budget, the non-streaming request returned HTTP
`200` in **9.278 s**, and the streaming request returned HTTP `200` and `[DONE]`
in **9.297 s**. Both reported `finish_reason: length`: the request budget was
exhausted during `<think>`, before the requested bullets. `[DONE]` meant the
stream ended normally, not that the answer was complete.

With the larger budget, the direct stream showed the three answer bullets after
`</think>`. No server reconfiguration was required.

## Streaming timings

All times are client-observed seconds since the start of the request. TTFT is
the first non-empty generated-text delta (`<think>` in these runs), not the
arrival of HTTP headers. First answer is the first non-whitespace content after
`</think>`. Completion is receipt of `[DONE]`.

| Route | Run | TTFT (s) | First answer (s) | Completion (s) | Finish |
|---|---:|---:|---:|---:|---|
| Direct | 1 | 0.691 | 20.811 | 23.050 | `stop` |
| Direct | 2 | 0.670 | 20.868 | 22.968 | `stop` |
| Direct | 3 | 0.656 | 20.889 | 22.974 | `stop` |
| NGINX | 1 | 0.634 | 20.832 | 22.891 | `stop` |
| NGINX | 2 | 0.570 | 20.766 | 22.824 | `stop` |
| NGINX | 3 | 0.567 | 20.764 | 22.824 | `stop` |
| **Direct** | **Median** | **0.670** | **20.868** | **22.974** | |
| **NGINX** | **Median** | **0.570** | **20.766** | **22.824** | |

## What this tells us

- **NGINX preserved streaming:** generated text arrived in under a second, well
  before the stream completed at about 23 seconds in each proxied run.
- **TTFT is not time to first answer:** for a client that hides thinking text,
  the observed wait for an answer was about 21 seconds on both paths.
- **This is not a proxy overhead benchmark.** The direct runs used an SSH
  tunnel; the NGINX runs used the private network directly. The routes also ran
  in separate batches. The lower NGINX medians do not establish that the proxy
  made inference faster. Nothing here establishes a prefix-cache hit either.

## Follow-up

[Phase 2](02-backend-failure-recovery.md) checks what the client sees when
vLLM stops before or during a request.

If measuring proxy overhead later, run both routes through forwards over the
same SSH connection and alternate direct and proxied requests using the same
payload. Changing the network path and adding a proxy at the same time does
not isolate the proxy's contribution.
