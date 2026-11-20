# Phase 3: NGINX admission control

**Status:** Manual check passed on the single-backend NGINX route from
[Phase 1](01-vllm-nginx-streaming.md).

## NGINX configuration

The [reusable gateway config](../../gateway/nginx/inference.conf) keeps these
settings and moves the backend address into an upstream pool. The snippets
below show the single-backend configuration used in the test.

Add the zone to the `http` context, outside any `server` block:

```nginx
limit_conn_zone $server_name zone=inference_inflight:1m;
```

In the existing NGINX `server` block, the chat route combines the streaming
proxy from Phase 1 with the admission limit:

```nginx
location = /v1/chat/completions {
    limit_conn inference_inflight 1;
    limit_conn_status 429;

    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_connect_timeout 5s;
    proxy_read_timeout 120s;
}
```

This snippet leaves the existing NGINX listener unchanged and keeps vLLM
on host loopback. `1m` reserves shared memory for counters across NGINX
workers; it is not a request count. `$server_name` makes this a shared limit
for clients of the same virtual server, not a per-client limit. `1` allows
one active request at a time through this location for that server name.
`proxy_read_timeout` limits the gap between upstream reads, not total
request duration.

On the NGINX host, validate before reloading:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

## Result

The first request streamed through NGINX. An overlapping second request
received `429`. After the first stream finished, retrying the second request
succeeded. The test did not capture how long the rejection took.

## What this tells us

NGINX rejected overlapping work at the gateway and admitted new work once the
slot was free. The limit applies only to requests through that NGINX location.
It is a deliberately small test setting, not a measured vLLM capacity limit.

## Not measured

The optional access-log comparison of client and upstream status was skipped.
This phase did not measure sustainable concurrency, queueing behavior inside
vLLM, or latency under load.
