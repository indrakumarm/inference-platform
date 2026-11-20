# NGINX gateway

[Back to quick start](../README.md)

The [gateway config](../gateway/nginx/inference.conf) is separate from the vLLM
deployment profile. By default it listens on `127.0.0.1:8080` on the NGINX host
and streams `POST /v1/chat/completions` to vLLM on `127.0.0.1:8000` on that
same host. The default profile is for operator testing over SSH and does not
enable TLS or authentication. Use the HTTPS profile below for VPN clients.

## Render and deploy

From the repo root on your workstation, render the config from
[`gateway.json`](../gateway.json):

```bash
python3 -m inference_platform gateway render --profile gateway.json
```

The render command validates the profile and prints NGINX config to stdout
without using SSH. The default profile reproduces the checked-in config.
To install the generated config and reload NGINX, explicitly target the
gateway host:

```bash
python3 -m inference_platform gateway deploy --host ubuntu@gateway.example --profile gateway.json
```

Add `--key` if you use an SSH key file. The command previews the config locally,
then asks `[y/N]` before contacting the host. Enter `y` or `yes` to proceed;
pressing Enter cancels. Set `INFERENCE_PLATFORM_LOG_LEVEL=INFO` to see live
progress through the remote service check, config validation, and reload.
It requires Python 3, non-interactive sudo,
and an existing, running systemd-managed NGINX service that loads
`/etc/nginx/conf.d/*.conf`. It writes only the managed site
`/etc/nginx/conf.d/inference-platform.conf`, validates the full config with
`nginx -T`, and requests a reload only if validation passes and includes that
site. On validation or reload-command failure, it restores the prior file
(or removes the new file on first deploy). If a reload was attempted and
failed, check the running service before retrying. This is separate from
vLLM `deploy` and does not check backend readiness.

If you already enabled a hand-written inference site, back it up and disable
it before using the managed deployment. Do not enable both configs for the
same listener. For a manual installation instead, run on the NGINX host
from the repo root:

```bash
sudo install -m 0644 gateway/nginx/inference.conf /etc/nginx/conf.d/inference.conf
sudo nginx -t && sudo systemctl reload nginx
```

This assumes the Ubuntu NGINX service is running and includes `conf.d/*.conf`
inside `http`. From your workstation, forward local port `8080` to
`127.0.0.1:8080` on the NGINX host over SSH. Then, using a known-good streaming
request body, verify that the gateway sends incremental events and `[DONE]`:

```bash
curl -fsSN --max-time 120 \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/inference-stream.json \
  http://127.0.0.1:8080/v1/chat/completions
```

## HTTPS and bearer authentication for VPN clients

Copy [`gateway.json`](../gateway.json) to `gateway-https.json`. Keep the existing
backend settings, change `listen` and add `tls`, for example:

```json
{
  "listen": {"host": "10.0.1.10", "port": 443},
  "backends": [{"host": "127.0.0.1", "port": 8000}],
  "max_in_flight": 1,
  "connect_timeout_seconds": 5,
  "upstream_read_timeout_seconds": 120,
  "tls": {"server_name": "inference.example.internal"}
}
```

Use the gateway host's actual private IP and a DNS name that VPN clients can
resolve and verify against the certificate. With `tls`, the listener accepts
loopback or a specific RFC 1918 IPv4 address, not `0.0.0.0`. Restrict the
listener port to VPN clients using the security group and host firewall.
Port `443` is allowed only with TLS and can be omitted from the HTTPS URL.
Check that NGINX can bind the port and that no other service already owns it.
For a nonstandard port, include the configured port in the URL. Keep vLLM
itself on loopback.

#### Quick setup with a self-signed lab certificate

On the gateway host, OpenSSL must be installed. Generate a token with
`openssl rand -hex 32` in a trusted local shell and save it in your secrets
manager. From the repo root, use the HTTPS profile you created above:

```bash
python3 -m inference_platform gateway bootstrap-tls \
  --host ubuntu@gateway.example --profile gateway-https.json
python3 -m inference_platform gateway deploy \
  --host ubuntu@gateway.example --profile gateway-https.json
```

Add `--key` to both commands if needed. Bootstrap prompts for the token
without echoing it, generates a 30-day self-signed certificate for
`tls.server_name` on the gateway, and installs the key and token map with
mode `0600`. It refuses to overwrite an existing credentials directory and
does not reload NGINX. The second command activates the HTTPS site.

Retrieve only the public certificate over trusted SSH and use it to verify
the gateway from the client (use your actual gateway SSH target and DNS name):

```bash
ssh ubuntu@gateway.example \
  'sudo -n cat /etc/nginx/inference-platform/fullchain.pem' > gateway-lab.pem
curl -sS --cacert gateway-lab.pem -o /dev/null -w 'HTTP %{http_code}\n' \
  -H 'Content-Type: application/json' --data '{}' \
  https://inference.example.internal/v1/chat/completions
```

Expect `401` without a token. Then send a known-good streaming request with
the token supplied from your secrets manager and check for `[DONE]`. Do not
use `curl -k`. This lab workflow requires distributing the public certificate
to each client; it does not automate renewal or token rotation.

#### Using an existing CA-issued certificate instead

Skip bootstrap and provision these files under `/etc/nginx/inference-platform/`
before running `gateway deploy` with the HTTPS profile:

- `fullchain.pem`: a certificate and chain for the configured DNS name, issued
  by a CA your clients trust.
- `privkey.pem`: the matching private key.
- `bearer-token.map`: a root-only file with one NGINX map entry. This example
  shows the format, not a usable credential:

  ```nginx
  "~^Bearer[ ]REPLACE_WITH_64_LOWERCASE_HEX_CHARACTERS$" 1;
  ```

Keep the directory root-owned and restrict the private key and token map to
mode `0600`. Generate the shared token from 32 random bytes (for example
`openssl rand -hex 32` in a trusted local shell) and distribute it through your
secrets manager. The regex is case-sensitive and anchored; do not use a plain
NGINX map string, which matches case-insensitively. Do not put credentials in
JSON, shell history, access logs, or the repo. Do not share `nginx -T` output:
it dumps included files, including the token map. The deployment command does
not print the configuration dump.

After deployment, call `https://inference.example.internal/v1/chat/completions`
from a VPN client, using the trusted CA (do not skip certificate verification).
With the same streaming payload, check for `401` with a missing or incorrect
`Authorization` header and incremental events plus `[DONE]` with
`Authorization: Bearer <token>`. NGINX strips that header before proxying.
TLS and the shared token are operator-managed; reload after changing either.
This is not per-user authentication or automatic credential rotation.

See the [gateway credentials design](gateway-credentials-design.md) for
the proposed token management workflow and deferred certificate automation.

## Verify the gateway after deployment

From a VPN client, load the bearer token into the `API_TOKEN` environment
variable using your secret manager. Then run:

```bash
python3 -m inference_platform gateway check --profile gateway-https.json \
  --ca-file gateway-lab.pem --model Qwen/Qwen3-8B
```

Use the model ID your backend serves. Omit `--ca-file` if the system already
trusts the issuer. The check uses the hostname and port in the profile and
verifies the TLS certificate; there is no insecure mode. It first expects
`401` without credentials, then checks that a small authenticated generation
streams text and reaches `[DONE]`. The default deadline for both requests is
30 seconds; override it with `--timeout`. Run this standalone command on
macOS or Linux. It does not reload or change the gateway.

## Gateway request timings

The managed gateway writes a timing-only access log for
`/v1/chat/completions`. After deploying or redeploying the profile, watch it
on the NGINX host:

```bash
sudo tail -f /var/log/nginx/inference_access.log
```

Entries are written when NGINX finishes handling the request. They include
the request ID, downstream and upstream status, the selected backend, and
the following times in seconds:

| Field | What it measures |
|---|---|
| `request_time` | NGINX's total request handling time, including streaming the response to the client. |
| `upstream_connect_time` | Time to establish the connection to vLLM. |
| `upstream_header_time` | Time until NGINX receives the upstream response headers. |
| `upstream_response_time` | Time spent receiving the upstream response, including the stream. |

Requests rejected locally (such as `401` or `429`) have no upstream timings.
This log does not include credentials, URL query strings, or request bodies.
Upstream headers can arrive before generated text, so `upstream_header_time`
is not TTFT. Use the client timing test for first text and first answer,
and compare repeated requests on the same path before attributing a latency
change to NGINX, TLS, or vLLM.

## Backend pool

Edit `backends` in your gateway profile, then run `gateway deploy`
again. For example, on a separate gateway host with two
backend ports forwarded to its loopback interface:

```json
[
  {"host": "127.0.0.1", "port": 8101},
  {"host": "127.0.0.1", "port": 8102}
]
```

Each address is reached **from the NGINX host**. For this example, establish
one SSH local forward per GPU host (e.g. `-L 127.0.0.1:8101:127.0.0.1:8000`
for the first one). Alternatively, use private backend addresses only if they
have a listener reachable by NGINX over a restricted network path. The current
vLLM deployment publishes only to each GPU host's loopback by default, so
substituting its private IP alone will not work unless a private binding is configured.

Use replicas serving the same client-visible model ID and compatible API
settings. `least_conn` favors the backend with fewer active connections; it
does not read vLLM queue or GPU metrics. The `max_in_flight` value is **shared
across the pool**: it defaults to `1` as in the smoke test. To exercise two
overlapping streams, set it to `2` on a dedicated test setup; choose the
operating limit from capacity measurements. This is static routing, not
service discovery or tested failover. Automatic upstream retries are off;
an interrupted stream is not replayed.
