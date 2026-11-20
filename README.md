# Inference Platform

An incremental inference-serving project, separate from the sibling
[inference lab](../inference-lab/README.md). The lab explains and measures engine
behavior; this project will turn that knowledge into an operable endpoint.
No chat history, agent harness or session management is planned here.

## Current slices: prepare a host, deploy one model

```text
inspect -> explicit confirmation -> optional host preparation -> GPU verification
   |
deployment profile -> preflight -> explicit confirmation -> pinned vLLM container
                                                -> model identity -> generation -> readiness
```

### Host inspection and preparation

- SSH inspection of OS/architecture, disk, NVIDIA GPUs/driver, compute processes,
  rootful Docker, runtime registration, containers and administrative access.
- Human-readable report or JSON, with probe failures and actionable blockers.
- Explicit interactive confirmation before installing or changing anything.
- Ubuntu 22.04/24.04 x86_64 preparation: Docker (`docker.io` if absent),
  NVIDIA Container Toolkit from NVIDIA's signed stable apt repository, runtime
  configuration and Docker restart when needed.
- Real GPU-container verification after confirmation, followed by reinspection.

### Single-model deployment

- A strict JSON profile pins the official vLLM image by digest and model and
  tokenizer by immutable 40-character Hugging Face revisions. GPU indices,
  minimum free VRAM/disk thresholds, context length, dtype, utilization, shared
  memory size and local endpoint port are explicit.
- Read-only preflight requires rootful Docker, the assigned GPUs to meet the
  configured free-memory threshold, no GPU compute processes, no running
  containers, enough free root-disk space and an available loopback port.
- Deployment is interactive and rechecks the host before starting. It tests
  assigned-GPU container passthrough, downloads the pinned image and public
  model, then starts a persistent container with a reusable Hugging Face cache.
- Readiness requires the expected model identity from `/v1/models` and a
  non-empty generation from `/v1/completions`.

### Container lifecycle

- `list` inventories running and stopped inference-platform containers on the
  host; `list --json` returns the same inventory as JSON.
- `status` reports only the container selected by the deployment profile.
- `start` resumes that stopped container after rechecking GPU, port and host
  workload readiness; `stop` gracefully stops only that container.
- `remove` deletes only the stopped container. It does not delete the vLLM
  image or the shared Hugging Face cache volume. New deployments carry labels
  for profile/model metadata; older `inference-*` containers are included by
  name and shown without metadata they do not have.
- `start`, `stop`, and `remove` prompt for interactive confirmation. Enter `y`
  or `yes` to proceed; pressing Enter cancels. A running container must be
  stopped before it can be removed.

By default the endpoint binds only to `127.0.0.1` on the inference host and
has no API authentication. Use an SSH tunnel for direct operator access.
For an explicitly trusted monitoring host, see [private metrics collection](#private-metrics-collection).
For example, from the operator machine:

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@10.0.1.10
```

**Not implemented:** NGINX package installation, certificate issuance and rotation automation,
predictive model-fit validation, secrets/private model authentication,
monitoring, autoscaling, result upload or cloud provisioning. The CLI does not
install drivers, reboot, stop existing workloads or open firewall ports.

## NGINX gateway

The [gateway config](gateway/nginx/inference.conf) is separate from the vLLM
deployment profile. By default it listens on `127.0.0.1:8080` on the NGINX host
and streams `POST /v1/chat/completions` to vLLM on `127.0.0.1:8000` on that
same host. The default profile is for operator testing over SSH and does not
enable TLS or authentication. Use the HTTPS profile below for VPN clients.

From the repo root on your workstation, render the config from
[`gateway.json`](gateway.json):

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

### HTTPS and bearer authentication for VPN clients

Copy `gateway.json` to a separate profile. Keep the existing backend settings
and change `listen` and add `tls`, for example:

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
manager. From the repo root, use the HTTPS profile saved as `gateway-https.json`:

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

See the [gateway credentials design](docs/gateway-credentials-design.md) for
the proposed token management workflow and deferred certificate automation.

### Verify the gateway after deployment

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

### Gateway request timings

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

### Backend pool

Edit `backends` in [`gateway.json`](gateway.json), then run `gateway deploy`
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

## Requirements

Operator machine: Python 3.10+ and OpenSSH client. No third-party Python packages
are required. Run the following from this project's directory; installation is
optional (`python3 -m pip install -e .`).

Host: Linux with Python 3, SSH access and a trusted SSH host key already in
`known_hosts`. For automatic preparation, use a **dedicated** Ubuntu 22.04/24.04
x86_64 systemd host with a functioning NVIDIA driver and root or passwordless
sudo. Inspection reports missing prerequisites; unsupported hosts can still be
inspected if their Python and `/etc/os-release` are available.

First establish SSH connectivity and verify the host fingerprint using a trusted
source. Host-key checking is never disabled. Prefer SSH agent/config over copying
keys; use `--key` only to select an existing private key.
SSH connection setup times out after 10 seconds; each operation also has an
overall timeout. Failures report the host, operation, and useful SSH/remote
stderr without dumping the embedded remote program into the error message.

```bash
python3 -m inference_platform inspect --host ubuntu@10.0.1.10 --key ~/.ssh/lab.pem
python3 -m inference_platform inspect --host ubuntu@10.0.1.10 --json

# Inspects again, prints the plan, then asks you to type "prepare ubuntu@10.0.1.10":
python3 -m inference_platform prepare --host ubuntu@10.0.1.10 --key ~/.ssh/lab.pem

# After successful preparation, deployment requires a reviewed JSON profile:
python3 -m inference_platform deploy --host ubuntu@10.0.1.10 --profile deployment.json

# Manage the one deployment named by deployment.json:
python3 -m inference_platform list --host ubuntu@10.0.1.10
python3 -m inference_platform list --host ubuntu@10.0.1.10 --json
python3 -m inference_platform status --host ubuntu@10.0.1.10 --profile deployment.json
python3 -m inference_platform stop --host ubuntu@10.0.1.10 --profile deployment.json
python3 -m inference_platform start --host ubuntu@10.0.1.10 --profile deployment.json
python3 -m inference_platform remove --host ubuntu@10.0.1.10 --profile deployment.json
```

`inspect`, `list` and `status` never install packages, pull images or mutate
containers. `list` reports Docker's current state, not deployment history.
Lifecycle commands target only the container derived from the profile name.
`prepare`, `deploy`, `start`, `stop` and `remove` require a terminal; there is
deliberately no unattended `--yes`.
Exit codes: 0 inspection completed without blockers / preparation succeeded or
deployment succeeded or was cancelled; 2 readiness/confirmation blocked;
1 execution error; 130 interruption.
An inspection without blockers is **eligible for preparation**, not proof of
GPU passthrough. Only a successful `prepare` verifies that with a container.

### Deployment profile

Create a JSON file with these required fields (replace placeholders with values
you have verified):

```json
{
  "name": "my-model",
  "image": "vllm/vllm-openai@sha256:<64-hex-image-digest>",
  "model": "organization/model",
  "model_revision": "<40-hex-commit>",
  "tokenizer": "organization/model",
  "tokenizer_revision": "<40-hex-commit>",
  "gpu_indices": [0],
  "min_gpu_memory_gib": 20,
  "min_free_disk_gib": 80,
  "max_model_len": 4096,
  "dtype": "auto",
  "gpu_memory_utilization": 0.9,
  "port": 8000,
  "shm_size_gib": 8
}
```

Only public Hugging Face repositories are supported; do not put credentials in
the profile. Use the official vLLM image repository and verify the image digest
and model/tokenizer commit hashes before deployment. `min_gpu_memory_gib` is
checked separately on every assigned GPU. Both `deploy` and `start` check the
space available on the remote host's `/` filesystem against `min_free_disk_gib`
and report both values if the check fails. This threshold is not an estimate of
model/image download size. Successful model load
and generation demonstrate that this exact deployment starts, but do not
replace a workload-specific capacity/performance evaluation.

### Private metrics collection

For direct scraping from a separate Prometheus host, optionally add
`"private_bind_ip": "10.0.1.10"` to the deployment profile, using the GPU
host's actual RFC 1918 IPv4 address. The platform checks both bindings before
deployment and publishes the same vLLM port on that private address and on
loopback. Omit the field or set it to `null` to keep loopback only. Changing
the binding requires recreating the container; `stop`/`start` alone does not
change its Docker port mappings.

**Before deploying, restrict inbound access to the vLLM port to the dedicated
monitoring security group.** Check all security groups attached to the GPU
host; allow rules are additive. The private binding exposes the entire
unauthenticated vLLM API, not just `/metrics`, to hosts that can reach it.
Do not allow this port from the general VPN or use it for untrusted clients.

For file-based discovery, create `/etc/prometheus/targets/inference-platform`
on the **monitoring host**. Give a dedicated SSH user write access to that
directory and give Prometheus read and traversal access. The SSH user needs
Python 3, not sudo. Configure Prometheus once to watch the directory:

```yaml
scrape_configs:
  - job_name: vllm
    metrics_path: /metrics
    file_sd_configs:
      - files: [/etc/prometheus/targets/inference-platform/vllm-*.json]
```

With `private_bind_ip` set in the deployment profile, opt in using
`--monitoring-host`. Add `--monitoring-key` if the monitoring host needs a
specific key. The monitoring connection uses SSH port 22.

```bash
python3 -m inference_platform deploy --host ubuntu@gpu.example \
  --profile deployment.json --monitoring-host metrics@monitor.example
python3 -m inference_platform remove --host ubuntu@gpu.example \
  --profile deployment.json --monitoring-host metrics@monitor.example
```

`deploy` writes one target file atomically after model readiness checks pass.
`remove` deletes that file after the container is removed. `stop` leaves the
target registered so Prometheus can report `up=0`. Prometheus picks up the
file changes without reloading. Check the Prometheus Targets page to verify
scrape reachability; registration does not test the network path.

Use the same monitoring host and the original deployment name, private IP,
and port when removing a deployment. If the IP changes, remove the old
deployment with its original profile and deploy using the new address. If
monitoring registration fails, the model remains deployed; correct the target
on the monitoring host without recreating the model. If cleanup fails after
container removal, rerun `remove` with the same settings to retry cleanup.

EC2 tag-based discovery remains independent: omit `--monitoring-host` for
hosts covered by EC2 discovery. A monitoring server can use both discovery
modes for different hosts; avoid registering the same endpoint in both jobs.

## Preparation safety and limitations

- Inspect the report before agreeing. Inspections run in sequence, not as an
  atomic host snapshot. The plan is checked again after confirmation; workload
  checks are repeated before host-wide mutations. Other operators can still
  race these checks: **do not prepare a shared host**.
- Active GPU compute processes block preparation. Running containers block
  Docker/runtime changes. Rootful local Docker is the only supported runtime;
  rootless/remote Docker configurations are not managed.
- NVIDIA driver failures need operator remediation first. Installing a host
  CUDA development toolkit is not required: the container carries CUDA
  user-space libraries; the host needs a compatible driver.
- Preparation needs apt/repository access; verification may pull
  `nvidia/cuda:12.4.1-base-ubuntu22.04`. Docker packages and toolkit follow the
  configured apt repositories, not a fully locked package manifest.
- Existing Docker daemon configuration is backed up before runtime changes.
  Operations are not transactional: failed installs/configuration are reported,
  not silently rolled back. Inspect before retrying. No rollback is promised.
- GPU verification proves a CUDA utility container can see GPUs. Deployment
  separately checks the assigned devices with that utility image before vLLM.
- At least 5 GiB free on `/` is required for preparation. Deployment uses the
  profile's free-disk threshold; it does not predict model/image size or KV fit.
- Reports include host/GPU/process details. Keep saved reports private. Never
  place SSH private keys, API keys or Hugging Face tokens in project config.
- Deployment is for a dedicated host only; any existing GPU compute process or
  running container blocks it. Starting a stopped deployment repeats those
  workload checks. Host checks cannot eliminate races with other operators.
  Failed deployments leave the container/cache for diagnosis rather than
  attempting a potentially destructive rollback.
- Host preparation never stops existing workloads. The explicit `stop` command
  only stops the container named by the selected profile.
- The endpoint is unauthenticated. Any local host user/process
  can connect; use a trusted dedicated host and SSH access controls.

Toolkit setup follows the [NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
No remote hosts are contacted by the offline test suite.

## Logging and troubleshooting

The CLI and remote operations use the standard-library logger. Set
`INFERENCE_PLATFORM_LOG_LEVEL` to `INFO` for operation milestones or `DEBUG` for
probe/command details; the default is `WARNING`. Log records go to stderr and
include timestamps, severity and the module name, so normal reports and JSON
output remain on stdout.

```bash
INFERENCE_PLATFORM_LOG_LEVEL=INFO python3 -m inference_platform inspect --host ubuntu@10.0.1.10
INFERENCE_PLATFORM_LOG_LEVEL=DEBUG python3 -m inference_platform deploy --host ubuntu@10.0.1.10 --profile deployment.json
```

The same level is forwarded to the remote Python payloads. Treat debug logs as
operationally sensitive: they include host details and command arguments, but
never put credentials in the deployment profile or shell arguments.
Deployment command output is streamed to stderr; at `INFO`, model startup logs
are sampled periodically while readiness checks wait for the API.

## Extension boundaries

- `ssh.py`: transport, not Ubuntu/package or model logic.
- `remote.py`: self-contained host probes and bounded preparation operations.
- `deployment.py`: profile validation and local preflight policy.
- `remote_deploy.py`: standalone remote Docker deployment and readiness checks.
- `ubuntu.py`: support policy and explicit action planning.
- `cli.py`: report, approval and workflow orchestration.

Future OS adapters can supply a planner and remote preparation operations while
retaining the inspection/report/confirmation workflow. Do not add a generic
plugin framework until a second implementation needs it.

Next slices:

1. Predictive fit checks and model-specific capacity/performance validation.
2. Persistent serving gateway: auth, admission limits, streaming/cancellation,
   deadlines, metrics and structured logs.
3. Replicas, draining, updates and failure exercises; then SLO/cost validation.

The gateway can reuse proven streaming/routing mechanisms from the lab, but
deployment must not depend on the lab's temporary benchmark container lifecycle.

## Phase findings

Record each completed phase under [`docs/findings/`](docs/findings/), with the
test setup, results, what they demonstrate, and what remains untested.

- [Phase 1: Direct vLLM and NGINX streaming](docs/findings/01-vllm-nginx-streaming.md) —
  manual streaming checks and client latency through both paths.
- [Phase 2: Backend failure and recovery](docs/findings/02-backend-failure-recovery.md) —
  requests during an outage, recovery without a proxy restart, and interrupted streams.
- [Phase 3: NGINX admission control](docs/findings/03-nginx-admission-control.md) —
  reject an overlapping request and accept a retry once the active stream finishes.

## Development

```bash
make test
make check
```
