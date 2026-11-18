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
- State-changing lifecycle actions require typed confirmation. A running
  container must be stopped before it can be removed.

The endpoint binds to `127.0.0.1` on the inference host; it has no API
authentication. Access it through an SSH tunnel, not by opening a firewall port.
For example, from the operator machine:

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@10.0.1.10
```

**Not implemented:** predictive model-fit validation, secrets/private model
authentication, gateway, TLS/auth, monitoring, autoscaling, result upload or
cloud provisioning. The code does not install drivers, reboot, stop existing
workloads or open firewall ports.

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

Create a JSON file with exactly these fields (replace placeholders with values
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
checked separately on every assigned GPU; `min_free_disk_gib` is a preflight
threshold, not an estimate of model/image download size. Successful model load
and generation demonstrate that this exact deployment starts, but do not
replace a workload-specific capacity/performance evaluation.

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
- The endpoint is unauthenticated on host loopback. Any local host user/process
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

## Development

```bash
make test
make check
```
