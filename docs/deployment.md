# GPU host deployment

[Back to quick start](../README.md)

## Requirements

Operator machine: Python 3.10+ and OpenSSH client. No third-party Python packages
are required. Run the following from this project's directory; installation is
optional (`python3 -m pip install -e .`).

For the default `ssh` provider, the host requires Linux with Python 3, SSH
access and a trusted SSH host key already in
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
GPU passthrough. Both `prepare` and `deploy` verify passthrough with a container.

## Providers and targets

Every command runs against a **provider**: an adapter that knows how a kind of
GPU host is inspected, prepared and how vLLM runs on it. The default provider
is `ssh`: a dedicated Ubuntu host you reach over SSH, running vLLM in rootful
Docker. The `vastai` provider runs vLLM as a process in a Vast container.

The connection can be given either with flags (`--host`, `--port`, `--key`,
`--known-hosts`) or as a JSON **target** file. The target file is the handoff
from whatever created the machine (a cloud script, a provider CLI) to this tool:

For this repository's Python AWS provisioning commands, see
[Infrastructure CLI](infra.md). Infra creates the machine; the commands here
continue to manage only host readiness and the inference workload.

```json
{
  "provider": "ssh",
  "host": "ubuntu@10.0.1.10",
  "port": 22,
  "key": "~/.ssh/lab.pem",
  "known_hosts": ".known_hosts/lab-a"
}
```

Only `host` is required. Relative `key`/`known_hosts` paths resolve against
the target file's directory. `--target` cannot be combined with the individual
connection flags.

```bash
python3 -m inference_platform inspect --target targets/lab-a.json
python3 -m inference_platform deploy --target targets/lab-a.json --profile deployment.json
```

`known_hosts` (or `--known-hosts`) selects the file SSH checks the host key
against; host-key checking itself is never relaxed. Use it when a per-instance
known_hosts file was recorded at provisioning time instead of
`~/.ssh/known_hosts`.

## Vast container workloads

After Vast reports the instance running, export its target as described in
[Infrastructure CLI](infra.md#vastai-container-infrastructure). Verify its
SSH host key through a trusted channel before connecting. The Vast instance
must already contain Python 3, PyTorch with working CUDA, `nvidia-smi`, and the
`vllm` executable. No Docker daemon, systemd, sudo or package installation is
required inside the container.

```bash
python3 -m inference_platform inspect --target targets/vast.json
python3 -m inference_platform deploy --target targets/vast.json --profile deployment.json
python3 -m inference_platform list --target targets/vast.json
python3 -m inference_platform status --target targets/vast.json --profile deployment.json
python3 -m inference_platform stop --target targets/vast.json --profile deployment.json
python3 -m inference_platform start --target targets/vast.json --profile deployment.json
python3 -m inference_platform remove --target targets/vast.json --profile deployment.json
```

`prepare` checks prerequisites but makes no changes. Deployment checks visible
GPUs through PyTorch, free GPU memory, idle compute processes, free Hugging
Face cache filesystem space, available `/dev/shm` and loopback port availability. Vast
assigns shared memory at provisioning; choose an appropriate instance or
adjust the profile only after checking model requirements. The workload image
was chosen at instance creation: the adapter **cannot verify the running image
digest from inside the container**. Compare the provider's instance image
with the profile before confirming deployment.

The adapter starts a detached vLLM process and checks both model identity and
generation. The endpoint binds only to `127.0.0.1`; use an SSH tunnel, not a
public Vast port mapping. `private_bind_ip`, Prometheus registration and
gateway setup are not supported for Vast workloads. Process start and restart
are manual; stopping or restarting the instance never automatically restarts
the model. `stop` and `remove` above affect only the model process and record,
not the billable instance. Remove retains the model cache and logs.

State and logs live under `~/.local/state/inference-platform/vast/<name>/`
inside the container. `vllm.log` contains model output and `supervisor.log`
contains supervisor failures. Use the original deployment profile for
lifecycle commands. If status is `unknown` or `orphaned`, the adapter refuses
to signal or remove the process: inspect the logs and processes manually before
reconciling. Process signalling requires Linux pidfd support. Local locks cover
only this state directory, not other users or external processes; use a
dedicated container and avoid concurrent external workload changes. `list
--json` retains the existing `container` field for compatibility, but it names
a process deployment for Vast.

## Deployment profile

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

## How deployment works

The following describes the default Ubuntu/Docker `ssh` provider. For Vast,
see [Vast container workloads](#vast-container-workloads).

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
For an explicitly trusted monitoring host, see [Prometheus monitoring](monitoring.md).
For example, from the operator machine:

```bash
ssh -L 8000:127.0.0.1:8000 ubuntu@10.0.1.10
```

The deployment CLI does not install drivers, reboot the host, stop existing
workloads or open firewall ports. It does not provision cloud hosts, autoscale,
predict model fit, upload results, or authenticate to private model repositories.

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
