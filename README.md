# Inference Platform

Deploy and manage a vLLM model on a GPU host over SSH. Optionally add an
authenticated HTTPS streaming gateway and scrape metrics from a separate
Prometheus host.

## Quick start

Run these commands from the repo root with Python 3.10+ and OpenSSH. Use a
dedicated Ubuntu 22.04 or 24.04 x86_64 GPU host with a working NVIDIA driver,
Python 3, and passwordless sudo for preparation. Verify the SSH host key
before connecting.

Set `GPU_HOST` to your SSH target. Review [`deployment.json`](deployment.json)
for the model, image, GPU assignment and resource settings before deploying.

```bash
GPU_HOST=ubuntu@gpu.example
python3 -m inference_platform inspect --host "$GPU_HOST"
python3 -m inference_platform prepare --host "$GPU_HOST"  # only if setup is needed
python3 -m inference_platform deploy --host "$GPU_HOST" --profile deployment.json
```

By default, the vLLM API is unauthenticated and reachable only on the GPU
host's loopback interface. To check it from your workstation, open a tunnel
(adjust `8000` if you changed the profile port):

```bash
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 "$GPU_HOST"
```

In another terminal, run `curl -fsS http://127.0.0.1:8000/v1/models`.

## Guides

| Guide | What it covers |
|---|---|
| [Infrastructure](docs/infra.md) | AWS and Vast instance lifecycle and target handoff |
| [Deployment](docs/deployment.md) | Host preparation, profile settings, container lifecycle and troubleshooting |
| [NGINX gateway](docs/gateway.md) | Guided setup, HTTPS and bearer auth, streaming verification, timing logs and backend pools |
| [Prometheus monitoring](docs/monitoring.md) | Private metrics access, target registration via SSH and EC2 discovery |
| [Development and findings](docs/development.md) | Tests, module boundaries and results from earlier experiments |
