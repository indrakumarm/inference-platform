# Development and findings

[Back to quick start](../README.md)

## Local checks

```bash
make test
make check
```

The offline test suite does not contact remote hosts.

## Extension boundaries

- `ssh.py`: SSH transport (`SSHHost.run`) plus the small monitoring/gateway
  payload helpers for hosts that are not managed by a provider.
- `providers/base.py`: the `Target` handoff contract and the `Provider`
  interface the CLI drives: `inspect`, `plan`, `prepare`, `deployment_inspect`,
  `deployment_blockers`, `deploy`, `lifecycle`, `deployment_list`.
- `providers/ssh.py`: the default `ssh` provider; composes the Ubuntu planner
  with the Docker runtime payloads (`remote.py`, `remote_deploy.py`).
- `remote.py`: self-contained host probes and bounded preparation operations.
- `deployment.py`: profile validation and local preflight policy.
- `remote_deploy.py`: standalone remote Docker deployment and readiness checks.
- `remote_vast.py`, `remote_vast_supervisor.py`: Vast container preflight and
  detached, manually restarted vLLM process lifecycle.
- `ubuntu.py`: support policy and explicit action planning.
- `cli.py`: report, approval and workflow orchestration; provider-agnostic.
- `platform_config.py`: reusable configuration parsing and data-only `.env`
  token loading for the gateway; independent of command-line prompts.
- `gateway_setup.py`: gateway orchestration and interactive confirmation.
- `infra/cli.py`: separately namespaced machine lifecycle and confirmation.
- `infra/aws.py`, `infra/vast.py`: EC2 and Vast instance adapters and target export.
- `infra/config.py`, `infra/state.py`: infra config, atomic state and local locking.

A new provider adds one module under `providers/`, registers itself in
`providers/__init__.py`, and supplies its own planner and remote payloads.
Reuse the Docker runtime payloads where the host has a Docker daemon; use a
separate remote payload for hosts where vLLM must run as a plain process.
Infrastructure creation (cloud instances) stays outside these commands: it
produces a target file, and the CLI takes over from there.
The AWS SDK lives in the optional `aws` extra, never in the SSH workload provider.
Run the full offline SDK-contract suite with `python3 -m pip install -e '.[aws]'`
followed by `make test`; SDK-specific tests skip when the extra is absent.
The top-level platform config currently drives gateway setup/verification only;
infra and deployment retain their existing CLI configuration. Gateway client
verification uses process-level SIGALRM and must be adapted before use in an
API server worker thread.

## Future work

- Predictive fit checks and model-specific capacity and performance validation.
- Replica management, draining, updates, and broader failure exercises.
- SLO and cost validation.

The sibling inference lab measures engine behavior. This project focuses on
operating a serving endpoint, without a chat history, agent harness, or session
management layer. Deployment does not depend on the lab's temporary benchmark
container lifecycle.

## Phase findings

Record each completed phase under [`docs/findings/`](findings/), with the
test setup, results, what they demonstrate, and what remains untested.

- [Phase 1: Direct vLLM and NGINX streaming](findings/01-vllm-nginx-streaming.md) —
  manual streaming checks and client latency through both paths.
- [Phase 2: Backend failure and recovery](findings/02-backend-failure-recovery.md) —
  requests during an outage, recovery without a proxy restart, and interrupted streams.
- [Phase 3: NGINX admission control](findings/03-nginx-admission-control.md) —
  reject an overlapping request and accept a retry once the active stream finishes.
