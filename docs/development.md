# Development and findings

[Back to quick start](../README.md)

## Local checks

```bash
make test
make check
```

The offline test suite does not contact remote hosts.

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
