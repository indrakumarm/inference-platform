# Prometheus monitoring

[Back to quick start](../README.md)

## Private metrics access

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

## File-based target discovery

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

## EC2 discovery

EC2 tag-based discovery remains independent: omit `--monitoring-host` for
hosts covered by EC2 discovery. A monitoring server can use both discovery
modes for different hosts; avoid registering the same endpoint in both jobs.
