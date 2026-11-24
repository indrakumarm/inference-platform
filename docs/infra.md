# Infrastructure CLI

[Back to quick start](../README.md)

Infrastructure and workloads have separate lifecycles in the same Python package:

- `inference_platform infra aws ...` creates, starts, stops and terminates EC2
  instances through boto3. It produces a target file.
- `inference_platform inspect/prepare/deploy/... --target ...` operates the
  inference workload over SSH.

There are no shell scripts or AWS CLI subprocesses. The Vast.ai infrastructure
adapter is described below.

## Vast.ai container infrastructure

The Vast adapter manages container instances using the Vast REST API. It does
not install Docker, CUDA or packages, and does not start vLLM. The separate
Vast workload adapter runs vLLM inside the instance; the NGINX gateway setup
does not currently support Vast containers.

Copy [infra-vast.example.json](../infra-vast.example.json) to a local file and
edit its GPU name, disk, SSH paths and deployment profile path. The profile's
pinned image is selected when the container is created; GPU count must match
the profile. The image must support Vast's SSH launch mode and contain Python,
vLLM and compatible CUDA runtime dependencies. Register your public SSH key
with Vast first. Supply `VAST_API_KEY` in the process environment or in `.env`
next to the config; the environment takes precedence. Do not put keys in JSON.

```bash
python3 -m inference_platform infra vastai search --config infra-vast.json
python3 -m inference_platform infra vastai up --config infra-vast.json --offer OFFER_ID
python3 -m inference_platform infra vastai status --config infra-vast.json
# Once running:
python3 -m inference_platform infra vastai resolve --config infra-vast.json --target targets/vast.json
# Verify and enroll the SSH host key using a trusted source before connecting:
python3 -m inference_platform inspect --target targets/vast.json
python3 -m inference_platform deploy --target targets/vast.json --profile deployment.json
python3 -m inference_platform stop --target targets/vast.json --profile deployment.json
# These stop/start the whole instance, not only the model:
python3 -m inference_platform infra vastai stop --config infra-vast.json
python3 -m inference_platform infra vastai start --config infra-vast.json
python3 -m inference_platform infra vastai down --config infra-vast.json
```

`search` lists up to 50 verified, rentable on-demand offers matching the config.
`up` requires a selected offer and interactive confirmation, then requests an
SSH proxy container without publishing a vLLM port. Creation is asynchronous;
wait for `running` before `resolve`. Mutations require interactive
confirmation. `stop` affects the entire instance; `down` destroys it and its
instance data. Review current Vast pricing, including storage charges, before
creating or stopping an instance.

Local state under `~/.local/state/inference-platform/infra/vastai` records a
unique label and instance ID. It is not a distributed lock: use one operator
and state directory. If creation times out or returns no ID, the outcome is
unknown and creation is blocked rather than retried. Check the Vast console
for the saved label and reconcile the state manually before proceeding; never
delete state blindly. Errors after a destroy request also require checking the
console before reconciliation. Resources may still be billable after failures.

`resolve` exports a snapshot of Vast's reported SSH proxy endpoint. It never
trusts the host key automatically; verify and enroll the key through a trusted
channel before SSH. Custom target paths are not overwritten. The endpoint may
change after restart, so resolve again. Image digest identity, shared-memory
size, GPU readiness and vLLM execution are not verified by infra. The workload
adapter checks GPU access, available memory and shared memory before deployment,
but cannot independently verify the running image digest. See
[Vast workload deployment](deployment.md#vast-container-workloads).

## Install and configure

Python 3.10+ on Linux/macOS:

```bash
python3 -m pip install -e '.[aws]'
cp infra-aws.example.json infra-aws.json
```

The AWS SDK is optional. Existing deployment/gateway commands still need only
Python and OpenSSH. Authenticate using the standard boto3 credential chain
(for example an AWS profile or environment variables). Never put AWS
credentials, SSH private-key contents or model tokens in config.

Edit [infra-aws.example.json](../infra-aws.example.json) after copying it:

| Field | Meaning |
|---|---|
| `name`, `project` | Resource identity (`project` defaults to `inference-platform`) |
| `region`, `profile` | AWS region and optional credential profile |
| `instance_type` | GPU EC2 instance type, e.g. `g6.xlarge` |
| `ami_ssm` or `ami_id` | Exactly one: an SSM parameter resolving to an AMI ID, or an explicit AMI ID |
| `root_gb` | Root EBS volume size; must be at least the AMI minimum |
| `subnet_id`, `security_group_ids` | Existing networking resources; no firewall changes |
| `key_name` | Existing EC2 key pair |
| `key`, `known_hosts` | Optional local SSH paths for the target; default is SSH agent/config and default known_hosts |
| `ssh_user` | SSH login, default `ubuntu` |
| `market` | `ondemand` (default) or persistent `spot`; no automatic price/market fallback |
| `private_ip` | Prefer private IP when true; otherwise public IP with private fallback |
| `instance_profile` | Optional IAM instance profile name |
| `state_dir` | Optional local state root; default `~/.local/state/inference-platform/infra` |

Relative paths resolve against the config file's directory. Use a dedicated
Ubuntu 22.04/24.04 x86_64, EBS-backed GPU AMI with a functioning NVIDIA driver.
The example SSM parameter follows the earlier AWS lab setup; verify that it
exists in your region and resolves to the intended image. For reproducible
creation, use a reviewed `ami_id` instead of a moving `latest` parameter.

Infra checks the image's architecture/root volume, not its Ubuntu or driver
readiness. It does not install drivers, run cloud-init scripts, sync the lab
repository, configure idle shutdown or deploy a model. The existing `inspect`
and `prepare` workflow handles runtime readiness.

The AWS principal needs STS caller identity and the relevant EC2 describe,
launch, tag, start, stop, terminate and spot-cancellation permissions. SSM
parameter read permission is needed with `ami_ssm`, and IAM PassRole permission
with `instance_profile`. Account quotas, GPU/spot capacity and networking remain
operator prerequisites.

## Create, then hand off

```bash
# Prints the launch proposal; requires typing "up lab-a".
python3 -m inference_platform infra aws up --config infra-aws.json \
  --target targets/lab-a.json

python3 -m inference_platform infra aws status --config infra-aws.json
python3 -m inference_platform infra aws status --config infra-aws.json --json

# After verifying and trusting the SSH host key:
python3 -m inference_platform inspect --target targets/lab-a.json
python3 -m inference_platform prepare --target targets/lab-a.json
python3 -m inference_platform deploy --target targets/lab-a.json --profile deployment.json
```

`up` creates one instance if none exists, or resumes the existing owned instance.
It does not replace an existing instance to apply config changes. EC2 reaching
`running` is not proof of SSH accessibility, cloud-init completion or GPU
readiness. Verify host-key fingerprints using a trusted source (for example
the instance's AWS console output) and enroll them in the selected known_hosts
file before the deployment CLI connects. Infra never uses `accept-new`,
disables key checking, or scans/enrolls host keys automatically.

The target's workload `provider` is **`ssh`**, not `aws`: EC2 provisioning and
Ubuntu/Docker workload execution are different adapters. Exported targets have
the same schema described in [deployment](deployment.md#providers-and-targets).
No private key contents or AWS credentials are exported.

If `--target` is omitted, infra prints the path of its managed target under the
state directory. `resolve` refreshes target details without mutating AWS:

```bash
python3 -m inference_platform infra aws resolve --config infra-aws.json \
  --target targets/lab-a.json
python3 -m inference_platform infra aws resolve --config infra-aws.json --json
```

`resolve --json` writes the managed target and emits the target object on stdout,
so it can also be redirected to a target file.

## Machine lifecycle versus model lifecycle

```bash
# Stops only vLLM; the EC2 machine keeps running and compute charges continue.
python3 -m inference_platform stop --target targets/lab-a.json --profile deployment.json

# Stops the entire EC2 machine and all its workloads; storage remains billable.
python3 -m inference_platform infra aws stop --config infra-aws.json

# Starts the machine; its public IP may change. Re-exports the target.
python3 -m inference_platform infra aws start --config infra-aws.json \
  --target targets/lab-a.json

# Terminates the machine, deleting its root volume/model cache.
python3 -m inference_platform infra aws down --config infra-aws.json
```

Every machine mutation requires a terminal and the exact phrase
`<operation> <name>`; there is no `--yes`. State/identity is rechecked after
confirmation. Changing the machine's state can interrupt every workload on it.
For other attached volumes, deletion follows their AWS DeleteOnTermination
settings; networking, key pairs and IAM resources are not removed.

Persistent spot requests are cancelled before termination to prevent relaunch.
`stop` retains the request so it can resume. Capacity errors never fall back
silently to on-demand. Confirm costs and AWS restrictions for your chosen market.

Stopping/terminating invalidates infra's managed target. Custom exported files
are snapshots and are not deleted: refresh them with `start` or `resolve`
before use. Infra does not update gateway or Prometheus configuration when
addresses change.

## State, failures and retries

Resources are selected by `Name`, `Project`, and `inference-platform:managed=true`
tags in the authenticated account/region. Untagged instances from the old lab
script are **not adopted**. Duplicate matches, recorded-ID mismatches and
changed ownership tags block operations.

Local state is separated by AWS account, region, project and name. Atomic writes
and a per-resource file lock prevent conflicting operations on the same machine.
This is not a distributed lock: do not concurrently manage the same name from
multiple operator machines or independent state directories.

Before a launch call, infra persists its exact request and an EC2 idempotency
token. If creation's result is uncertain, `up` retries that saved request,
including the original AMI and disk settings, instead of generating a new
request from changed config. Once an ID is returned it is saved before waiting.
Do not delete state to retry a failed launch: first inspect AWS to avoid creating
duplicates. Missing or corrupted state, lost tags and externally removed
resources need operator reconciliation.

EC2 transitions are awaited for approximately ten minutes, with bounded SDK
connect/read timeouts and retries. Failures/interruption do not automatically
terminate the instance or roll back changes. **Resources may remain billable**;
run `status` and check AWS before retrying. Externally terminated persistent
spot instances require `down` to cancel the old request before a fresh `up`.

Exit codes follow deployment conventions: 0 success/cancellation, 2 confirmation
blocked, 1 config/API/execution failure, 130 interruption. Standard logging goes
to stderr; set `INFERENCE_PLATFORM_LOG_LEVEL=INFO` for milestones. Treat config,
state, targets and status output as operationally sensitive.
