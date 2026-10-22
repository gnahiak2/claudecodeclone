# relay-code

`relay` is a three-tier coding agent written in Rust. It separates thinking,
editing and verification so that a model never grades its own work.

- **Tier 1 — planner (read-only).** Inspects the repository with `list_files`,
  `read_file` and `search`, then returns a structured plan. It has no write,
  shell or network tools.
- **Tier 2 — executor (staged workspace).** A native Rust model/tool loop edits
  an isolated copy of the repository through hash-checked patches. It cannot
  run shell commands or reach the original checkout.
- **Tier 3 — evaluator (Docker).** Runs the approved test argv inside a
  hardened container against a fresh copy of the staged snapshot. The verdict
  is deterministic process exit status, not a model opinion.

The harness owns approvals, budgets, sessions, evidence and the final apply. It
supports local Ollama and hosted OpenAI, Anthropic, Cohere and OpenAI-compatible
endpoints.

## Requirements

- Rust 1.88 or newer (edition 2024; let-chains are used).
- For evaluation: Docker (or a compatible runtime), a running daemon, and the
  test images already pulled.
- For local models: Ollama running at `http://localhost:11434`.

Hosted providers need valid credentials and may charge money. There is no
payment bypass, and no provider is contacted without a configured key.

## Build

```sh
cargo build --release
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

If Cargo fails TLS verification because the system certificate path is not a
PEM bundle, pass a real bundle per command instead of disabling verification:

```sh
CARGO_HTTP_CAINFO=/path/to/cert.pem SSL_CERT_FILE=/path/to/cert.pem cargo test
```

## Quick start with Ollama

```sh
cp config.example.toml relay.toml
ollama pull qwen2.5-coder:7b
docker pull rust:1.88-bookworm   # or change the test image
./target/release/relay plan "add a --verbose flag to the CLI"
./target/release/relay run "add a --verbose flag to the CLI"
```

`plan` is strictly read-only. `run` shows the plan, the evaluation policy and
the final diff and asks for approval at each gate. `chat` is a read-only
conversation about the repository.

Every session is stored under the platform state directory (or
`harness.sessions_dir`) with private permissions:

```sh
relay session inspect ses_...
relay session resume ses_... --approve-plan
```

## Commands

| Command | Behaviour |
| --- | --- |
| `relay providers` | Prints role models, endpoints and whether key env vars are set. Never prints values. |
| `relay plan <task>` | Read-only planning; prints the JSON-derived plan and usage. |
| `relay chat` | Read-only tool-assisted Q&A loop. |
| `relay run <task>` | Full plan → approve → stage → execute → evaluate → repair → review → approve apply flow. |
| `relay session inspect <id>` | Phase, budgets, plan, evidence and recent redacted events. |
| `relay session resume <id>` | Resumes only from safe checkpoints; interrupted mutations are never replayed. |

Flag `--non-interactive` refuses any step that needs approval. `--approve-plan`,
`--approve-evaluation` and `--approve-apply` pre-approve that exact gate only.
Approval flags are never accepted from the model.

## Flow and approvals

1. **Plan.** Produced by the read-only tier and approved by you. Approval is
   bound to the plan hash.
2. **Stage.** The project is snapshotted into a private run directory. The
   snapshot refuses to run inside the project tree.
3. **Execute.** The executor may only write paths listed in the approved plan.
   Every edit needs the current BLAKE3 preimage hash. Missing or stale hashes
   are rejected.
4. **Evaluate.** You approve the exact image, argv, working directory, network
   flags and sandbox policy. Approvals are hashed; changing any of them
   invalidates the approval and any prior evidence.
5. **Repair.** At most `harness.max_repairs` (default 2) repair attempts, within
   run-wide model/tool/token budgets. Infrastructure failures are never treated
   as repairable test failures.
6. **Review.** The diff is shown with terminal control characters escaped.
7. **Apply.** Application requires all of: an approval bound to the current
   snapshot hash, passing evaluation evidence for that same snapshot, unchanged
   project preimages and modes, and unchanged staged bytes. The apply is
   journaled per operation; partial rollback failures are reported truthfully.

No command commits or pushes. No model can approve itself.

## Evaluation sandbox

For each profile the evaluator:

- copies the staged snapshot into a fresh directory under the private run;
- runs the container as `65534:65534` with a read-only root filesystem;
- drops all capabilities, sets `no-new-privileges`, limits CPUs, memory, PIDs,
  tmpfs size and captured output, and enforces a wall-clock timeout;
- disables the network by default (explicit per-profile approval enables it);
- mounts only the isolated copy at `/workspace`, never the project, home
  directory, Docker socket or credentials;
- passes only `HOME` and `TMPDIR`, never host provider secrets;
- removes containers on success, failure and timeout, and reports cleanup
  failures.

Images are never pulled implicitly. If the daemon or an approved image is
missing, evaluation fails closed. Generated tests are **never** run on the host.

## Configuration

See `config.example.toml`. Important fields:

- `harness.max_repairs`, `max_planner_calls`, `max_executor_calls`,
  `max_tool_calls`, `max_model_tokens`, `max_context_tokens`.
- `harness.sessions_dir` must be absolute and outside the project.
- `harness.redact_env` lists extra environment variables to redact.
- `docker.runtime` is limited to `docker`, `podman` or `nerdctl`; CPU, memory,
  PID, tmpfs and output bounds are validated.
- `[[tests]]` defines the approved test profiles. `allow_network` and
  `prepare_allow_network` default to false.
- `providers.*.trust_custom_endpoint` must be set before a key is sent to a
  non-default endpoint.
- `providers.*.tools = false` is rejected for roles, because both tiers require
  tool calling.

Custom endpoints may be unauthenticated (for example a local OpenAI-compatible
server); hosted providers still require their key environment variable.

## Security model

Implemented protections include:

- capability-relative filesystem operations (`openat` with `O_NOFOLLOW`,
  `renameat`, `unlinkat`) for workspace reads, writes and removals;
- private run directories, exclusive session locks, redacted append-only
  events and atomic state writes;
- authoritative snapshot manifests independent of editable ignore rules;
  tracked-but-ignored files are preserved and ignore edits cannot hide staged
  changes;
- bounded reads, traversal limits, regex size limits and bounded provider
  response bodies;
- redirects disabled for provider requests so credentials cannot be forwarded;
  credential headers marked sensitive; TLS verification left enabled;
- rejection of truncated or filtered provider responses before any tool call,
  and rejection of non-object tool arguments;
- terminal control-character escaping at display boundaries (patch bytes are
  never altered);
- snapshot source reads through the same no-follow handle, with source changes
  during copying causing a hard failure instead of a partial baseline.

### Threat model and limitations

Containers reduce risk but are not a perfect boundary for hostile workloads.
Assumptions:

- The Docker daemon and the images you approve are trusted. A compromised
  daemon or image can defeat container isolation.
- The host kernel and runtime are patched and not adversarial.
- The project is not a hostile symlink farm; the agent refuses to follow
  symlinks, but the filesystem itself is trusted to behave.
- Models, repository content, tool output and test output are untrusted data.
  Prompt injection is mitigated by role separation, not eliminated.
- Network-enabled test profiles can exfiltrate data visible inside the
  container copy; approve them deliberately.
- Secret filename filtering is heuristic. Values named in `redact_env` and
  provider key variables are redacted, but unknown secrets inside readable
  files can still reach an approved provider.
- Streaming responses are not implemented; requests are non-streaming with
  bounded response sizes and conservative retries. Retrying a request after a
  transport failure can duplicate a charge at a hosted provider.
- Ollama and custom-endpoint behaviour depends on the model's tool-calling
  support.
- Session resume supports safe checkpoints. Interrupted execution/evaluation
  or a prepared apply journal is marked `needs_attention` for manual review;
  it is not replayed automatically.

## Testing

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

End-to-end coverage uses deterministic scripted providers (no network) and a
fake evaluator to exercise plan → failing evaluation → repair → passing
evidence → review → approved apply, plus rejection and stale-evidence cases.

Docker integration tests are ignored by default because they need a daemon and
explicitly pulled images:

```sh
RELAY_DOCKER_TESTS=1 RELAY_TEST_IMAGE=alpine:3.20 \
  cargo test --test docker_evaluator -- --ignored --test-threads=1
```

They cover pass/fail, timeout, network isolation, credential absence, non-root
and read-only root, output bounds, cleanup and missing images.

Ollama and hosted-provider smoke tests are opt-in and were not run in the
environment where this implementation was validated; see `todo.md` for the
exact blockers.
