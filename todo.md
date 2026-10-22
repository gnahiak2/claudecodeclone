# relay-code implementation handoff

## goal and agreed scope

Build an original Rust coding agent, not a proprietary clone or payment bypass. Hosted APIs require valid credentials and may charge; local Ollama is the no-hosted-fee option.

Approved architecture:

- agent harness owns permissions, approvals, state, budgets and retries.
- tier 1 planner: read-only repository map/read/search.
- tier 2 executor: native Rust model/tool loop, reads and patches only a staged workspace; no host shell.
- tier 3 evaluator: deterministic Docker-isolated test execution, not an LLM verdict.
- providers: Ollama, Cohere, OpenAI, Anthropic and configurable OpenAI-compatible endpoints.
- flow: inspect → plan → approval → stage → execute → evaluate → bounded repair → diff review → explicit apply approval.
- full-screen TUI, external Codex/OpenCode executor wrappers and unrelated service APIs are out of initial scope.

The original approved plan is at `/Users/kaihang/.commandcode/plans/rust-three-tier-coding-agent.md`. This file summarizes the current state independently.

## current state (verified 2026-10-22)

The project now compiles as a complete library plus `relay` binary. The previous
stub and uncompiled drafts are gone; roles and evaluator are exported and tested.

### foundation and security fixes

- [x] `src/main.rs` is a real clap CLI: `chat`, `plan`, `run`, `providers`,
  `session inspect`, `session resume`.
- [x] `src/lib.rs` exports `roles`, `evaluator`, `harness`, `session` and `safe_fs`.
- [x] MSRV corrected in `Cargo.toml` to `1.88` (edition 2024 plus let-chains).
- [x] Capability-relative filesystem layer in `src/safe_fs.rs`: `openat`/
  `fstatat`/`renameat`/`unlinkat` with `O_NOFOLLOW`, used for tool reads,
  patches, deletes and project-side apply/rollback. Final symlinks are never
  followed; atomic replacement refuses to replace one.
- [x] Snapshot source reads also go through the no-follow handle. Inconsistent
  or unreadable sources fail the snapshot instead of producing a partial
  baseline. Run directories inside the project are rejected.
- [x] `collect_files` is an authoritative inventory independent of editable
  ignore rules. `.gitignore` edits cannot hide staged changes. Tracked-but-
  ignored files are preserved by removing the old build-directory filter.
- [x] `list_files`/`search` use gitignore-aware walking from the repository root,
  so subdirectory searches keep root-relative exclusions. Reads and hashes come
  from the same bounded file read.
- [x] Reads, searches, diffs and provider responses are byte-bounded; searches
  have traversal limits and regex size limits. Diff refuses non-regular files
  and files over 8 MiB.
- [x] Patch batches validate all paths and normalized duplicates before any
  mutation. On mid-batch failure, earlier edits are restored from exact
  preimages; incomplete rollback is reported, never swallowed.
- [x] Executor mutations are enforced against the approved plan file scope in
  tool dispatch, not just the prompt.
- [x] Apply is bound to the reviewed snapshot and passing evidence:
  `Workspace::apply` requires matching snapshot hash, evaluation id, approval
  and evidence, re-verifies staged bytes and project content/mode preimages, and
  writes per-operation journal states. Rollback errors are recorded and yield
  `rollback_failed`, not a false success.
- [x] Provider redirects are disabled; credential headers are marked sensitive;
  TLS verification remains at reqwest's secure default. Responses are bounded
  before allocation.
- [x] Tool results and errors are redacted before being sent back to a model;
  sessions/events are redacted; CLI output escapes terminal control characters.
- [x] Provider parsers reject `length`/filter/incomplete finish reasons and
  non-object tool arguments before dispatch. Tool names are validated.
- [x] Capabilities are configurable (`providers.*.tools`); roles fail
  validation if tools are disabled. Hosted providers require a key env name;
  local Ollama and explicitly configured compatible endpoints may be
  unauthenticated. Custom endpoints still require `trust_custom_endpoint`
  before a key is sent.
- [x] Git subprocesses run with a cleared environment, no system/global config,
  prompts disabled and hooks/fsmonitor disabled.
- [x] Config validation covers Rust paths only after parsing, https-or-loopback
  endpoints, Docker runtime whitelist, CPU range, memory/PID/output/tmpfs
  bounds, argv/image/timeout/cwd, duplicate profiles and run-wide budgets.
- [x] Docker profiles are revalidated by `EvaluationPolicy::validate`, so an
  approved policy cannot smuggle invalid argv or an invalid image.

### evaluator

- [x] `src/evaluator/docker.rs` implemented, exported, and exercised for
  fail-closed preflight logic. Docker runtime and images are never auto-pulled.
- [x] A fresh isolated copy per approved profile; preparation and test share
  that profile copy. Original checkout, home, credentials and Docker socket are
  never mounted.
- [x] Non-root UID/GID 65534, read-only root, `--cap-drop ALL`,
  `no-new-privileges`, CPU/memory/PID/tmpfs limits, bounded stdout/stderr,
  wall-clock timeout, fixed argument arrays and generated container names.
- [x] Network disabled by default; preparation network is separately modelled
  and approved. Preparation may add files but cannot modify reviewed source
  files or their modes (verified after preparation).
- [x] Containers are removed on success, failure and timeout; cleanup failure is
  reported. A Drop guard handles cancellation.
- [x] Evidence binds snapshot hash, exact approved profile policy (image, argv,
  cwd, timeout, network flags), resolved image id and outcomes.
- [ ] Docker integration tests exist in `tests/docker_evaluator.rs` but are
  ignored. They have not run; see blockers.

### harness, sessions and CLI

- [x] Typed phases and flow in `src/harness.rs`; failures now persist a truthful
  `failed` phase instead of leaving an in-flight phase behind.
- [x] Plan, evaluation policy and apply are separate external approvals. The
  model cannot approve; non-interactive runs reject unapproved gates.
- [x] Run-wide budgets for planner/executor calls, tool calls and tokens are
  reserved and persisted before each request; usage is recorded afterwards.
  Repairs default to two and never reset on resume.
- [x] Private sessions outside the project, versioned state, exclusive locks,
  atomic writes, redacted append-only events, and no credential persistence.
- [x] Interrupted execution/evaluation and prepared apply journals are marked
  `needs_attention`; no mutation is replayed automatically. Resume re-verifies
  the saved plan hash and reuses approvals only when their exact hashes match.
- [x] No automatic commits or pushes; the workspace is still not a Git repo and
  nothing was committed.

### docs

- [x] `README.md` covers local Ollama setup, hosted credentials/costs, Docker
  images, approvals, threat model, limitations and test commands.
- [x] `config.example.toml` with local-only defaults and commented hosted
  providers.
- [x] `.gitignore` for build output and secrets.

## validation evidence

Run with the per-command CA override because the system PEM path is unusable:

```sh
CARGO_HTTP_CAINFO=/opt/homebrew/etc/ca-certificates/cert.pem \
SSL_CERT_FILE=/opt/homebrew/etc/ca-certificates/cert.pem cargo test
```

Results:

- `cargo fmt --all -- --check`: pass.
- `cargo clippy --all-targets --all-features -- -D warnings`: pass.
- `cargo test`: 73 passed, 9 ignored, 0 failed.
  - 64 library unit tests.
  - 3 CLI tests (`--help`, provider display/redaction, unknown session).
  - 6 end-to-end harness tests: plan → failing evaluation → repair → passing
    evidence → review → approved apply; plan rejection; apply rejection; stale
    snapshot evidence; ignore-rule tampering; scripted provider determinism.
  - 9 ignored Docker integration tests.
- `cargo build --release`: pass.
- `relay --config config.example.toml providers`: pass; prints roles, endpoints,
  key presence (never values) and the sandbox policy.

## blockers and skipped checks

- Docker is not installed (`docker: command not found`). Podman 6.1.4 is
  installed but its VM is stopped (`podman machine list` shows `Never` started;
  the socket is refused). No containers were started and no sandbox integration
  test ran. Do not treat the Docker tests as passed.
- Ollama is not installed/running, so no local-model smoke test ran.
- No hosted-provider request was made. Credentials were not available/created,
  and no billable call was attempted. Adapter schemas beyond Cohere have not
  been re-verified against official docs in this session.
- MSRV is declared as 1.88 but only Rust/Cargo 1.98.1 is installed; no 1.88
  toolchain was available to verify the minimum. (Let-chains require 1.88.)
- Streaming responses are still not implemented. Requests are non-streaming with
  bounded bodies and conservative retries.

## remaining known limitations

- Containers are a strong isolation layer but not a perfect boundary for hostile
  workloads or a compromised daemon/image; this is documented in the README.
- Secret filename filtering is heuristic. Values listed in `redact_env` and
  provider key variables are redacted; unknown secrets in readable files can
  still reach an approved provider.
- Provider retries after a transport failure can duplicate a charge at a hosted
  provider.
- Patch batch rollback is in-memory and reports incomplete rollback; it is not a
  durable cross-process journal (apply has a durable per-operation journal).
- Preparation that modifies a reviewed source file is treated as setup failure;
  projects whose dependency install rewrites tracked files need a different
  approved workflow.
- `max_context_tokens` can only be enforced when the provider reports input
  token usage.
- Ignore-rule-independent inventory intentionally excludes constant build/secret
  directories (e.g. `target`, `.git`, `.ssh`), even if a model stages files
  there.
