use std::collections::BTreeMap;
use std::path::Path;
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::{
    ApprovedProfile, CommandResult, EvaluationPolicy, EvaluationReport, Evaluator, TestOutcome,
    all_passed,
};
use crate::config::DockerConfig;
use crate::repo_fs::FileEntry;
use crate::safe_fs::DirHandle;
use crate::workspace::{Workspace, tree_hash};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_COPY_FILE_BYTES: usize = 256 * 1024 * 1024;
const CONTAINER_USER: &str = "65534:65534";

/// Docker (or a compatible runtime) sandbox for deterministic test execution.
///
/// It fails closed: if the daemon, an approved image or the runtime is
/// unavailable, evaluation fails instead of falling back to host execution.
pub struct DockerEvaluator {
    config: DockerConfig,
}

impl DockerEvaluator {
    pub fn new(config: DockerConfig) -> Self {
        Self { config }
    }

    async fn runtime(&self, args: &[&str]) -> Result<std::process::Output> {
        let mut command = Command::new(&self.config.runtime);
        command
            .args(args)
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
            .env("DOCKER_CLI_HINTS", "false")
            .stdin(Stdio::null());
        let output = tokio::time::timeout(CONTROL_TIMEOUT, command.output())
            .await
            .context("container runtime command timed out")?
            .with_context(|| format!("failed to execute {}", self.config.runtime))?;
        Ok(output)
    }

    fn container_name(&self, run_id: &str, profile: &str, phase: &str, index: usize) -> String {
        let run: String = run_id
            .chars()
            .filter(|character| character.is_ascii_alphanumeric())
            .take(12)
            .collect();
        let profile: String = profile
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character
                } else {
                    '-'
                }
            })
            .take(24)
            .collect();
        format!(
            "relay-{run}-{profile}-{phase}-{index}-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        )
    }

    fn base_args<'a>(
        &'a self,
        name: &'a str,
        mount: &'a str,
        network: bool,
        workdir: &'a str,
    ) -> Vec<String> {
        let mut args = vec![
            "run".to_owned(),
            "--rm".to_owned(),
            "--name".to_owned(),
            name.to_owned(),
            "--network".to_owned(),
            if network { "bridge" } else { "none" }.to_owned(),
            "--read-only".to_owned(),
            "--cap-drop".to_owned(),
            "ALL".to_owned(),
            "--security-opt".to_owned(),
            "no-new-privileges".to_owned(),
            "--user".to_owned(),
            CONTAINER_USER.to_owned(),
            "--memory".to_owned(),
            format!("{}m", self.config.memory_mb),
            "--memory-swap".to_owned(),
            format!("{}m", self.config.memory_mb),
            "--cpus".to_owned(),
            self.config.cpus.clone(),
            "--pids-limit".to_owned(),
            self.config.pids_limit.to_string(),
            "--tmpfs".to_owned(),
            format!("/tmp:rw,noexec,nosuid,nodev,size={}m", self.config.tmpfs_mb),
            "--workdir".to_owned(),
            workdir.to_owned(),
            "--env".to_owned(),
            "HOME=/tmp".to_owned(),
            "--env".to_owned(),
            "TMPDIR=/tmp".to_owned(),
            "--mount".to_owned(),
            mount.to_owned(),
        ];
        args.retain(|argument| !argument.is_empty());
        args
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_container(
        &self,
        name: &str,
        mount: &str,
        network: bool,
        workdir: &str,
        image: &str,
        argv: &[String],
        timeout: Duration,
    ) -> Result<CommandResult> {
        let mut args = self.base_args(name, mount, network, workdir);
        args.push(image.to_owned());
        args.extend(argv.iter().cloned());

        let mut command = Command::new(&self.config.runtime);
        command
            .args(&args)
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
            .env("DOCKER_CLI_HINTS", "false")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let guard = ContainerGuard::new(&self.config.runtime, name);
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to start {}", self.config.runtime))?;
        let stdout = child.stdout.take().context("missing container stdout")?;
        let stderr = child.stderr.take().context("missing container stderr")?;
        let limit = self.config.output_limit_bytes;
        let stdout_task = tokio::spawn(read_bounded(stdout, limit));
        let stderr_task = tokio::spawn(read_bounded(stderr, limit));

        let started = Instant::now();
        let (timed_out, exit_code) = match tokio::time::timeout(timeout, child.wait()).await {
            Ok(result) => {
                let status = result.with_context(|| format!("container {name} failed"))?;
                (false, status.code())
            }
            Err(_) => {
                let _ = child.start_kill();
                let kill = self.runtime(&["kill", name]).await;
                if let Ok(output) = &kill
                    && !output.status.success()
                {
                    let message = String::from_utf8_lossy(&output.stderr);
                    if !message.contains("No such container") {
                        tracing::warn!(container = name, %message, "docker kill failed");
                    }
                }
                let status = tokio::time::timeout(CONTROL_TIMEOUT, child.wait())
                    .await
                    .ok()
                    .and_then(Result::ok);
                (true, status.and_then(|status| status.code()))
            }
        };
        let (stdout_bytes, stdout_truncated) =
            stdout_task.await.context("stdout reader failed")??;
        let (stderr_bytes, stderr_truncated) =
            stderr_task.await.context("stderr reader failed")??;
        let cleanup_error = self
            .cleanup(name)
            .await
            .err()
            .map(|error| format!("{error:#}"));
        drop(guard);

        Ok(CommandResult {
            exit_code,
            timed_out,
            duration_ms: started.elapsed().as_millis() as u64,
            stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
            stderr: String::from_utf8_lossy(&stderr_bytes).into_owned(),
            stdout_truncated,
            stderr_truncated,
            cleanup_error,
        })
    }

    async fn cleanup(&self, name: &str) -> Result<()> {
        let output = self.runtime(&["rm", "-f", name]).await?;
        if output.status.success() {
            return Ok(());
        }
        let message = String::from_utf8_lossy(&output.stderr);
        if message.contains("No such container") {
            Ok(())
        } else {
            bail!("failed to remove container {name}: {}", message.trim())
        }
    }

    fn mount_spec(&self, copy: &Path) -> Result<String> {
        let path = copy.canonicalize().with_context(|| {
            format!("failed to resolve evaluation workspace {}", copy.display())
        })?;
        let text = path.to_str().context("evaluation path is not UTF-8")?;
        if text.contains(',') {
            bail!("evaluation workspace path must not contain a comma");
        }
        if !path.is_absolute() {
            bail!("evaluation workspace must be absolute");
        }
        Ok(format!("type=bind,source={text},target=/workspace"))
    }

    fn workdir(&self, cwd: &str) -> String {
        if cwd == "." || cwd.is_empty() {
            "/workspace".to_owned()
        } else {
            format!("/workspace/{}", cwd.trim_matches('/'))
        }
    }
}

struct ContainerGuard {
    runtime: String,
    name: String,
    armed: bool,
}

impl ContainerGuard {
    fn new(runtime: &str, name: &str) -> Self {
        Self {
            runtime: runtime.to_owned(),
            name: name.to_owned(),
            armed: true,
        }
    }
}

impl Drop for ContainerGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Best-effort synchronous cleanup for cancellation paths.
        let _ = std::process::Command::new(&self.runtime)
            .args(["rm", "-f", &self.name])
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

async fn read_bounded<R>(mut reader: R, limit: usize) -> Result<(Vec<u8>, bool)>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut retained = Vec::new();
    let mut buffer = vec![0u8; 16 * 1024];
    let mut truncated = false;
    loop {
        let count = reader.read(&mut buffer).await?;
        if count == 0 {
            break;
        }
        if retained.len() < limit {
            let room = limit - retained.len();
            let take = room.min(count);
            retained.extend_from_slice(&buffer[..take]);
            if take < count {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
    Ok((retained, truncated))
}

fn copy_snapshot_to(
    stage: &DirHandle,
    files: &BTreeMap<String, FileEntry>,
    destination: &Path,
) -> Result<()> {
    std::fs::create_dir_all(destination)?;
    for (relative, entry) in files {
        let read = stage
            .read(relative, MAX_COPY_FILE_BYTES)
            .with_context(|| format!("failed to read staged {relative}"))?;
        if read.truncated {
            bail!("staged file {relative} is too large to evaluate safely");
        }
        if crate::util::hash_bytes(&read.bytes) != entry.hash {
            bail!("staged file {relative} changed while preparing evaluation");
        }
        let target = destination.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &read.bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            // The container runs as an unprivileged user; the isolated copy is
            // writable only inside this run directory.
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o777))?;
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut directories = vec![destination.to_path_buf()];
        while let Some(directory) = directories.pop() {
            std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o777))?;
            let entries = std::fs::read_dir(&directory)?;
            for entry in entries {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    directories.push(entry.path());
                }
            }
        }
    }
    Ok(())
}

fn verify_source_identity(copy: &DirHandle, files: &BTreeMap<String, FileEntry>) -> Result<()> {
    for (relative, entry) in files {
        let read = copy
            .read(relative, 1)
            .with_context(|| format!("preparation removed or changed {relative}"))?;
        if read.hash != entry.hash {
            bail!("preparation modified the reviewed source file {relative}");
        }
        if let Some(mode) = entry.mode
            && copy.mode_of(relative).ok() != Some(mode)
        {
            bail!("preparation changed permissions on the reviewed file {relative}");
        }
    }
    Ok(())
}

#[async_trait]
impl Evaluator for DockerEvaluator {
    async fn preflight(&self, policy: &EvaluationPolicy) -> Result<BTreeMap<String, String>> {
        policy.validate()?;
        let output = self
            .runtime(&["version", "--format", "{{.Server.Version}}"])
            .await?;
        if !output.status.success() {
            bail!(
                "container runtime is unavailable; evaluation refuses to fall back to the host: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let mut images = BTreeMap::new();
        for profile in &policy.profiles {
            if images.contains_key(&profile.image) {
                continue;
            }
            let output = self
                .runtime(&["image", "inspect", "--format", "{{.Id}}", &profile.image])
                .await?;
            if !output.status.success() {
                bail!(
                    "approved image {:?} is not available locally; pull it explicitly before running: {}",
                    profile.image,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            let id = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if id.is_empty() {
                bail!("could not resolve image identity for {:?}", profile.image);
            }
            images.insert(profile.image.clone(), id);
        }
        Ok(images)
    }

    async fn evaluate(
        &self,
        workspace: &Workspace,
        policy: &EvaluationPolicy,
        run_dir: &Path,
        run_id: &str,
    ) -> Result<EvaluationReport> {
        policy.validate()?;
        let files = workspace.current_files()?;
        let snapshot_hash = tree_hash(&files)?;
        let image_ids = self.preflight(policy).await?;
        if workspace.current_files()? != files {
            bail!("staged snapshot changed while evaluation was starting");
        }
        let stage = DirHandle::open_root(workspace.stage_root())?;
        let eval_root = run_dir.join("eval").join(format!(
            "{}-{}",
            run_id
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .collect::<String>(),
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        ));
        if eval_root.exists() {
            bail!("evaluation directory already exists");
        }
        std::fs::create_dir_all(&eval_root)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&eval_root, std::fs::Permissions::from_mode(0o700))?;
        }

        let policy_hash = policy.hash();
        let mut outcomes = Vec::with_capacity(policy.profiles.len());

        for (index, profile) in policy.profiles.iter().enumerate() {
            // Each profile gets its own fresh copy so tests cannot interfere
            // with each other or with preparation.
            let copy = eval_root.join(format!("workspace-{index}"));
            copy_snapshot_to(&stage, &files, &copy)?;
            let mount = self.mount_spec(&copy)?;
            let outcome = self
                .evaluate_profile(
                    run_id,
                    profile,
                    index,
                    &copy,
                    &mount,
                    &files,
                    &snapshot_hash,
                    &policy_hash,
                    image_ids.get(&profile.image).cloned(),
                )
                .await;
            outcomes.push(outcome);
        }
        let all_passed = all_passed(&outcomes);
        Ok(EvaluationReport {
            id: format!("eval_{}", uuid::Uuid::new_v4().simple()),
            snapshot_hash,
            policy_hash,
            outcomes,
            all_passed,
        })
    }
}

impl DockerEvaluator {
    #[allow(clippy::too_many_arguments)]
    async fn evaluate_profile(
        &self,
        run_id: &str,
        profile: &ApprovedProfile,
        index: usize,
        copy: &Path,
        mount: &str,
        files: &BTreeMap<String, FileEntry>,
        snapshot_hash: &str,
        policy_hash: &str,
        image_id: Option<String>,
    ) -> TestOutcome {
        let workdir = self.workdir(&profile.cwd);
        let mut prepare = None;
        let mut setup_failure = None;
        let mut cleanup_error = None;

        if !profile.prepare_command.is_empty() {
            let name = self.container_name(run_id, &profile.name, "prepare", index);
            match self
                .run_container(
                    &name,
                    mount,
                    profile.prepare_allow_network,
                    &workdir,
                    &profile.image,
                    &profile.prepare_command,
                    Duration::from_secs(profile.timeout_secs),
                )
                .await
            {
                Ok(result) => {
                    let succeeded = result.succeeded();
                    if let Some(error) = result.cleanup_error.clone() {
                        cleanup_error = Some(error);
                    }
                    if !succeeded {
                        setup_failure = Some("dependency preparation failed".to_owned());
                    }
                    prepare = Some(result);
                }
                Err(error) => {
                    setup_failure = Some(format!("preparation could not start: {error:#}"))
                }
            }
            if setup_failure.is_none() {
                match DirHandle::open_root(copy)
                    .and_then(|handle| verify_source_identity(&handle, files))
                {
                    Ok(()) => {}
                    Err(error) => setup_failure = Some(format!("{error:#}")),
                }
            }
        }

        let (result, passed) = if setup_failure.is_some() {
            (None, false)
        } else {
            let name = self.container_name(run_id, &profile.name, "test", index);
            match self
                .run_container(
                    &name,
                    mount,
                    profile.allow_network,
                    &workdir,
                    &profile.image,
                    &profile.command,
                    Duration::from_secs(profile.timeout_secs),
                )
                .await
            {
                Ok(result) => {
                    if let Some(error) = result.cleanup_error.clone() {
                        cleanup_error = Some(error);
                    }
                    let passed = result.succeeded() && cleanup_error.is_none();
                    (Some(result), passed)
                }
                Err(error) => {
                    setup_failure = Some(format!("test container could not start: {error:#}"));
                    (None, false)
                }
            }
        };

        TestOutcome {
            name: profile.name.clone(),
            image: profile.image.clone(),
            image_id,
            command: profile.command.clone(),
            cwd: profile.cwd.clone(),
            allow_network: profile.allow_network,
            prepare_allow_network: profile.prepare_allow_network,
            snapshot_hash: snapshot_hash.to_owned(),
            policy_hash: policy_hash.to_owned(),
            passed,
            setup_failure,
            prepare,
            result,
            cleanup_error,
        }
    }
}
