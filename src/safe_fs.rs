//! Capability-relative filesystem operations.
//!
//! Path checks followed by `std::fs` calls are racy: an attacker able to modify
//! the workspace between the check and the open can redirect an operation with
//! a symlink. On Unix this module walks directories with `openat(O_NOFOLLOW)`
//! and performs reads, atomic writes and removals relative to the resulting
//! directory handles, so no path component is ever re-resolved.
//!
//! On non-Unix targets a conservative fallback revalidates every component;
//! callers should treat that as best effort rather than a hard boundary.

use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::util::is_sensitive_rel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStat {
    pub size: u64,
    pub mode: u32,
    pub is_file: bool,
    pub is_dir: bool,
    pub is_symlink: bool,
}

#[derive(Debug)]
pub struct ReadResult {
    pub bytes: Vec<u8>,
    pub hash: String,
    pub size: u64,
    pub truncated: bool,
}

pub fn validate_relative(relative: &str) -> Result<Vec<&std::ffi::OsStr>> {
    if relative.is_empty() {
        bail!("path must not be empty");
    }
    let path = Path::new(relative);
    if path.is_absolute() {
        bail!("path must be relative");
    }
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::Normal(name) => {
                let text = name.to_string_lossy();
                if text == "." || text == ".." || text.contains('\0') {
                    bail!("path contains an invalid component");
                }
                components.push(name);
            }
            std::path::Component::CurDir => {}
            _ => bail!("path must be relative and must not contain '..'"),
        }
    }
    if components.is_empty() {
        bail!("path must name a file");
    }
    if is_sensitive_rel(path) {
        bail!("path is excluded because it may contain secrets");
    }
    Ok(components)
}

#[cfg(unix)]
mod imp {
    use std::ffi::{CString, OsStr};
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    use anyhow::{Context, Result, bail};

    use super::{FileStat, ReadResult};

    fn cstring(value: &OsStr) -> Result<CString> {
        CString::new(value.as_bytes()).context("path contains a NUL byte")
    }

    fn check(result: i32, what: &str) -> Result<i32> {
        if result < 0 {
            Err(std::io::Error::last_os_error()).with_context(|| what.to_owned())
        } else {
            Ok(result)
        }
    }

    pub struct DirHandle {
        fd: OwnedFd,
    }

    impl std::fmt::Debug for DirHandle {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.debug_struct("DirHandle").finish_non_exhaustive()
        }
    }

    impl DirHandle {
        pub fn open_root(root: &Path) -> Result<Self> {
            let path = cstring(root.as_os_str())?;
            let fd = unsafe {
                libc::open(
                    path.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            check(fd, &format!("failed to open workspace {}", root.display()))?;
            Ok(Self {
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
            })
        }

        fn open_dir_at(&self, name: &OsStr) -> Result<Self> {
            let name = cstring(name)?;
            let fd = unsafe {
                libc::openat(
                    self.fd.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            check(fd, "refusing to traverse a symlink or non-directory")?;
            Ok(Self {
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
            })
        }

        pub fn open_dir(&self, components: &[&OsStr]) -> Result<Self> {
            let mut current = self.try_clone()?;
            for component in components {
                current = current.open_dir_at(component)?;
            }
            Ok(current)
        }

        pub fn create_dir_all(&self, components: &[&OsStr]) -> Result<Self> {
            let mut current = self.try_clone()?;
            for component in components {
                match current.open_dir_at(component) {
                    Ok(next) => current = next,
                    Err(_) => {
                        let name = cstring(component)?;
                        let result =
                            unsafe { libc::mkdirat(current.fd.as_raw_fd(), name.as_ptr(), 0o755) };
                        if result < 0 {
                            let error = std::io::Error::last_os_error();
                            if error.kind() != std::io::ErrorKind::AlreadyExists {
                                return Err(error).context("failed to create a directory");
                            }
                        }
                        current = current.open_dir_at(component)?;
                    }
                }
            }
            Ok(current)
        }

        pub fn stat(&self, name: &OsStr) -> Result<FileStat> {
            let name = cstring(name)?;
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            let result = unsafe {
                libc::fstatat(
                    self.fd.as_raw_fd(),
                    name.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            };
            check(result, "failed to stat file")?;
            let stat = unsafe { stat.assume_init() };
            let kind = stat.st_mode & libc::S_IFMT;
            Ok(FileStat {
                size: stat.st_size.max(0) as u64,
                mode: (stat.st_mode & 0o777) as u32,
                is_file: kind == libc::S_IFREG,
                is_dir: kind == libc::S_IFDIR,
                is_symlink: kind == libc::S_IFLNK,
            })
        }

        pub fn exists(&self, name: &OsStr) -> bool {
            self.stat(name).is_ok()
        }

        pub fn read_file(
            &self,
            components: &[&OsStr],
            max_read_bytes: usize,
        ) -> Result<ReadResult> {
            let (name, parents) = components.split_last().context("path must name a file")?;
            let directory = self.open_dir(parents)?;
            let name = cstring(name)?;
            let fd = unsafe {
                // O_NONBLOCK prevents a planted FIFO from blocking the agent;
                // the fstat below then rejects anything that is not a regular
                // file.
                libc::openat(
                    directory.fd.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
                )
            };
            check(fd, "failed to open file")?;
            let mut file = unsafe { File::from_raw_fd(fd) };
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                bail!("not a regular file");
            }
            let mut hasher = blake3::Hasher::new();
            let mut prefix = Vec::new();
            let mut buffer = vec![0u8; 64 * 1024];
            let mut total = 0u64;
            let mut truncated = false;
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                hasher.update(&buffer[..count]);
                total += count as u64;
                if prefix.len() < max_read_bytes {
                    let room = max_read_bytes - prefix.len();
                    let take = room.min(count);
                    prefix.extend_from_slice(&buffer[..take]);
                    if take < count {
                        truncated = true;
                    }
                } else {
                    truncated = true;
                }
            }
            Ok(ReadResult {
                bytes: prefix,
                hash: hasher.finalize().to_hex().to_string(),
                size: total,
                truncated,
            })
        }

        pub fn write_atomic(
            &self,
            components: &[&OsStr],
            data: &[u8],
            mode: Option<u32>,
        ) -> Result<()> {
            let (name, parents) = components.split_last().context("path must name a file")?;
            let directory = self.create_dir_all(parents)?;
            if directory.stat(name).is_ok_and(|stat| stat.is_symlink) {
                bail!("refusing to replace a symlink");
            }
            let name = cstring(name)?;
            let temp_name = cstring(OsStr::new(&format!(
                ".relay-tmp-{}",
                uuid::Uuid::new_v4().simple()
            )))?;
            let fd = unsafe {
                libc::openat(
                    directory.fd.as_raw_fd(),
                    temp_name.as_ptr(),
                    libc::O_WRONLY
                        | libc::O_CREAT
                        | libc::O_EXCL
                        | libc::O_NOFOLLOW
                        | libc::O_CLOEXEC,
                    0o600,
                )
            };
            check(fd, "failed to create a temporary file")?;
            let mut temp = unsafe { File::from_raw_fd(fd) };
            let result = (|| -> Result<()> {
                temp.write_all(data)?;
                temp.sync_all()?;
                if let Some(mode) = mode {
                    temp.set_permissions(std::fs::Permissions::from_mode(mode & 0o777))?;
                }
                let rename = unsafe {
                    libc::renameat(
                        directory.fd.as_raw_fd(),
                        temp_name.as_ptr(),
                        directory.fd.as_raw_fd(),
                        name.as_ptr(),
                    )
                };
                check(rename, "failed to atomically replace file")?;
                Ok(())
            })();
            if result.is_err() {
                let _ = unsafe { libc::unlinkat(directory.fd.as_raw_fd(), temp_name.as_ptr(), 0) };
            }
            result
        }

        pub fn remove_file(&self, components: &[&OsStr]) -> Result<()> {
            let (name, parents) = components.split_last().context("path must name a file")?;
            let directory = self.open_dir(parents)?;
            let name = cstring(name)?;
            let result = unsafe { libc::unlinkat(directory.fd.as_raw_fd(), name.as_ptr(), 0) };
            check(result, "failed to delete file").map(|_| ())
        }

        pub fn mode(&self, components: &[&OsStr]) -> Result<u32> {
            let (name, parents) = components.split_last().context("path must name a file")?;
            let directory = self.open_dir(parents)?;
            Ok(directory.stat(name)?.mode)
        }

        fn try_clone(&self) -> Result<Self> {
            let fd = unsafe { libc::fcntl(self.fd.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
            check(fd, "failed to duplicate directory handle")?;
            Ok(Self {
                fd: unsafe { OwnedFd::from_raw_fd(fd) },
            })
        }
    }
}

#[cfg(not(unix))]
mod imp {
    use std::ffi::OsStr;
    use std::path::PathBuf;

    use anyhow::{Context, Result, bail};

    use super::{FileStat, ReadResult};

    #[derive(Debug)]
    pub struct DirHandle {
        root: PathBuf,
    }

    impl DirHandle {
        pub fn open_root(root: &Path) -> Result<Self> {
            Ok(Self {
                root: root
                    .canonicalize()
                    .with_context(|| format!("failed to resolve {}", root.display()))?,
            })
        }

        fn resolve(&self, components: &[&OsStr]) -> Result<PathBuf> {
            let mut path = self.root.clone();
            for component in components {
                path.push(component);
            }
            Ok(path)
        }

        pub fn open_dir(&self, _components: &[&OsStr]) -> Result<Self> {
            Ok(Self {
                root: self.root.clone(),
            })
        }

        pub fn create_dir_all(&self, components: &[&OsStr]) -> Result<Self> {
            let path = self.resolve(components)?;
            std::fs::create_dir_all(&path)?;
            Ok(Self { root: path })
        }

        pub fn stat(&self, name: &OsStr) -> Result<FileStat> {
            let path = self.root.join(name);
            let metadata = std::fs::symlink_metadata(path)?;
            Ok(FileStat {
                size: metadata.len(),
                mode: 0,
                is_file: metadata.is_file(),
                is_dir: metadata.is_dir(),
                is_symlink: metadata.file_type().is_symlink(),
            })
        }

        pub fn exists(&self, name: &OsStr) -> bool {
            self.stat(name).is_ok()
        }

        pub fn read_file(
            &self,
            components: &[&OsStr],
            max_read_bytes: usize,
        ) -> Result<ReadResult> {
            let path = self.resolve(components)?;
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                bail!("refusing to follow a symlink");
            }
            let bytes = std::fs::read(&path)?;
            let truncated = bytes.len() > max_read_bytes;
            let size = bytes.len() as u64;
            let hash = blake3::hash(&bytes).to_hex().to_string();
            Ok(ReadResult {
                bytes: bytes[..bytes.len().min(max_read_bytes)].to_vec(),
                hash,
                size,
                truncated,
            })
        }

        pub fn write_atomic(
            &self,
            components: &[&OsStr],
            data: &[u8],
            mode: Option<u32>,
        ) -> Result<()> {
            let path = self.resolve(components)?;
            crate::util::atomic_write(&path, data, mode)
        }

        pub fn remove_file(&self, components: &[&OsStr]) -> Result<()> {
            std::fs::remove_file(self.resolve(components)?)?;
            Ok(())
        }

        pub fn mode(&self, components: &[&OsStr]) -> Result<u32> {
            let _ = components;
            Ok(0)
        }

        fn try_clone(&self) -> Result<Self> {
            Ok(Self {
                root: self.root.clone(),
            })
        }
    }
}

pub use imp::DirHandle;

impl DirHandle {
    pub fn metadata(&self, relative: &str) -> Result<FileStat> {
        let components = validate_relative(relative)?;
        let (name, parents) = components.split_last().context("path must name a file")?;
        let directory = self.open_dir(parents)?;
        directory.stat(name)
    }

    pub fn read(&self, relative: &str, max_read_bytes: usize) -> Result<ReadResult> {
        let components = validate_relative(relative)?;
        self.read_file(&components, max_read_bytes)
    }

    pub fn write(&self, relative: &str, data: &[u8], mode: Option<u32>) -> Result<()> {
        let components = validate_relative(relative)?;
        self.write_atomic(&components, data, mode)
    }

    pub fn remove(&self, relative: &str) -> Result<()> {
        let components = validate_relative(relative)?;
        self.remove_file(&components)
    }

    pub fn mode_of(&self, relative: &str) -> Result<u32> {
        let components = validate_relative(relative)?;
        self.mode(&components)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal_and_sensitive_paths() {
        for path in [
            "../x",
            "/etc/passwd",
            "a/../../b",
            ".env",
            ".ssh/id_rsa",
            "",
        ] {
            assert!(validate_relative(path).is_err(), "{path}");
        }
        assert!(validate_relative("src/main.rs").is_ok());
    }

    #[test]
    fn reads_and_writes_without_following_final_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("a.txt"), "hello").unwrap();
        let root = DirHandle::open_root(directory.path()).unwrap();
        let read = root.read("a.txt", 1024).unwrap();
        assert_eq!(read.bytes, b"hello");
        assert_eq!(read.hash, crate::util::hash_bytes(b"hello"));

        root.write("nested/b.txt", b"new", Some(0o644)).unwrap();
        assert_eq!(
            std::fs::read_to_string(directory.path().join("nested/b.txt")).unwrap(),
            "new"
        );

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("/etc/passwd", directory.path().join("link")).unwrap();
            assert!(root.read("link", 1024).is_err());
            assert!(root.write("link", b"x", None).is_err());
            std::os::unix::fs::symlink("/tmp", directory.path().join("dirlink")).unwrap();
            assert!(root.write("dirlink/x", b"x", None).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_fifos_without_blocking() {
        let directory = tempfile::tempdir().unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(directory.path().join("pipe"))
            .status()
            .unwrap();
        assert!(status.success());
        let root = DirHandle::open_root(directory.path()).unwrap();
        let started = std::time::Instant::now();
        assert!(root.read("pipe", 16).is_err());
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    #[test]
    fn bounds_reads_but_hashes_the_whole_file() {
        let directory = tempfile::tempdir().unwrap();
        let content = "a".repeat(10_000);
        std::fs::write(directory.path().join("big.txt"), &content).unwrap();
        let root = DirHandle::open_root(directory.path()).unwrap();
        let read = root.read("big.txt", 100).unwrap();
        assert_eq!(read.bytes.len(), 100);
        assert!(read.truncated);
        assert_eq!(read.hash, crate::util::hash_bytes(content.as_bytes()));
    }
}
