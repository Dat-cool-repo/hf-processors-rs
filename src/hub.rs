//! Locate `preprocessor_config.json`: local file/dir, or the Hugging Face Hub over HTTPS.

use crate::error::{Error, Result};
use std::path::{Path, PathBuf};

pub const CONFIG_NAME: &str = "preprocessor_config.json";

/// Resolve `repo_or_path` to a local config file, downloading it if needed.
///
/// * an existing file is used as-is;
/// * an existing directory must contain `preprocessor_config.json`;
/// * anything else is treated as a Hub repo id (`org/name`, optionally `org/name@revision`)
///   and fetched from `https://huggingface.co/<repo>/resolve/<revision>/preprocessor_config.json`
///   (requires the `hub` feature). Downloads are cached under `$HF_PROCESSORS_CACHE`, else
///   `$XDG_CACHE_HOME/hf-processors`, else `~/.cache/hf-processors`. `HF_TOKEN` is sent
///   as a bearer token when set; `HF_ENDPOINT` overrides the host.
pub fn resolve_config(repo_or_path: &str) -> Result<PathBuf> {
    let p = Path::new(repo_or_path);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    resolve_file(repo_or_path, CONFIG_NAME)
}

/// Like [`resolve_config`] for any file of the repo (e.g. `config.json`). For a local file
/// path, `filename` is looked up next to it.
pub fn resolve_file(repo_or_path: &str, filename: &str) -> Result<PathBuf> {
    let p = Path::new(repo_or_path);
    if p.is_file() {
        if p.file_name().is_some_and(|n| n == filename) {
            return Ok(p.to_path_buf());
        }
        let sib = p.with_file_name(filename);
        return if sib.is_file() {
            Ok(sib)
        } else {
            Err(Error::Config(format!("no {filename} next to {}", p.display())))
        };
    }
    if p.is_dir() {
        let f = p.join(filename);
        return if f.is_file() {
            Ok(f)
        } else {
            Err(Error::Config(format!("{} has no {filename}", p.display())))
        };
    }
    download(repo_or_path, filename)
}

fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("HF_PROCESSORS_CACHE") {
        return PathBuf::from(d);
    }
    if let Ok(d) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(d).join("hf-processors");
    }
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("hf-processors")
}

fn split_revision(repo: &str) -> (&str, &str) {
    match repo.split_once('@') {
        Some((r, rev)) => (r, rev),
        None => (repo, "main"),
    }
}

#[cfg(feature = "hub")]
fn download(repo: &str, filename: &str) -> Result<PathBuf> {
    let (repo_id, revision) = split_revision(repo);
    if repo_id.split('/').count() > 2 || repo_id.is_empty() || repo_id.contains("..") {
        return Err(Error::Hub(format!("`{repo}` is neither a local path nor a valid repo id")));
    }
    let target = cache_dir().join(repo_id.replace('/', "--")).join(revision).join(filename);
    if target.is_file() {
        return Ok(target);
    }
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let url = format!("{endpoint}/{repo_id}/resolve/{revision}/{filename}");
    let mut req = ureq::get(&url).header("User-Agent", "hf-processors-rs/0.1");
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", &format!("Bearer {tok}"));
    }
    let mut resp = req.call().map_err(|e| Error::Hub(format!("GET {url}: {e}")))?;
    let body = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| Error::Hub(format!("reading {url}: {e}")))?;
    // Validate before caching.
    serde_json::from_str::<serde_json::Value>(&body)?;
    std::fs::create_dir_all(target.parent().expect("has parent"))?;
    let tmp = target.with_extension("json.tmp");
    std::fs::write(&tmp, body)?;
    std::fs::rename(&tmp, &target)?;
    Ok(target)
}

#[cfg(not(feature = "hub"))]
fn download(repo: &str, _filename: &str) -> Result<PathBuf> {
    let _ = (split_revision, cache_dir);
    Err(Error::Hub(format!(
        "`{repo}` is not a local path and the `hub` feature is disabled"
    )))
}
