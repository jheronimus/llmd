use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::copy;
use std::path::{Path, PathBuf};
use tracing::info;

pub const DEFAULT_LAYA_TOKENIZER_URL: &str =
    "https://huggingface.co/Emerald7664/laya-multilingual-onnx/resolve/main/tokenizer.json";

pub const DEFAULT_LAYA_MODEL_URL: &str =
    "https://huggingface.co/Emerald7664/laya-multilingual-onnx/resolve/main/model.onnx";

pub const DEFAULT_QWEN_URL: &str =
    "https://huggingface.co/Qwen/Qwen3-0.6B-Instruct-GGUF/resolve/main/qwen3-0.6b-instruct-q4_k_m.gguf";

pub const DEFAULT_QWEN_FILENAME: &str = "qwen3-0.6b-instruct-q4_k_m.gguf";

/// Ensure System 1 (Laya ONNX + tokenizer.json) is present in target directory.
pub fn ensure_sys1_model(target_dir: &Path) -> Result<PathBuf> {
    let tok_path = target_dir.join("tokenizer.json");
    let onnx_path = target_dir.join("model.onnx");

    if tok_path.exists() && onnx_path.exists() {
        return Ok(target_dir.to_path_buf());
    }

    fs::create_dir_all(target_dir)?;

    if !tok_path.exists() {
        info!(
            "System 1 tokenizer missing at {:?}. Downloading from {}...",
            tok_path, DEFAULT_LAYA_TOKENIZER_URL
        );
        let resp = ureq::get(DEFAULT_LAYA_TOKENIZER_URL)
            .call()
            .context("Failed downloading Laya tokenizer")?;
        let mut dest = File::create(&tok_path)?;
        let mut reader = resp.into_body().into_reader();
        copy(&mut reader, &mut dest)?;
        info!("System 1 tokenizer downloaded successfully to {:?}", tok_path);
    }

    if !onnx_path.exists() {
        info!(
            "System 1 model missing at {:?}. Downloading from {}...",
            onnx_path, DEFAULT_LAYA_MODEL_URL
        );
        let resp = ureq::get(DEFAULT_LAYA_MODEL_URL)
            .call()
            .context("Failed downloading Laya ONNX model")?;
        let mut dest = File::create(&onnx_path)?;
        let mut reader = resp.into_body().into_reader();
        copy(&mut reader, &mut dest)?;
        info!("System 1 ONNX model downloaded successfully to {:?}", onnx_path);
    }

    Ok(target_dir.to_path_buf())
}

/// Ensure System 2 (Qwen3 GGUF) is present in target file path.
pub fn ensure_sys2_model(target_path: &Path) -> Result<PathBuf> {
    if target_path.exists() {
        return Ok(target_path.to_path_buf());
    }

    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)?;
    }

    info!(
        "System 2 model missing at {:?}. Downloading from {}...",
        target_path, DEFAULT_QWEN_URL
    );

    let resp = ureq::get(DEFAULT_QWEN_URL)
        .call()
        .context("Failed downloading Qwen3 GGUF")?;

    let mut dest = File::create(target_path)?;
    let mut reader = resp.into_body().into_reader();
    copy(&mut reader, &mut dest)?;

    info!("System 2 model downloaded successfully to {:?}", target_path);
    Ok(target_path.to_path_buf())
}
