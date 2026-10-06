use anyhow::{anyhow, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Sys2Engine {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    lock: Mutex<()>,
}

impl Sys2Engine {
    pub fn new(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(anyhow!("model file not found at {:?}", model_path));
        }

        let backend = Arc::new(LlamaBackend::init()?);
        let model_params = LlamaModelParams::default();
        let model = Arc::new(LlamaModel::load_from_file(
            &backend,
            model_path,
            &model_params,
        )?);

        Ok(Self {
            backend,
            model,
            lock: Mutex::new(()),
        })
    }

    pub async fn generate(
        &self,
        system: &str,
        prompt: &str,
        grammar: Option<&str>,
        temp: f32,
        top_p: f32,
        max_tokens: usize,
    ) -> Result<String> {
        let _guard = self.lock.lock().await;

        let formatted = format_chatml(system, prompt);
        let tokens = self.model.str_to_token(&formatted, AddBos::Never)?;

        let n_ctx = (tokens.len() + max_tokens + 64).max(2048) as u32;
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

        let mut batch = LlamaBatch::new(n_ctx as usize, 1);
        batch.add_sequence(&tokens, 0, false)?;
        ctx.decode(&mut batch)?;

        let mut samplers = Vec::new();
        if let Some(g) = grammar {
            let trimmed = g.trim();
            if !trimmed.is_empty() {
                let g_sampler = LlamaSampler::grammar(&self.model, trimmed, "root")
                    .map_err(|e| anyhow!("failed to initialize grammar sampler: {e}"))?;
                samplers.push(g_sampler);
            }
        }
        samplers.push(LlamaSampler::penalties(
            self.model.n_vocab(),
            64,
            1.1,
            0.0,
            0.0,
        ));
        samplers.push(LlamaSampler::temp(if temp <= 0.0 { 0.2 } else { temp }));
        samplers.push(LlamaSampler::top_p(
            if top_p <= 0.0 { 0.9 } else { top_p },
            1,
        ));
        samplers.push(LlamaSampler::dist(1337));

        let mut sampler = LlamaSampler::chain_simple(samplers);
        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_past = tokens.len() as i32;

        let limit = if max_tokens == 0 { 1024 } else { max_tokens };

        for _ in 0..limit {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, false, None)?;
            output.push_str(&piece);

            if output.contains("<|im_end|>") || output.contains("<end_of_turn>") {
                break;
            }

            if grammar.is_some() {
                let trimmed = output.trim();
                if (trimmed.starts_with('{') && trimmed.ends_with('}'))
                    || (trimmed.starts_with('[') && trimmed.ends_with(']'))
                {
                    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
                        break;
                    }
                }
            }

            batch.clear();
            batch.add(token, n_past, &[0], true)?;
            n_past += 1;
            ctx.decode(&mut batch)?;
        }

        Ok(clean_response(&output))
    }
}

fn format_chatml(system: &str, user: &str) -> String {
    let mut s = String::new();
    s.push_str("<|im_start|>system\n");
    if !system.trim().is_empty() {
        s.push_str(system.trim());
        s.push('\n');
    }
    s.push_str("Do not output thoughts. Directly output valid answer only.\n<|im_end|>\n");
    s.push_str("<|im_start|>user\n");
    s.push_str(user.trim());
    s.push_str("\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n");
    s
}

fn clean_response(raw: &str) -> String {
    let mut s = raw.trim();
    if let Some(idx) = s.find("</think>") {
        s = s[idx + "</think>".len()..].trim();
    }
    if let Some(stripped) = s.strip_suffix("<|im_end|>") {
        s = stripped.trim();
    }
    if let Some(stripped) = s.strip_suffix("<end_of_turn>") {
        s = stripped.trim();
    }
    s.to_string()
}
