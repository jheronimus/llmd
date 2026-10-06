use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::Tokenizer;

const CLS_TOKEN_ID: i64 = 50281;
const SEP_TOKEN_ID: i64 = 50282;
const PAD_TOKEN_ID: i64 = 50283;
const MASK_TOKEN_ID: i64 = 50284;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QuestionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            QuestionType::Choice => "choice",
            QuestionType::Score => "score",
            QuestionType::Noul => "noul",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QuestionType,
    pub instructions: String,
    pub criteria: Option<serde_json::Value>,
    pub labels: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Answer {
    pub answer: serde_json::Value,
    pub confidence: f32,
    pub act_cost: f32,
}

pub struct Sys1Engine {
    tokenizer: Tokenizer,
    session: Session,
}

impl Sys1Engine {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let tok_path = model_dir.join("tokenizer.json");
        let onnx_path = model_dir.join("model.onnx");

        if !tok_path.exists() {
            return Err(anyhow!("tokenizer not found at {:?}", tok_path));
        }
        if !onnx_path.exists() {
            return Err(anyhow!("ONNX model not found at {:?}", onnx_path));
        }

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow!("failed loading tokenizer: {e}"))?;

        let session = Session::builder()?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("failed loading ONNX model at {:?}", onnx_path))?;

        Ok(Self { tokenizer, session })
    }

    pub fn decide(
        &mut self,
        state: &serde_json::Value,
        questions: &BTreeMap<String, Question>,
    ) -> Result<BTreeMap<String, Answer>> {
        if questions.is_empty() {
            return Ok(BTreeMap::new());
        }

        let state_str = if let Some(s) = state.as_str() {
            s.to_string()
        } else {
            serde_json::to_string(state)?
        };

        struct Seq {
            qid: String,
            input_ids: Vec<i64>,
            marker_positions: Vec<i64>,
            qtype: i64,
            opts: Vec<String>,
        }

        let mut seqs = Vec::with_capacity(questions.len());
        let mut max_seq_len = 0usize;
        let mut max_markers = 0usize;

        for (qid, q) in questions {
            let opts = render_options(q.qtype, q.criteria.as_ref(), q.labels.as_ref())?;
            let (input_ids, markers) = build_sequence(
                &self.tokenizer,
                q.qtype,
                &q.instructions,
                &opts,
                &state_str,
                512,
                192,
            )?;

            max_seq_len = max_seq_len.max(input_ids.len());
            max_markers = max_markers.max(markers.len());

            seqs.push(Seq {
                qid: qid.clone(),
                input_ids,
                marker_positions: markers,
                qtype: q.qtype as i64,
                opts,
            });
        }

        if max_markers == 0 {
            max_markers = 1;
        }

        let batch = seqs.len();
        let mut padded_input_ids = vec![PAD_TOKEN_ID; batch * max_seq_len];
        let mut padded_att_mask = vec![0i64; batch * max_seq_len];
        let mut padded_marker_pos = vec![0i64; batch * max_markers];
        let mut padded_marker_mask = vec![false; batch * max_markers];
        let mut qtypes = vec![0i64; batch];

        for (i, seq) in seqs.iter().enumerate() {
            let off_seq = i * max_seq_len;
            for (j, &id) in seq.input_ids.iter().enumerate() {
                padded_input_ids[off_seq + j] = id;
                padded_att_mask[off_seq + j] = 1;
            }

            let off_mark = i * max_markers;
            for (j, &pos) in seq.marker_positions.iter().enumerate() {
                padded_marker_pos[off_mark + j] = pos;
                padded_marker_mask[off_mark + j] = true;
            }

            qtypes[i] = seq.qtype;
        }

        let t_ids =
            Tensor::from_array(([batch, max_seq_len], padded_input_ids.into_boxed_slice()))?;
        let t_att = Tensor::from_array(([batch, max_seq_len], padded_att_mask.into_boxed_slice()))?;
        let t_pos =
            Tensor::from_array(([batch, max_markers], padded_marker_pos.into_boxed_slice()))?;
        let t_mask =
            Tensor::from_array(([batch, max_markers], padded_marker_mask.into_boxed_slice()))?;
        let t_qtype = Tensor::from_array(([batch], qtypes.into_boxed_slice()))?;

        let inputs = ort::inputs![
            "input_ids" => t_ids,
            "attention_mask" => t_att,
            "marker_pos" => t_pos,
            "marker_mask" => t_mask,
            "qtype" => t_qtype,
        ];

        let outputs = self.session.run(inputs)?;
        let (_shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;

        let mut answers = BTreeMap::new();

        for (i, seq) in seqs.iter().enumerate() {
            let k = seq.marker_positions.len();
            if k == 0 {
                continue;
            }

            let off_mark = i * max_markers;
            let mut logits = Vec::with_capacity(k);
            for m in 0..k {
                logits.push(data[off_mark + m]);
            }

            let probs = softmax(&logits);
            let mut best_idx = 0;
            let mut best_prob = 0.0f32;
            for (idx, &p) in probs.iter().enumerate() {
                if p > best_prob {
                    best_prob = p;
                    best_idx = idx;
                }
            }

            let ans_val = match questions[&seq.qid].qtype {
                QuestionType::Noul => serde_json::Value::Bool(best_idx == 1),
                QuestionType::Score => {
                    serde_json::Value::Number(serde_json::Number::from(best_idx))
                }
                QuestionType::Choice => {
                    let label = if let Some(opt) = seq.opts.get(best_idx) {
                        if let Some((k, _)) = opt.split_once(':') {
                            k.trim().to_string()
                        } else {
                            opt.trim().to_string()
                        }
                    } else {
                        best_idx.to_string()
                    };
                    serde_json::Value::String(label)
                }
            };

            answers.insert(
                seq.qid.clone(),
                Answer {
                    answer: ans_val,
                    confidence: best_prob,
                    act_cost: 0.0,
                },
            );
        }

        Ok(answers)
    }
}

fn render_options(
    qtype: QuestionType,
    criteria: Option<&serde_json::Value>,
    labels: Option<&BTreeMap<String, String>>,
) -> Result<Vec<String>> {
    match qtype {
        QuestionType::Choice => {
            if let Some(c) = criteria {
                if let Some(arr) = c.as_array() {
                    let opts: Vec<String> = arr
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    return Ok(opts);
                }
                if let Some(map) = c.as_object() {
                    let mut keys: Vec<&String> = map.keys().collect();
                    keys.sort();
                    let mut opts = Vec::new();
                    for k in keys {
                        let desc = map[k].as_str().unwrap_or("");
                        if desc.is_empty() {
                            opts.push(k.clone());
                        } else {
                            opts.push(format!("{k}: {desc}"));
                        }
                    }
                    return Ok(opts);
                }
            }
            Ok(vec!["option A".into(), "option B".into()])
        }
        QuestionType::Score => {
            if let Some(c) = criteria {
                if let Some(arr) = c.as_array() {
                    let opts = arr
                        .iter()
                        .enumerate()
                        .map(|(i, v)| format!("level {i}: {}", v.as_str().unwrap_or("")))
                        .collect();
                    return Ok(opts);
                }
            }
            Ok(vec![
                "level 0: low".into(),
                "level 1: medium".into(),
                "level 2: high".into(),
            ])
        }
        QuestionType::Noul => {
            let mut false_lbl = "false".to_string();
            let mut true_lbl = "true".to_string();
            if let Some(lbls) = labels {
                if let Some(f) = lbls.get("false") {
                    false_lbl = f.clone();
                }
                if let Some(t) = lbls.get("true") {
                    true_lbl = t.clone();
                }
            }
            Ok(vec![
                format!("{false_lbl}: no, does not hold"),
                format!("{true_lbl}: yes, holds"),
            ])
        }
    }
}

fn build_sequence(
    tok: &Tokenizer,
    qtype: QuestionType,
    instructions: &str,
    opts: &[String],
    state: &str,
    max_len: usize,
    head_max_len: usize,
) -> Result<(Vec<i64>, Vec<i64>)> {
    let head_prompt = format!("{} question: {}", qtype.as_str(), instructions);
    let head_enc = tok.encode(head_prompt, false).map_err(|e| anyhow!("{e}"))?;
    let mut head_ids: Vec<i64> = head_enc.get_ids().iter().map(|&x| x as i64).collect();

    let mut opt_ids: Vec<Vec<i64>> = Vec::new();
    for opt in opts {
        let enc = tok
            .encode(format!(" {opt}"), false)
            .map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        if ids.len() > 48 {
            ids.truncate(48);
        }
        let mut with_mask = vec![MASK_TOKEN_ID];
        with_mask.extend(ids);
        opt_ids.push(with_mask);
    }

    let mut total_opt_tokens: usize = opt_ids.iter().map(|v| v.len()).sum();
    if total_opt_tokens >= head_max_len {
        let per = (head_max_len.saturating_sub(16) / opts.len().max(1)).max(4);
        for o in &mut opt_ids {
            if o.len() > per {
                o.truncate(per);
            }
        }
        total_opt_tokens = opt_ids.iter().map(|v| v.len()).sum();
    }

    let opt_budget = head_max_len.saturating_sub(total_opt_tokens).max(8);
    if head_ids.len() > opt_budget {
        head_ids.truncate(opt_budget);
    }

    let mut ids = Vec::with_capacity(max_len);
    ids.push(CLS_TOKEN_ID);
    ids.extend(&head_ids);
    ids.push(SEP_TOKEN_ID);

    let mut markers = Vec::new();
    for o in &opt_ids {
        markers.push(ids.len() as i64);
        ids.extend(o);
    }
    ids.push(SEP_TOKEN_ID);

    let state_budget = max_len.saturating_sub(ids.len() + 1);
    let state_enc = tok.encode(state, false).map_err(|e| anyhow!("{e}"))?;
    let mut state_ids: Vec<i64> = state_enc.get_ids().iter().map(|&x| x as i64).collect();
    if state_ids.len() > state_budget {
        state_ids.truncate(state_budget);
    }

    ids.extend(&state_ids);
    ids.push(SEP_TOKEN_ID);

    Ok((ids, markers))
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}
