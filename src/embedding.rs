use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use fastembed::{InitOptionsUserDefined, TextEmbedding, UserDefinedEmbeddingModel};

pub const EMBEDDING_DIM: i32 = 512;

static EMBEDDING_MODEL: OnceLock<Result<Mutex<TextEmbedding>, String>> = OnceLock::new();

fn load_file(path: PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(&path).map_err(|e| format!("无法读取离线模型文件 {}: {}", path.display(), e))
}

pub fn init_model() -> Result<&'static Mutex<TextEmbedding>, &'static str> {
    EMBEDDING_MODEL
        .get_or_init(|| {
            tracing::info!("正在加载本地离线中文向量模型...");

            let model_dir = std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("data")
                .join("bge-small-zh-v1.5-onnx");

            if !Path::new(&model_dir).exists() {
                return Err(format!(
                    "离线模型目录不存在: {}。请将模型放到 data/bge-small-zh-v1.5-onnx",
                    model_dir.display()
                ));
            }

            let tokenizers = fastembed::TokenizerFiles {
                tokenizer_file: load_file(model_dir.join("tokenizer.json"))?,
                config_file: load_file(model_dir.join("config.json"))?,
                special_tokens_map_file: load_file(model_dir.join("special_tokens_map.json"))?,
                tokenizer_config_file: load_file(model_dir.join("tokenizer_config.json"))?,
            };

            let model_param = UserDefinedEmbeddingModel::new(
                load_file(model_dir.join("onnx").join("model.onnx"))?,
                tokenizers,
            );

            let model = TextEmbedding::try_new_from_user_defined(
                model_param,
                InitOptionsUserDefined::default(),
            )
            .map_err(|e| {
                format!(
                    "无法加载离线模型，请确认 data/bge-small-zh-v1.5-onnx 内容完整: {}",
                    e
                )
            })?;

            tracing::info!("本地离线向量模型加载完成");
            Ok(Mutex::new(model))
        })
        .as_ref()
        .map_err(|e| e.as_str())
}

pub async fn embed_texts(texts: &[String], _api_key: &str) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let model_mutex = init_model().map_err(|e| e.to_string())?;
    let texts_clone = texts.to_vec();

    let embeddings = tokio::task::spawn_blocking(move || {
        let mut model = model_mutex
            .lock()
            .map_err(|_| "大模型 Mutex 锁异常".to_string())?;
        model.embed(texts_clone, None).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| format!("执行线程池任务失败: {:?}", e))?
    .map_err(|e| format!("模型内部处理发生错误: {:?}", e))?;

    Ok(embeddings)
}
