use std::sync::{OnceLock, Mutex};
use fastembed::{TextEmbedding, InitOptionsUserDefined, UserDefinedEmbeddingModel};

/// 我们选用的 BGESmallZHV15 的确是 512 维的
pub const EMBEDDING_DIM: i32 = 512;

// 全局唯一的模型实例，赋予 Mutex 处理模型底层并行并发状态
static EMBEDDING_MODEL: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

/// 初始化本地模型并在后台加载到内存（只执行一次）
pub fn init_model() -> &'static Mutex<TextEmbedding> {
    EMBEDDING_MODEL.get_or_init(|| {
        tracing::info!("正在装载本地纯离线中文向量模型...");

        // 获取当前工作目录，并构建离线模型路径
        let model_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("data").join("bge-small-zh-v1.5-onnx");

        let read_file = |path: std::path::PathBuf| -> Vec<u8> {
            std::fs::read(&path).unwrap_or_else(|_| panic!("抱歉，无法读取离线模型相关文件: {:?}", path))
        };

        let tokenizers = fastembed::TokenizerFiles {
            tokenizer_file: read_file(model_dir.join("tokenizer.json")),
            config_file: read_file(model_dir.join("config.json")),
            special_tokens_map_file: read_file(model_dir.join("special_tokens_map.json")),
            tokenizer_config_file: read_file(model_dir.join("tokenizer_config.json")),
        };

        let model_param = UserDefinedEmbeddingModel::new(
            read_file(model_dir.join("onnx").join("model.onnx")),
            tokenizers
        );

        let model = TextEmbedding::try_new_from_user_defined(
            model_param,
            InitOptionsUserDefined::default()
        )
        .expect("无法加载离线模型文件，请确保执行了下载脚本或模型放置在正确的 data/bge-small-zh-v1.5-onnx 目录下");
        
        tracing::info!("⚡ 完全离线本地特征向量引擎编译及挂载就绪！");
        Mutex::new(model)
    })
}

/// 调用本地 Embedding 向量化引擎
pub async fn embed_texts(
    texts: &[String],
    _api_key: &str, // 保留签名兼容旧代码
) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    // 取得静态生命周期的 Mutex 引用
    let model_mutex = init_model();

    // 在针对大模型推理计算时，必须放到专门的阻塞线程中进行防止卡死 Tokio 事件循环
    let texts_clone = texts.to_vec();
    let embeddings = tokio::task::spawn_blocking(move || {
        let mut model = model_mutex.lock().map_err(|_| "大模型 Mutex 锁异常崩溃")?;
        model.embed(texts_clone, None).map_err(|e| format!("{}", e))
    })
    .await
    .map_err(|e| format!("执行线程池崩溃: {:?}", e))?
    .map_err(|e| format!("模型内部处理发生报错: {:?}", e))?;

    Ok(embeddings)
}
