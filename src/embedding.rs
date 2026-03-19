use serde::{Deserialize, Serialize};
use tracing::info;

/// DeepSeek Embedding API 的向量维度
pub const EMBEDDING_DIM: i32 = 768;

/// DeepSeek Embedding API 端点
const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/v1/embeddings";

/// DeepSeek Embedding 模型名称
const DEEPSEEK_MODEL: &str = "deepseek-embedding-v2";

/// Embedding 请求体（兼容 OpenAI 格式）
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

/// Embedding 响应体
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// 单条 Embedding 数据
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// 调用 DeepSeek Embedding API，将文本列表转换为向量
///
/// # 参数
/// - `texts`: 待向量化的文本列表
/// - `api_key`: DeepSeek API Key
///
/// # 返回
/// - 与 texts 一一对应的向量列表
pub async fn embed_texts(
    texts: &[String],
    api_key: &str,
) -> Result<Vec<Vec<f32>>, String> {
    if texts.is_empty() {
        return Ok(vec![]);
    }

    let client = reqwest::Client::new();

    // DeepSeek API 支持批量输入，但需注意 token 限制
    // 我们分批处理，每批最多 16 条文本
    let batch_size = 16;
    let mut all_embeddings = Vec::new();

    for (batch_idx, batch) in texts.chunks(batch_size).enumerate() {
        let request = EmbeddingRequest {
            model: DEEPSEEK_MODEL.to_string(),
            input: batch.to_vec(),
        };

        let response = client
            .post(DEEPSEEK_API_URL)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("调用 DeepSeek API 失败: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "DeepSeek API 返回错误 ({}): {}",
                status, body
            ));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| format!("解析 DeepSeek API 响应失败: {}", e))?;

        for data in embedding_response.data {
            all_embeddings.push(data.embedding);
        }

        info!(
            "Embedding 批次 {} 完成，处理 {} 条文本",
            batch_idx + 1,
            batch.len()
        );
    }

    Ok(all_embeddings)
}
