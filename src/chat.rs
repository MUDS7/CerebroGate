use axum::{Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::handler::ApiResponse;

// ──────────────────────────────────────
// 请求 / 响应 数据结构
// ──────────────────────────────────────

/// 前端发来的对话请求
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// 用户输入的消息
    pub message: String,
    /// 历史对话记录（可选），格式与 DeepSeek/OpenAI 一致
    #[serde(default)]
    pub history: Vec<ChatMessage>,
    /// 使用的模型，默认 deepseek-chat
    /// 可选值: "deepseek-chat"（普通模式）、"deepseek-reasoner"（思考模式）
    #[serde(default = "default_model")]
    pub model: String,
    /// 是否开启思考模式（可选）
    /// 设为 true 时等同于 {"type": "enabled"}，也可通过 model="deepseek-reasoner" 开启
    #[serde(default)]
    pub thinking: Option<bool>,
}

fn default_model() -> String {
    "deepseek-chat".to_string()
}

/// 单条对话消息（注意：多轮对话拼接时不要把 reasoning_content 传入 history）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// 返回给前端的对话响应
#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub success: bool,
    /// 最终回答内容
    pub reply: String,
    /// 思维链内容（仅思考模式下有值）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// token 用量统计
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ──────────────────────────────────────
// DeepSeek API 的请求 / 响应结构
// ──────────────────────────────────────

#[derive(Serialize)]
struct DeepSeekRequest {
    model: String,
    messages: Vec<ChatMessage>,
    /// 控制思考模式开关
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingParam>,
}

#[derive(Serialize)]
struct ThinkingParam {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct DeepSeekResponse {
    choices: Vec<DeepSeekChoice>,
    usage: Option<DeepSeekUsage>,
}

#[derive(Deserialize)]
struct DeepSeekChoice {
    message: DeepSeekMessage,
}

#[derive(Deserialize)]
struct DeepSeekMessage {
    /// 最终回答（普通模式必有；思考模式下也可能为 null）
    content: Option<String>,
    /// 思维链内容（仅思考模式返回）
    reasoning_content: Option<String>,
}

#[derive(Deserialize)]
struct DeepSeekUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

// ──────────────────────────────────────
// Handler
// ──────────────────────────────────────

/// POST /api/chat
pub async fn chat(
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ApiResponse>)> {
    // 1. 读取 API Key
    let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: "未设置 DEEPSEEK_API_KEY 环境变量".to_string(),
            }),
        )
    })?;

    // 2. 组装消息列表：历史 + 本次用户消息
    let mut messages = payload.history;
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: payload.message.clone(),
    });

    // 3. 构建 thinking 参数
    let thinking = match payload.thinking {
        Some(true) => Some(ThinkingParam { kind: "enabled".to_string() }),
        Some(false) => Some(ThinkingParam { kind: "disabled".to_string() }),
        None => None, // 不传此参数，由 model 名决定
    };

    let body = DeepSeekRequest {
        model: payload.model.clone(),
        messages,
        thinking,
    };

    info!(
        "调用 DeepSeek Chat API (model={})，消息数: {}",
        payload.model,
        body.messages.len()
    );

    // 4. 调用 DeepSeek API
    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.deepseek.com/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(ApiResponse {
                    success: false,
                    message: format!("请求 DeepSeek API 失败: {}", e),
                }),
            )
        })?;

    // 5. 检查 HTTP 状态码
    if !resp.status().is_success() {
        let status = resp.status();
        let error_text = resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(ApiResponse {
                success: false,
                message: format!("DeepSeek API 返回错误 ({}): {}", status, error_text),
            }),
        ));
    }

    // 6. 解析响应
    let ds_resp: DeepSeekResponse = resp.json().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("解析 DeepSeek 响应失败: {}", e),
            }),
        )
    })?;

    let first_choice = ds_resp.choices.first();

    let reply = first_choice
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    let reasoning_content = first_choice
        .and_then(|c| c.message.reasoning_content.clone())
        .filter(|s| !s.is_empty());

    let usage = ds_resp.usage.map(|u| UsageInfo {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    info!(
        "DeepSeek 回复长度: {} 字符{}",
        reply.len(),
        if reasoning_content.is_some() { "（含思维链）" } else { "" }
    );

    Ok(Json(ChatResponse {
        success: true,
        reply,
        reasoning_content,
        usage,
    }))
}
