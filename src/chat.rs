use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::handler::{ApiResponse, AppState};

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
    /// 会话 ID（可选），如果不传则创建新会话
    #[serde(default)]
    pub session_id: Option<String>,
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
    /// 会话 ID（新建或已有）
    pub session_id: String,
}

#[derive(Debug, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ──────────────────────────────────────
// 会话记录相关数据结构
// ──────────────────────────────────────

/// 会话索引条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    /// 会话标题（5-10字总结）
    pub title: String,
    /// 使用的模型
    pub model: String,
    /// 消息数量
    pub message_count: usize,
    /// 创建时间
    pub created_at: String,
    /// 最后更新时间
    pub updated_at: String,
}

/// 完整的会话记录
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub title: String,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub created_at: String,
    pub updated_at: String,
}

/// 获取会话列表的响应
#[derive(Debug, Serialize)]
pub struct SessionListResponse {
    pub success: bool,
    pub sessions: Vec<SessionMeta>,
}

/// 获取会话详情的请求
#[derive(Debug, Deserialize)]
pub struct GetSessionRequest {
    pub session_id: String,
}

/// 获取会话详情的响应
#[derive(Debug, Serialize)]
pub struct SessionDetailResponse {
    pub success: bool,
    pub session: SessionRecord,
}

/// 删除会话的请求
#[derive(Debug, Deserialize)]
pub struct DeleteSessionRequest {
    pub session_id: String,
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
// 会话存储管理
// ──────────────────────────────────────

const CHAT_HISTORY_DIR: &str = "data/chat_history";
const SESSIONS_INDEX_FILE: &str = "data/chat_history/sessions.json";

/// 确保聊天记录目录存在
async fn ensure_chat_dir() -> Result<(), std::io::Error> {
    tokio::fs::create_dir_all(CHAT_HISTORY_DIR).await
}

/// 读取会话索引
async fn load_sessions_index() -> Vec<SessionMeta> {
    match tokio::fs::read_to_string(SESSIONS_INDEX_FILE).await {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// 保存会话索引
async fn save_sessions_index(sessions: &[SessionMeta]) -> Result<(), std::io::Error> {
    ensure_chat_dir().await?;
    let json = serde_json::to_string_pretty(sessions).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, format!("序列化索引失败: {}", e))
    })?;
    tokio::fs::write(SESSIONS_INDEX_FILE, json).await
}

/// 加载单个会话记录
async fn load_session(session_id: &str) -> Option<SessionRecord> {
    let path = format!("{}/{}.json", CHAT_HISTORY_DIR, session_id);
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => serde_json::from_str(&content).ok(),
        Err(_) => None,
    }
}

/// 保存单个会话记录
async fn save_session(record: &SessionRecord) -> Result<(), std::io::Error> {
    ensure_chat_dir().await?;
    let path = format!("{}/{}.json", CHAT_HISTORY_DIR, record.session_id);
    let json = serde_json::to_string_pretty(record).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, format!("序列化会话失败: {}", e))
    })?;
    tokio::fs::write(&path, json).await
}

/// 调用大模型总结会话标题（5-10字）
async fn summarize_title(user_message: &str, assistant_reply: &str) -> String {
    let api_key = match std::env::var("DEEPSEEK_API_KEY") {
        Ok(k) => k,
        Err(_) => return "新对话".to_string(),
    };

    let prompt = format!(
        "请用5-10个中文字总结以下对话的主题，只输出标题文字，不要加标点符号、引号或其他任何额外内容。\n\n用户: {}\n助手: {}",
        // 只取前200字避免 token 浪费
        &user_message.chars().take(200).collect::<String>(),
        &assistant_reply.chars().take(200).collect::<String>(),
    );

    let body = DeepSeekRequest {
        model: "deepseek-chat".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt,
        }],
        thinking: None,
    };

    let client = reqwest::Client::new();
    match client
        .post("https://api.deepseek.com/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(ds_resp) = resp.json::<DeepSeekResponse>().await {
                let title = ds_resp
                    .choices
                    .first()
                    .and_then(|c| c.message.content.clone())
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if !title.is_empty() && title.chars().count() <= 15 {
                    return title;
                }
                // 如果模型返回太长，截取前10个字
                if !title.is_empty() {
                    return title.chars().take(10).collect();
                }
            }
            "新对话".to_string()
        }
        Ok(resp) => {
            warn!("总结标题 API 返回错误: {}", resp.status());
            "新对话".to_string()
        }
        Err(e) => {
            warn!("总结标题请求失败: {}", e);
            "新对话".to_string()
        }
    }
}

// ──────────────────────────────────────
// Handlers
// ──────────────────────────────────────

/// POST /api/chat
pub async fn chat(
    State(_state): State<Arc<AppState>>,
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

    let user_message = payload.message.clone();

    // 2. 组装消息列表：历史 + 本次用户消息
    let mut messages = payload.history.clone();
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

    // ── 7. 保存会话记录 ───────────────────────
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let is_new_session = payload.session_id.is_none();

    let session_id = payload
        .session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // 加载或创建会话记录
    let mut record = if is_new_session {
        SessionRecord {
            session_id: session_id.clone(),
            title: "新对话".to_string(),
            model: payload.model.clone(),
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now.clone(),
        }
    } else {
        load_session(&session_id).await.unwrap_or_else(|| SessionRecord {
            session_id: session_id.clone(),
            title: "新对话".to_string(),
            model: payload.model.clone(),
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now.clone(),
        })
    };

    // 追加用户消息和助手回复
    record.messages.push(ChatMessage {
        role: "user".to_string(),
        content: user_message.clone(),
    });
    record.messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: reply.clone(),
    });
    record.updated_at = now.clone();

    // 保存会话记录文件
    if let Err(e) = save_session(&record).await {
        warn!("保存会话记录失败: {}", e);
    }

    // 更新会话索引
    let mut sessions = load_sessions_index().await;
    if let Some(meta) = sessions.iter_mut().find(|s| s.session_id == session_id) {
        meta.message_count = record.messages.len();
        meta.updated_at = now.clone();
    } else {
        sessions.push(SessionMeta {
            session_id: session_id.clone(),
            title: "新对话".to_string(),
            model: payload.model.clone(),
            message_count: record.messages.len(),
            created_at: now.clone(),
            updated_at: now,
        });
    }
    if let Err(e) = save_sessions_index(&sessions).await {
        warn!("保存会话索引失败: {}", e);
    }

    // 新会话时异步总结标题（不阻塞响应）
    if is_new_session {
        let sid = session_id.clone();
        let user_msg = user_message.clone();
        let assistant_reply = reply.clone();
        tokio::spawn(async move {
            let title = summarize_title(&user_msg, &assistant_reply).await;
            info!("会话 {} 标题总结: {}", sid, title);

            // 更新会话记录文件中的标题
            if let Some(mut rec) = load_session(&sid).await {
                rec.title = title.clone();
                if let Err(e) = save_session(&rec).await {
                    warn!("更新会话标题失败: {}", e);
                }
            }

            // 更新索引中的标题
            let mut sessions = load_sessions_index().await;
            if let Some(meta) = sessions.iter_mut().find(|s| s.session_id == sid) {
                meta.title = title;
            }
            if let Err(e) = save_sessions_index(&sessions).await {
                warn!("更新索引标题失败: {}", e);
            }
        });
    }

    Ok(Json(ChatResponse {
        success: true,
        reply,
        reasoning_content,
        usage,
        session_id,
    }))
}

/// GET /api/chat/sessions - 获取所有会话列表
pub async fn get_sessions() -> Json<SessionListResponse> {
    let sessions = load_sessions_index().await;
    Json(SessionListResponse {
        success: true,
        sessions,
    })
}

/// POST /api/chat/session - 获取指定会话的完整对话记录
pub async fn get_session_detail(
    Json(payload): Json<GetSessionRequest>,
) -> Result<Json<SessionDetailResponse>, (StatusCode, Json<ApiResponse>)> {
    let session = load_session(&payload.session_id).await.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiResponse {
                success: false,
                message: format!("会话 {} 不存在", payload.session_id),
            }),
        )
    })?;

    Ok(Json(SessionDetailResponse {
        success: true,
        session,
    }))
}

/// POST /api/chat/session/delete - 删除指定会话
pub async fn delete_session(
    Json(payload): Json<DeleteSessionRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    let session_id = &payload.session_id;

    // 删除会话文件
    let path = format!("{}/{}.json", CHAT_HISTORY_DIR, session_id);
    let _ = tokio::fs::remove_file(&path).await;

    // 从索引中移除
    let mut sessions = load_sessions_index().await;
    sessions.retain(|s| s.session_id != *session_id);
    save_sessions_index(&sessions).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("更新索引失败: {}", e),
            }),
        )
    })?;

    info!("删除会话: {}", session_id);

    Ok(Json(ApiResponse {
        success: true,
        message: format!("会话 {} 已删除", session_id),
    }))
}
