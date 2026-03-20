use std::path::Path;
use std::sync::Arc;

use arrow_array::StringArray;
use axum::{Json, extract::{Multipart, State}, http::StatusCode};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::embedding;
use crate::extractor;
use crate::handler::{ApiResponse, AppState};

// ──────────────────────────────────────
// 请求 / 响应 数据结构
// ──────────────────────────────────────

// 注：ChatRequest 已废弃，/api/chat 接口现在使用 multipart/form-data 格式
// 各字段通过 multipart 字段解析: message, history, model, thinking, session_id, files

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

/// 聊天临时文件目录
const CHAT_TMP_DIR: &str = "data/chat_tmp";

/// 获取文件扩展名（小写）
fn get_extension(file_name: &str) -> String {
    Path::new(file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// 已上传文件的信息（用于提取文本后拼接上下文）
struct UploadedFile {
    file_name: String,
    extracted_text: String,
}

/// POST /api/chat
///
/// 支持 multipart/form-data 格式，字段说明：
/// - `message`    (必填) 用户消息文本
/// - `history`    (可选) 历史对话 JSON 数组字符串，格式: [{"role":"user","content":"..."},…]
/// - `model`      (可选) 模型名，默认 "deepseek-chat"
/// - `thinking`   (可选) "true" / "false"
/// - `session_id` (可选) 会话 ID
/// - `files`      (可选，可多个) 上传的文件，支持 pdf/docx/doc/xlsx/xls/csv/md/txt
pub async fn chat(
    State(_state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ApiResponse>)> {
    // ── 0. 解析 multipart 字段 ──────────────────
    let mut message: Option<String> = None;
    let mut history: Vec<ChatMessage> = Vec::new();
    let mut model = "deepseek-chat".to_string();
    let mut thinking_flag: Option<bool> = None;
    let mut session_id_input: Option<String> = None;
    let mut uploaded_files: Vec<UploadedFile> = Vec::new();

    // 确保临时目录存在
    let _ = tokio::fs::create_dir_all(CHAT_TMP_DIR).await;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: format!("解析 multipart 请求失败: {}", e),
            }),
        )
    })? {
        let name = field.name().unwrap_or("").to_string();
        let field_file_name = field.file_name().map(|s| s.to_string());

        match name.as_str() {
            "message" => {
                let text = field.text().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse {
                            success: false,
                            message: format!("读取 message 字段失败: {}", e),
                        }),
                    )
                })?;
                message = Some(text);
            }
            "history" => {
                let text = field.text().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse {
                            success: false,
                            message: format!("读取 history 字段失败: {}", e),
                        }),
                    )
                })?;
                // 尝试解析 JSON 数组
                if !text.is_empty() {
                    history = serde_json::from_str(&text).unwrap_or_default();
                }
            }
            "model" => {
                if let Ok(text) = field.text().await {
                    if !text.is_empty() {
                        model = text;
                    }
                }
            }
            "thinking" => {
                if let Ok(text) = field.text().await {
                    thinking_flag = match text.as_str() {
                        "true" | "1" => Some(true),
                        "false" | "0" => Some(false),
                        _ => None,
                    };
                }
            }
            "session_id" => {
                if let Ok(text) = field.text().await {
                    if !text.is_empty() {
                        session_id_input = Some(text);
                    }
                }
            }
            "files" | "file" => {
                // 处理文件上传
                let file_name = field_file_name
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());

                let data = field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse {
                            success: false,
                            message: format!("读取文件 '{}' 数据失败: {}", file_name, e),
                        }),
                    )
                })?;

                info!(
                    "[文件上传] 收到文件: '{}', 大小: {} 字节, 扩展名: '{}'",
                    file_name, data.len(), get_extension(&file_name)
                );

                let extension = get_extension(&file_name);

                // 跳过图片文件（不含可提取文本）
                if matches!(extension.as_str(), "png" | "jpg" | "jpeg") {
                    info!("跳过图片文件 '{}' (聊天上下文不支持图片)", file_name);
                    continue;
                }

                // 写入临时文件
                let tmp_name = format!("{}_{}", uuid::Uuid::new_v4(), file_name);
                let tmp_path = format!("{}/{}", CHAT_TMP_DIR, tmp_name);
                tokio::fs::write(&tmp_path, &data).await.map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ApiResponse {
                            success: false,
                            message: format!("保存临时文件失败: {}", e),
                        }),
                    )
                })?;

                // 提取文本
                let extracted = extractor::extract_text(&tmp_path, &extension);

                // 清理临时文件
                let _ = tokio::fs::remove_file(&tmp_path).await;

                match extracted {
                    Ok(pages) => {
                        let text: String = pages
                            .iter()
                            .map(|p| p.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !text.trim().is_empty() {
                            info!("文件 '{}' 成功提取文本，长度: {} 字符", file_name, text.len());
                            uploaded_files.push(UploadedFile {
                                file_name: file_name.clone(),
                                extracted_text: text,
                            });
                        } else {
                            warn!("文件 '{}' 未提取到有效文本内容", file_name);
                        }
                    }
                    Err(e) => {
                        warn!("文件 '{}' 文本提取失败: {}", file_name, e);
                    }
                }
            }
            _ => {
                // 忽略未知字段
            }
        }
    }

    // ── 1. 校验必填字段 ──────────────────────
    let user_message = message.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "缺少 message 字段".to_string(),
            }),
        )
    })?;

    if user_message.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "message 不能为空".to_string(),
            }),
        ));
    }

    // ── 2. 读取 API Key ──────────────────────
    let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: "未设置 DEEPSEEK_API_KEY 环境变量".to_string(),
            }),
        )
    })?;

    // ── 3. 拼接文件上下文到用户消息 ──────────────────
    let final_message = if uploaded_files.is_empty() {
        user_message.clone()
    } else {
        let mut context_parts = Vec::new();
        for uf in &uploaded_files {
            context_parts.push(format!(
                "【文件: {}】\n{}\n【文件结束】",
                uf.file_name, uf.extracted_text
            ));
        }
        let file_context = context_parts.join("\n\n");
        info!("附加 {} 个文件上下文，总长度: {} 字符", uploaded_files.len(), file_context.len());
        format!(
            "以下是用户上传的文件内容，请结合文件内容回答用户的问题：\n\n{}\n\n用户问题：{}",
            file_context, user_message
        )
    };

    // ── 4. 组装消息列表：历史 + 本次用户消息 ──────────
    let mut messages = history.clone();
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: final_message,
    });

    // ── 5. 构建 thinking 参数 ──────────────────
    let thinking = match thinking_flag {
        Some(true) => Some(ThinkingParam { kind: "enabled".to_string() }),
        Some(false) => Some(ThinkingParam { kind: "disabled".to_string() }),
        None => None,
    };

    let body = DeepSeekRequest {
        model: model.clone(),
        messages,
        thinking,
    };

    info!(
        "调用 DeepSeek Chat API (model={})，消息数: {}，附件数: {}",
        model,
        body.messages.len(),
        uploaded_files.len()
    );

    // ── 6. 调用 DeepSeek API ──────────────────
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

    // ── 7. 检查 HTTP 状态码 ──────────────────
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

    // ── 8. 解析响应 ──────────────────────────
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

    // ── 9. 保存会话记录 ───────────────────────
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let is_new_session = session_id_input.is_none();

    let session_id = session_id_input
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // 加载或创建会话记录
    let mut record = if is_new_session {
        SessionRecord {
            session_id: session_id.clone(),
            title: "新对话".to_string(),
            model: model.clone(),
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now.clone(),
        }
    } else {
        load_session(&session_id).await.unwrap_or_else(|| SessionRecord {
            session_id: session_id.clone(),
            title: "新对话".to_string(),
            model: model.clone(),
            messages: Vec::new(),
            created_at: now.clone(),
            updated_at: now.clone(),
        })
    };

    // 追加用户消息和助手回复（保存原始用户消息，不含文件上下文）
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
            model: model.clone(),
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

// ──────────────────────────────────────
// RAG 对话（优先检索本地向量库）
// ──────────────────────────────────────

/// RAG 对话请求体
#[derive(Debug, Deserialize)]
pub struct RagChatRequest {
    /// 用户输入的消息
    pub message: String,
    /// 历史对话记录（可选）
    #[serde(default)]
    pub history: Vec<ChatMessage>,
    /// 使用的模型，默认 deepseek-chat
    #[serde(default = "rag_default_model")]
    pub model: String,
    /// 是否开启思考模式
    #[serde(default)]
    pub thinking: Option<bool>,
    /// 会话 ID（可选）
    #[serde(default)]
    pub session_id: Option<String>,
    /// 向量搜索返回的最大结果数，默认 5
    #[serde(default = "rag_default_top_k")]
    pub top_k: usize,
}

fn rag_default_model() -> String {
    "deepseek-chat".to_string()
}

fn rag_default_top_k() -> usize {
    5
}

/// 向量搜索命中的片段
#[derive(Debug)]
struct RagHit {
    text: String,
    file_path: String,
    distance: f32,
}

/// POST /api/chat/rag
///
/// 优先检索本地向量库（所有已上传文档），将相关内容作为上下文发给大模型
pub async fn chat_rag(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RagChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, Json<ApiResponse>)> {
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
    info!("[RAG] 收到用户问题: {}", user_message.chars().take(100).collect::<String>());

    // ── 1. 将用户问题本地向量化 ──────────────────────
    let query_texts = vec![user_message.clone()];
    let query_vectors = embedding::embed_texts(&query_texts, &api_key)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("用户问题向量化失败: {}", e),
                }),
            )
        })?;

    let query_vector = query_vectors.into_iter().next().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: "向量化结果为空".to_string(),
            }),
        )
    })?;

    info!("[RAG] 用户问题向量化完成，维度: {}", query_vector.len());

    // ── 2. 遍历所有 files_* 表进行向量搜索 ──────────────
    let table_names = state.db.table_names().execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("获取表列表失败: {}", e),
            }),
        )
    })?;

    let files_tables: Vec<&String> = table_names
        .iter()
        .filter(|n| n.starts_with("files_"))
        .collect();

    info!("[RAG] 检索 {} 个文档表: {:?}", files_tables.len(), files_tables);

    let mut all_hits: Vec<RagHit> = Vec::new();

    for table_name in &files_tables {
        let table = match state.db.open_table(table_name.as_str()).execute().await {
            Ok(t) => t,
            Err(e) => {
                warn!("[RAG] 打开表 '{}' 失败: {}", table_name, e);
                continue;
            }
        };

        // 检查表是否有数据
        let row_count = table.count_rows(None).await.unwrap_or(0);
        if row_count == 0 {
            continue;
        }

        let results = match table
            .vector_search(query_vector.clone())
            .map_err(|e| format!("{}", e))
        {
            Ok(query) => match query.limit(payload.top_k).execute().await {
                Ok(stream) => stream,
                Err(e) => {
                    warn!("[RAG] 表 '{}' 向量搜索执行失败: {}", table_name, e);
                    continue;
                }
            },
            Err(e) => {
                warn!("[RAG] 表 '{}' 创建搜索查询失败: {}", table_name, e);
                continue;
            }
        };

        let batches: Vec<arrow_array::RecordBatch> = match results.try_collect().await {
            Ok(b) => b,
            Err(e) => {
                warn!("[RAG] 表 '{}' 收集搜索结果失败: {}", table_name, e);
                continue;
            }
        };

        for batch in &batches {
            let text_col = batch
                .column_by_name("text")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let path_col = batch
                .column_by_name("file_path")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let dist_col = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());

            if let (Some(texts), Some(paths), Some(dists)) = (text_col, path_col, dist_col) {
                for i in 0..batch.num_rows() {
                    all_hits.push(RagHit {
                        text: texts.value(i).to_string(),
                        file_path: paths.value(i).to_string(),
                        distance: dists.value(i),
                    });
                }
            }
        }
    }

    // 按距离排序，取 top_k
    all_hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    all_hits.truncate(payload.top_k);

    // ── 3. 打印检索结果日志 ──────────────────────
    if all_hits.is_empty() {
        info!("[RAG] ❌ 本地向量库未检索到相关内容，将直接提问大模型");
    } else {
        info!("[RAG] ✅ 本地向量库命中 {} 条相关内容:", all_hits.len());
        for (i, hit) in all_hits.iter().enumerate() {
            info!(
                "[RAG]   #{}: 距离={:.4}, 来源='{}', 内容前80字: {}",
                i + 1,
                hit.distance,
                hit.file_path,
                hit.text.chars().take(80).collect::<String>()
            );
        }
    }

    // ── 4. 构建发送给大模型的消息 ──────────────────────
    let final_message = if all_hits.is_empty() {
        user_message.clone()
    } else {
        let mut context_parts = Vec::new();
        for hit in &all_hits {
            context_parts.push(format!(
                "【来源: {}】\n{}",
                hit.file_path, hit.text
            ));
        }
        let rag_context = context_parts.join("\n\n");
        format!(
            "以下是从知识库中检索到的与用户问题相关的参考内容：\n\n{}\n\n请结合以上参考内容回答用户的问题。如果参考内容与问题无关，请忽略参考内容直接回答。\n\n用户问题：{}",
            rag_context, user_message
        )
    };

    // ── 5. 组装消息列表 ──────────────────────
    let mut messages = payload.history.clone();
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: final_message,
    });

    let thinking = match payload.thinking {
        Some(true) => Some(ThinkingParam { kind: "enabled".to_string() }),
        Some(false) => Some(ThinkingParam { kind: "disabled".to_string() }),
        None => None,
    };

    let body = DeepSeekRequest {
        model: payload.model.clone(),
        messages,
        thinking,
    };

    info!(
        "[RAG] 调用 DeepSeek (model={})，消息数: {}，本地命中: {} 条",
        payload.model,
        body.messages.len(),
        all_hits.len()
    );

    // ── 6. 调用 DeepSeek API ──────────────────────
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

    info!("[RAG] DeepSeek 回复长度: {} 字符", reply.len());

    // ── 7. 保存会话记录 ──────────────────────
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let is_new_session = payload.session_id.is_none();

    let session_id = payload
        .session_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

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

    record.messages.push(ChatMessage {
        role: "user".to_string(),
        content: user_message.clone(),
    });
    record.messages.push(ChatMessage {
        role: "assistant".to_string(),
        content: reply.clone(),
    });
    record.updated_at = now.clone();

    if let Err(e) = save_session(&record).await {
        warn!("保存会话记录失败: {}", e);
    }

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

    if is_new_session {
        let sid = session_id.clone();
        let user_msg = user_message.clone();
        let assistant_reply = reply.clone();
        tokio::spawn(async move {
            let title = summarize_title(&user_msg, &assistant_reply).await;
            info!("会话 {} 标题总结: {}", sid, title);
            if let Some(mut rec) = load_session(&sid).await {
                rec.title = title.clone();
                if let Err(e) = save_session(&rec).await {
                    warn!("更新会话标题失败: {}", e);
                }
            }
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
