use std::sync::Arc;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use tracing::info;
use crate::handler::{ApiResponse, AppState};
use regex::Regex;

/// 敏感词结构
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SensitiveWord {
    pub word: String,
    pub created_at: String,
    pub match_strategy: String, // 例如: "exact", "fuzzy"
    pub replace_strategy: String, // 例如: "mask", "remove", "custom_text"
}

/// 新增敏感词请求体
#[derive(Debug, Deserialize)]
pub struct CreateSensitiveWordRequest {
    pub word: String,
    pub match_strategy: String,
    pub replace_strategy: String,
}

pub const SENSITIVE_WORDS_TABLE: &str = "sensitive_words";

fn validate_sensitive_word(word: &str, match_strategy: &str) -> Result<(), String> {
    let word = word.trim();
    if word.is_empty() {
        return Err("敏感词内容不能为空".to_string());
    }

    let strategy = match_strategy.trim();
    if strategy.is_empty() {
        return Err("match_strategy 不能为空".to_string());
    }

    let strategy_lower = strategy.to_lowercase();
    if strategy_lower == "exact" {
        return Ok(());
    }

    if matches!(strategy_lower.as_str(), "fuzzy" | "regex" | "re") {
        return Err("match_strategy 不再支持关键字 fuzzy/regex/re，请直接填写正则到 match_strategy 字段".to_string());
    }

    // match_strategy 直接作为正则校验
    let mut patterns = vec![strategy.to_string()];
    if strategy.contains("\\\\") {
        let normalized = strategy.replace("\\\\", "\\");
        if normalized != strategy {
            patterns.push(normalized);
        }
    }
    let mut ok = false;
    let mut last_err: Option<String> = None;
    for p in patterns {
        match Regex::new(&p) {
            Ok(_) => {
                ok = true;
                break;
            }
            Err(e) => last_err = Some(e.to_string()),
        }
    }
    if !ok {
        return Err(format!("正则表达式无效: {}", last_err.unwrap_or_default()));
    }

    Ok(())
}

fn make_sensitive_words_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("word", DataType::Utf8, false),
        Field::new("created_at", DataType::Utf8, false),
        Field::new("match_strategy", DataType::Utf8, false),
        Field::new("replace_strategy", DataType::Utf8, false),
    ]))
}

/// 初始化敏感词表
pub async fn init_sensitive_table(db: &lancedb::Connection) -> Result<(), Box<dyn std::error::Error>> {
    let table_names = db.table_names().execute().await?;
    if !table_names.contains(&SENSITIVE_WORDS_TABLE.to_string()) {
        let schema = make_sensitive_words_schema();
        let batches = RecordBatchIterator::new(vec![], schema);
        db.create_table(SENSITIVE_WORDS_TABLE, Box::new(batches))
            .execute()
            .await?;
        info!("表 '{}' 创建成功", SENSITIVE_WORDS_TABLE);
    }
    Ok(())
}

/// 获取所有敏感词名字 (也可以返回完整信息)
pub async fn list_sensitive_words(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<SensitiveWord>>, (StatusCode, Json<ApiResponse>)> {
    let table = state.db.open_table(SENSITIVE_WORDS_TABLE).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("打开敏感词表失败: {}", e),
            }),
        )
    })?;

    let results = table.query().execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("查询敏感词失败: {}", e),
            }),
        )
    })?;

    let batches: Vec<RecordBatch> = results.try_collect().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("收集查询结果失败: {}", e),
            }),
        )
    })?;

    let mut words = Vec::new();
    for batch in &batches {
        let word_col = batch.column_by_name("word").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let created_at_col = batch.column_by_name("created_at").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let match_strategy_col = batch.column_by_name("match_strategy").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let replace_strategy_col = batch.column_by_name("replace_strategy").and_then(|c| c.as_any().downcast_ref::<StringArray>());

        if let (Some(w), Some(c), Some(m), Some(r)) = (word_col, created_at_col, match_strategy_col, replace_strategy_col) {
            for i in 0..batch.num_rows() {
                words.push(SensitiveWord {
                    word: w.value(i).to_string(),
                    created_at: c.value(i).to_string(),
                    match_strategy: m.value(i).to_string(),
                    replace_strategy: r.value(i).to_string(),
                });
            }
        }
    }

    Ok(Json(words))
}

/// 新增敏感词
pub async fn add_sensitive_word(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateSensitiveWordRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    if let Err(msg) = validate_sensitive_word(&payload.word, &payload.match_strategy) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: msg,
            }),
        ));
    }
    let table = state.db.open_table(SENSITIVE_WORDS_TABLE).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("打开敏感词表失败: {}", e),
            }),
        )
    })?;

    let created_at = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    let schema = make_sensitive_words_schema();

    let word_array = StringArray::from(vec![payload.word.as_str()]);
    let created_at_array = StringArray::from(vec![created_at.as_str()]);
    let match_strategy_array = StringArray::from(vec![payload.match_strategy.as_str()]);
    let replace_strategy_array = StringArray::from(vec![payload.replace_strategy.as_str()]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(word_array),
            Arc::new(created_at_array),
            Arc::new(match_strategy_array),
            Arc::new(replace_strategy_array),
        ],
    ).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("构建 RecordBatch 失败: {}", e),
            }),
        )
    })?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    table.add(Box::new(batches)).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("插入敏感词失败: {}", e),
            }),
        )
    })?;

    info!("成功添加敏感词: {}", payload.word);

    Ok(Json(ApiResponse {
        success: true,
        message: "敏感词添加成功".to_string(),
    }))
}
/// 更新敏感词请求体
#[derive(Debug, Deserialize)]
pub struct UpdateSensitiveWordRequest {
    pub old_word: String,          // 用于定位源记录
    pub new_word: String,          // 新的敏感词内容
    pub match_strategy: String,
    pub replace_strategy: String,
}

/// 更新敏感词
pub async fn update_sensitive_word(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<UpdateSensitiveWordRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    if let Err(msg) = validate_sensitive_word(&payload.new_word, &payload.match_strategy) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: msg,
            }),
        ));
    }
    let table = state.db.open_table(SENSITIVE_WORDS_TABLE).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("打开敏感词表失败: {}", e),
            }),
        )
    })?;

    // 1. 查找现有记录以获取其创建时间
    let filter = format!("word = '{}'", payload.old_word.replace("'", "''"));
    let query = table.query();
    let results = query.only_if(filter.as_str()).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("查询原始敏感词失败: {}", e),
            }),
        )
    })?;

    let batches: Vec<RecordBatch> = results.try_collect().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("收集查询结果失败: {}", e),
            }),
        )
    })?;

    let mut created_at = None;
    for batch in &batches {
        let word_col = batch.column_by_name("word").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let created_at_col = batch.column_by_name("created_at").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        if let (Some(w), Some(c)) = (word_col, created_at_col) {
            for i in 0..batch.num_rows() {
                if w.value(i) == payload.old_word {
                    created_at = Some(c.value(i).to_string());
                    break;
                }
            }
        }
        if created_at.is_some() { break; }
    }

    let created_at = match created_at {
        Some(t) => t,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    success: false,
                    message: format!("未找到敏感词: {}", payload.old_word),
                }),
            ));
        }
    };

    // 2. 删除旧记录
    table.delete(&filter).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("删除旧敏感词失败: {}", e),
            }),
        )
    })?;

    // 3. 插入新记录（保持原创建时间）
    let schema = make_sensitive_words_schema();
    let word_array = StringArray::from(vec![payload.new_word.as_str()]);
    let created_at_array = StringArray::from(vec![created_at.as_str()]);
    let match_strategy_array = StringArray::from(vec![payload.match_strategy.as_str()]);
    let replace_strategy_array = StringArray::from(vec![payload.replace_strategy.as_str()]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(word_array),
            Arc::new(created_at_array),
            Arc::new(match_strategy_array),
            Arc::new(replace_strategy_array),
        ],
    ).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("构建 RecordBatch 失败: {}", e),
            }),
        )
    })?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    table.add(Box::new(batches)).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("重新插入敏感词失败: {}", e),
            }),
        )
    })?;

    info!("成功更新敏感词: {} -> {}", payload.old_word, payload.new_word);

    Ok(Json(ApiResponse {
        success: true,
        message: "敏感词更新成功".to_string(),
    }))
}
