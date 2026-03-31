use std::sync::Arc;
use arrow_array::{RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
};
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use serde::{Deserialize, Serialize};
use tracing::info;
use crate::handler::{ApiResponse, AppState};

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

const SENSITIVE_WORDS_TABLE: &str = "sensitive_words";

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
