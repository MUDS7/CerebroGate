use std::sync::Arc;

use futures::TryStreamExt;

use arrow_array::{Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
};
use lancedb::{Connection, query::{ExecutableQuery, QueryBase}};
use serde::{Deserialize, Serialize};
use tracing::info;

/// 应用共享状态
pub struct AppState {
    pub db: Connection,
}

/// 添加项目的请求体
#[derive(Debug, Deserialize)]
pub struct AddItemRequest {
    id: String,
    text: String,
    vector: Vec<f32>,
}

/// 响应体
#[derive(Debug, Serialize)]
pub struct ApiResponse {
    pub success: bool,
    pub message: String,
}

/// 搜索请求体
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    vector: Vec<f32>,
    limit: Option<usize>,
}

/// 搜索结果
#[derive(Debug, Serialize)]
pub struct SearchResult {
    id: String,
    text: String,
    distance: f32,
}

const VECTOR_DIM: i32 = 128;

fn make_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                VECTOR_DIM,
            ),
            false,
        ),
    ]))
}

/// 初始化 LanceDB 表
pub async fn init_table(db: &Connection) -> Result<(), Box<dyn std::error::Error>> {
    let table_name = "items";

    // 检查表是否已存在
    let table_names = db.table_names().execute().await?;
    if table_names.contains(&table_name.to_string()) {
        info!("表 '{}' 已存在，跳过创建", table_name);
        return Ok(());
    }

    let schema = make_schema();

    // 创建空表
    let batches = RecordBatchIterator::new(vec![], schema.clone());
    db.create_table(table_name, Box::new(batches))
        .execute()
        .await?;

    info!("表 '{}' 创建成功", table_name);
    Ok(())
}

/// 健康检查
pub async fn health_check() -> Json<ApiResponse> {
    Json(ApiResponse {
        success: true,
        message: "CerebroGate 服务运行中".to_string(),
    })
}

/// 添加项目到 LanceDB
pub async fn add_item(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AddItemRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    let table = state
        .db
        .open_table("items")
        .execute()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("打开表失败: {}", e),
                }),
            )
        })?;

    let schema = make_schema();

    let id_array = StringArray::from(vec![payload.id.as_str()]);
    let text_array = StringArray::from(vec![payload.text.as_str()]);

    // 构建 FixedSizeList 向量列
    let values = Float32Array::from(payload.vector);
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_array =
        arrow_array::FixedSizeListArray::try_new(field, VECTOR_DIM, Arc::new(values), None)
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        success: false,
                        message: format!("向量维度错误: {}", e),
                    }),
                )
            })?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_array),
            Arc::new(text_array),
            Arc::new(vector_array),
        ],
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("构建数据失败: {}", e),
            }),
        )
    })?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);

    table
        .add(Box::new(batches))
        .execute()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("插入数据失败: {}", e),
                }),
            )
        })?;

    info!("成功添加项目: {}", payload.id);

    Ok(Json(ApiResponse {
        success: true,
        message: "项目添加成功".to_string(),
    }))
}

/// 向量搜索
pub async fn search(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<Vec<SearchResult>>, (StatusCode, Json<ApiResponse>)> {
    let table = state
        .db
        .open_table("items")
        .execute()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("打开表失败: {}", e),
                }),
            )
        })?;

    let limit = payload.limit.unwrap_or(10);

    let results = table
        .vector_search(payload.vector)
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse {
                    success: false,
                    message: format!("搜索参数错误: {}", e),
                }),
            )
        })?
        .limit(limit)
        .execute()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("搜索失败: {}", e),
                }),
            )
        })?;

    // 收集流式结果
    let batches: Vec<RecordBatch> = results.try_collect().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("收集搜索结果失败: {}", e),
            }),
        )
    })?;

    // 解析搜索结果
    let mut search_results = Vec::new();
    for batch in &batches {
        let id_col = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let text_col = batch
            .column_by_name("text")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let dist_col = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

        if let (Some(ids), Some(texts), Some(dists)) = (id_col, text_col, dist_col) {
            for i in 0..batch.num_rows() {
                search_results.push(SearchResult {
                    id: ids.value(i).to_string(),
                    text: texts.value(i).to_string(),
                    distance: dists.value(i),
                });
            }
        }
    }

    Ok(Json(search_results))
}

/// 文件夹信息响应
#[derive(Debug, Serialize)]
pub struct FolderInfo {
    pub name: String,
    pub file_count: usize,
}

/// 获取所有的 LanceDB 文件夹及对应文件数量
pub async fn get_folders(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<FolderInfo>>, (StatusCode, Json<ApiResponse>)> {
    let table_names = state.db.table_names().execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("获取表列表失败: {}", e),
            }),
        )
    })?;

    let mut folders = Vec::new();

    // 筛选出前缀为 file_meta_ 的表名
    for table_name in table_names {
        if let Some(folder_name) = table_name.strip_prefix("file_meta_") {
            let table = state.db.open_table(&table_name).execute().await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse {
                        success: false,
                        message: format!("打开表 {} 失败: {}", table_name, e),
                    }),
                )
            })?;
            
            // 查出该表的行数，从而得到文件的数量
            let file_count = table.count_rows(None).await.unwrap_or(0);
            
            folders.push(FolderInfo {
                name: folder_name.to_string(),
                file_count,
            });
        }
    }

    Ok(Json(folders))
}
