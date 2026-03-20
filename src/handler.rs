use std::sync::Arc;

use futures::TryStreamExt;

use arrow_array::{Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use axum::{
    Json,
    extract::{Query, State},
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
        if let Some(enc_folder) = table_name.strip_prefix("file_meta_") {
            let folder_name = crate::upload::decode_folder_name(enc_folder);
            
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
                name: folder_name,
                file_count,
            });
        }
    }

    Ok(Json(folders))
}

/// 创建文件夹请求体
#[derive(Debug, Deserialize)]
pub struct CreateFolderRequest {
    pub name: String,
}

/// 创建一个新的文件夹
pub async fn create_folder(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateFolderRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    if payload.name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "文件夹名称不能为空".to_string(),
            }),
        ));
    }

    crate::upload::create_folder_tables(&state.db, &payload.name).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("创建文件夹失败: {}", e),
            }),
        )
    })?;

    Ok(Json(ApiResponse {
        success: true,
        message: format!("文件夹 '{}' 创建成功", payload.name),
    }))
}

/// 删除文件夹请求体
#[derive(Debug, Deserialize)]
pub struct DeleteFolderRequest {
    pub name: String,
}

/// 重命名文件夹请求体
#[derive(Debug, Deserialize)]
pub struct RenameFolderRequest {
    pub old_name: String,
    pub new_name: String,
}

/// 删除文件夹（包括所有文件数据与表）
pub async fn delete_folder(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<DeleteFolderRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    if payload.name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse { success: false, message: "参数错误: name 不能为空".to_string() }),
        ));
    }

    let folder_name = &payload.name;
    let enc_folder = crate::upload::encode_folder_name(folder_name);
    let chunks_tbl = format!("files_{}", enc_folder);
    let meta_tbl = format!("file_meta_{}", enc_folder);

    // 删除数据库中的表
    let _ = state.db.drop_table(&chunks_tbl).await;
    let _ = state.db.drop_table(&meta_tbl).await;

    info!("表 {}, {} 已被删除 (如存在)", chunks_tbl, meta_tbl);

    // 删除本地物理文件夹
    let upload_dir = format!("data/uploads/{}", folder_name);
    let _ = tokio::fs::remove_dir_all(&upload_dir).await;

    Ok(Json(ApiResponse {
        success: true,
        message: format!("文件夹 '{}' 及包含的所有文件已成功删除", folder_name),
    }))
}

/// 重命名文件夹
pub async fn rename_folder(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RenameFolderRequest>,
) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
    if payload.old_name.trim().is_empty() || payload.new_name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse { success: false, message: "参数错误: 文件夹名不能为空".to_string() }),
        ));
    }

    let old_name = &payload.old_name;
    let new_name = &payload.new_name;

    let table_names = state.db.table_names().execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse { success: false, message: format!("获取表失败: {}", e) }),
        )
    })?;

    let old_enc = crate::upload::encode_folder_name(old_name);
    let new_enc = crate::upload::encode_folder_name(new_name);

    let old_chunks = format!("files_{}", old_enc);
    let old_meta = format!("file_meta_{}", old_enc);
    let new_chunks = format!("files_{}", new_enc);
    let new_meta = format!("file_meta_{}", new_enc);

    // 检查新的名字是否冲突
    if table_names.contains(&new_chunks) || table_names.contains(&new_meta) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse { success: false, message: "目标文件夹名已存在".to_string() }),
        ));
    }

    // 重命名表 (LanceDB OSS 暂不支持 rename_table()，改用物理重命名目录)
    let db_path = "./.lancedb";
    if table_names.contains(&old_chunks) {
        let old_dir = format!("{}/{}.lance", db_path, old_chunks);
        let new_dir = format!("{}/{}.lance", db_path, new_chunks);
        if let Err(e) = tokio::fs::rename(&old_dir, &new_dir).await {
            tracing::error!("重命名表文件失败 {}: {}", old_dir, e);
        }
    }
    if table_names.contains(&old_meta) {
        let old_dir = format!("{}/{}.lance", db_path, old_meta);
        let new_dir = format!("{}/{}.lance", db_path, new_meta);
        if let Err(e) = tokio::fs::rename(&old_dir, &new_dir).await {
            tracing::error!("重命名表文件失败 {}: {}", old_dir, e);
        }
    }

    // 重命名本地文件夹
    let old_dir = format!("data/uploads/{}", old_name);
    let new_dir = format!("data/uploads/{}", new_name);

    if tokio::fs::try_exists(&old_dir).await.unwrap_or(false) {
        let _ = tokio::fs::rename(&old_dir, &new_dir).await;
    } else {
        let _ = tokio::fs::create_dir_all(&new_dir).await;
    }

    info!("文件夹 '{}' 重命名为 '{}'", old_name, new_name);

    Ok(Json(ApiResponse {
        success: true,
        message: format!("文件夹 '{}' 已重命名为 '{}'", old_name, new_name),
    }))
}

/// 查询文件夹文件列表请求参数
#[derive(Debug, Deserialize)]
pub struct GetFolderFilesQuery {
    pub folder: String,
}

/// 文件信息响应
#[derive(Debug, Serialize)]
pub struct FileInfo {
    pub file_id: String,
    pub file_name: String,
    pub file_type: String,
    pub file_path: String,
    pub file_size: i64,
    pub chunk_count: i32,
    pub created_at: String,
}

/// 查询指定文件夹下的所有文件
pub async fn get_folder_files(
    State(state): State<Arc<AppState>>,
    Query(params): Query<GetFolderFilesQuery>,
) -> Result<Json<Vec<FileInfo>>, (StatusCode, Json<ApiResponse>)> {
    // URL 解码文件夹名，防止前端二次编码
    let folder = urlencoding::decode(&params.folder)
        .unwrap_or(std::borrow::Cow::Borrowed(&params.folder))
        .to_string();

    if folder.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "参数错误: folder 不能为空".to_string(),
            }),
        ));
    }

    let enc_folder = crate::upload::encode_folder_name(&folder);
    let meta_table_name = format!("file_meta_{}", enc_folder);

    // 检查表是否存在
    let table_names = state.db.table_names().execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("获取表列表失败: {}", e),
            }),
        )
    })?;

    if !table_names.contains(&meta_table_name) {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ApiResponse {
                success: false,
                message: format!("文件夹 '{}' 不存在", folder),
            }),
        ));
    }

    let table = state.db.open_table(&meta_table_name).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("打开元数据表失败: {}", e),
            }),
        )
    })?;

    let batches: Vec<RecordBatch> = table
        .query()
        .execute()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("查询文件列表失败: {}", e),
                }),
            )
        })?
        .try_collect()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("收集查询结果失败: {}", e),
                }),
            )
        })?;

    let mut files = Vec::new();

    for batch in &batches {
        let file_id_col = batch
            .column_by_name("file_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let file_name_col = batch
            .column_by_name("file_name")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let file_type_col = batch
            .column_by_name("file_type")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let file_path_col = batch
            .column_by_name("file_path")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let file_size_col = batch
            .column_by_name("file_size")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::Int64Array>());
        let chunk_count_col = batch
            .column_by_name("chunk_count")
            .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
        let created_at_col = batch
            .column_by_name("created_at")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        if let (
            Some(file_ids),
            Some(file_names),
            Some(file_types),
            Some(file_paths),
            Some(file_sizes),
            Some(chunk_counts),
            Some(created_ats),
        ) = (
            file_id_col,
            file_name_col,
            file_type_col,
            file_path_col,
            file_size_col,
            chunk_count_col,
            created_at_col,
        ) {
            for i in 0..batch.num_rows() {
                files.push(FileInfo {
                    file_id: file_ids.value(i).to_string(),
                    file_name: file_names.value(i).to_string(),
                    file_type: file_types.value(i).to_string(),
                    file_path: file_paths.value(i).to_string(),
                    file_size: file_sizes.value(i),
                    chunk_count: chunk_counts.value(i),
                    created_at: created_ats.value(i).to_string(),
                });
            }
        }
    }

    Ok(Json(files))
}
