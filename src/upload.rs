use std::path::Path;
use std::sync::Arc;

use arrow_array::{Float32Array, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use axum::{
    Json,
    extract::{Multipart, State},
    http::StatusCode,
};
use serde::Serialize;
use tracing::info;
use uuid::Uuid;

use crate::chunker::{TextChunk, chunk_text};
use crate::embedding::{EMBEDDING_DIM, embed_texts};
use crate::extractor::{self, ExtractedText};
use crate::handler::{ApiResponse, AppState};

/// 上传响应
#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub success: bool,
    pub message: String,
    pub file_id: String,
    pub file_name: String,
    pub file_type: String,
    pub chunk_count: i32,
    pub file_path: String,
}

/// ===== Schema 定义 =====

/// 文档切片表 Schema
fn make_chunks_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("file_id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM,
            ),
            false,
        ),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("page_number", DataType::Int32, false),
        Field::new("chunk_index", DataType::Int32, false),
    ]))
}

/// 文件元数据表 Schema
fn make_meta_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("file_id", DataType::Utf8, false),
        Field::new("file_name", DataType::Utf8, false),
        Field::new("file_type", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("file_size", DataType::Int64, false),
        Field::new("chunk_count", DataType::Int32, false),
        Field::new("created_at", DataType::Utf8, false),
    ]))
}

/// ===== 工具函数 =====

/// 确保 LanceDB 表存在，不存在则创建
async fn ensure_table(
    db: &lancedb::Connection,
    table_name: &str,
    schema: Arc<Schema>,
) -> Result<(), (StatusCode, Json<ApiResponse>)> {
    let table_names = db.table_names().execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("获取表列表失败: {}", e),
            }),
        )
    })?;

    if !table_names.contains(&table_name.to_string()) {
        let batches = RecordBatchIterator::new(vec![], schema);
        db.create_table(table_name, Box::new(batches))
            .execute()
            .await
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse {
                        success: false,
                        message: format!("创建表 '{}' 失败: {}", table_name, e),
                    }),
                )
            })?;
        info!("表 '{}' 创建成功", table_name);
    }

    Ok(())
}

/// 初始化默认文件夹
pub async fn init_default_folders(db: &lancedb::Connection) -> Result<(), Box<dyn std::error::Error>> {
    let default_folders = ["项目文件", "其他文件"];
    let table_names = db.table_names().execute().await?;

    for folder in default_folders {
        let enc_folder = encode_folder_name(folder);
        let chunks_table_name = format!("files_{}", enc_folder);
        let meta_table_name = format!("file_meta_{}", enc_folder);

        if !table_names.contains(&chunks_table_name) {
            let batches = RecordBatchIterator::new(vec![], make_chunks_schema());
            db.create_table(&chunks_table_name, Box::new(batches)).execute().await?;
            info!("基础表 '{}' 创建成功", chunks_table_name);
        }

        if !table_names.contains(&meta_table_name) {
            let batches = RecordBatchIterator::new(vec![], make_meta_schema());
            db.create_table(&meta_table_name, Box::new(batches)).execute().await?;
            info!("基础表 '{}' 创建成功", meta_table_name);
        }
    }

    Ok(())
}

/// 创建文件夹（对应的表及本地目录）
pub async fn create_folder_tables(db: &lancedb::Connection, folder: &str) -> Result<(), Box<dyn std::error::Error>> {
    let table_names = db.table_names().execute().await?;
    let enc_folder = encode_folder_name(folder);
    let chunks_table_name = format!("files_{}", enc_folder);
    let meta_table_name = format!("file_meta_{}", enc_folder);

    if !table_names.contains(&chunks_table_name) {
        let batches = RecordBatchIterator::new(vec![], make_chunks_schema());
        db.create_table(&chunks_table_name, Box::new(batches)).execute().await?;
        info!("基础表 '{}' 创建成功", chunks_table_name);
    }

    if !table_names.contains(&meta_table_name) {
        let batches = RecordBatchIterator::new(vec![], make_meta_schema());
        db.create_table(&meta_table_name, Box::new(batches)).execute().await?;
        info!("基础表 '{}' 创建成功", meta_table_name);
    }

    let upload_dir = format!("data/uploads/{}", folder);
    tokio::fs::create_dir_all(&upload_dir).await?;

    Ok(())
}

/// 文件夹名称编码（转为十六进制以符合 LanceDB 表名只允许字母数字下划线的限制）
pub fn encode_folder_name(folder: &str) -> String {
    folder.bytes().map(|b| format!("{:02x}", b)).collect::<String>()
}

/// 文件夹名称解码
pub fn decode_folder_name(encoded: &str) -> String {
    let bytes: Vec<u8> = (0..encoded.len())
        .step_by(2)
        .filter_map(|i| {
            if i + 2 <= encoded.len() {
                u8::from_str_radix(&encoded[i..i + 2], 16).ok()
            } else {
                None
            }
        })
        .collect();
    String::from_utf8(bytes).unwrap_or_else(|_| encoded.to_string())
}

/// 获取文件扩展名（小写）
fn get_extension(file_name: &str) -> String {
    Path::new(file_name)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase()
}

/// 判断是否为图片文件（不需要文本提取）
fn is_image(extension: &str) -> bool {
    matches!(extension, "png" | "jpg" | "jpeg")
}

/// ===== 主 Handler =====

/// 文件上传 handler
///
/// 接收 multipart/form-data 请求：
/// - `file`: 上传的文件（必填）
/// - `folder`: 目标文件夹名（必填）
/// - `chunk_size`: 切片大小，默认 500（可选）
/// - `chunk_overlap`: 切片重叠，默认 50（可选）
pub async fn upload_file(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, (StatusCode, Json<ApiResponse>)> {
    let mut file_data: Option<Vec<u8>> = None;
    let mut file_name: Option<String> = None;
    let mut folder: Option<String> = None;
    let mut chunk_size: usize = 500;
    let mut chunk_overlap: usize = 50;

    // 解析 multipart 字段
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

        match name.as_str() {
            "file" => {
                file_name = field.file_name().map(|s| s.to_string());
                let data = field.bytes().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse {
                            success: false,
                            message: format!("读取文件数据失败: {}", e),
                        }),
                    )
                })?;
                file_data = Some(data.to_vec());
            }
            "folder" => {
                let text = field.text().await.map_err(|e| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse {
                            success: false,
                            message: format!("读取 folder 字段失败: {}", e),
                        }),
                    )
                })?;
                folder = Some(text);
            }
            "chunk_size" => {
                if let Ok(text) = field.text().await {
                    if let Ok(size) = text.parse::<usize>() {
                        chunk_size = size;
                    }
                }
            }
            "chunk_overlap" => {
                if let Ok(text) = field.text().await {
                    if let Ok(overlap) = text.parse::<usize>() {
                        chunk_overlap = overlap;
                    }
                }
            }
            _ => {
                // 忽略未知字段
            }
        }
    }

    // 校验必填字段
    let file_data = file_data.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "缺少 file 字段".to_string(),
            }),
        )
    })?;

    let file_name = file_name.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "文件名为空".to_string(),
            }),
        )
    })?;

    let folder = folder.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "缺少 folder 字段".to_string(),
            }),
        )
    })?;

    if folder.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiResponse {
                success: false,
                message: "folder 不能为空".to_string(),
            }),
        ));
    }

    let extension = get_extension(&file_name);
    let file_id = Uuid::new_v4().to_string();
    let file_size = file_data.len() as i64;

    // === 1. 保存文件到本地 ===
    let upload_dir = format!("data/uploads/{}", folder);
    tokio::fs::create_dir_all(&upload_dir).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("创建上传目录失败: {}", e),
            }),
        )
    })?;

    let file_path = format!("{}/{}", upload_dir, file_name);
    tokio::fs::write(&file_path, &file_data)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse {
                    success: false,
                    message: format!("保存文件失败: {}", e),
                }),
            )
        })?;

    info!("文件已保存: {}", file_path);

    // === 2. 构建表名 ===
    let enc_folder = encode_folder_name(&folder);
    let chunks_table_name = format!("files_{}", enc_folder);
    let meta_table_name = format!("file_meta_{}", enc_folder);

    // 确保表存在
    ensure_table(&state.db, &chunks_table_name, make_chunks_schema()).await?;
    ensure_table(&state.db, &meta_table_name, make_meta_schema()).await?;

    // === 3. 判断文件类型并处理 ===
    let chunk_count: i32;

    if is_image(&extension) {
        // 图片文件：不做文本提取，只记录元数据
        chunk_count = 0;
        info!("图片文件 '{}' 仅保存，跳过文本处理", file_name);
    } else {
        // 文档文件：提取 → 切片 → 向量化 → 存储
        let extracted: Vec<ExtractedText> = extractor::extract_text(&file_path, &extension)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse {
                        success: false,
                        message: format!("文本提取失败: {}", e),
                    }),
                )
            })?;

        if extracted.is_empty() {
            // 提取不到文本，跳过后续处理
            chunk_count = 0;
            info!("文件 '{}' 未提取到文本内容", file_name);
        } else {
            // 切片
            let chunks: Vec<TextChunk> = chunk_text(&extracted, chunk_size, chunk_overlap);
            chunk_count = chunks.len() as i32;

            if chunk_count > 0 {
                // 向量化
                let api_key = std::env::var("DEEPSEEK_API_KEY").map_err(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ApiResponse {
                            success: false,
                            message: "未设置 DEEPSEEK_API_KEY 环境变量".to_string(),
                        }),
                    )
                })?;

                let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
                let embeddings = embed_texts(&texts, &api_key).await.map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ApiResponse {
                            success: false,
                            message: format!("向量化失败: {}", e),
                        }),
                    )
                })?;

                // 写入切片表
                store_chunks(
                    &state.db,
                    &chunks_table_name,
                    &file_id,
                    &file_path,
                    &chunks,
                    &embeddings,
                )
                .await?;

                info!(
                    "文件 '{}' 处理完成：{} 个切片已存入表 '{}'",
                    file_name, chunk_count, chunks_table_name
                );
            }
        }
    }

    // === 4. 写入元数据表 ===
    let created_at = chrono::Utc::now().to_rfc3339();
    store_metadata(
        &state.db,
        &meta_table_name,
        &file_id,
        &file_name,
        &extension,
        &file_path,
        file_size,
        chunk_count,
        &created_at,
    )
    .await?;

    Ok(Json(UploadResponse {
        success: true,
        message: format!("文件 '{}' 上传并处理成功", file_name),
        file_id,
        file_name,
        file_type: extension,
        chunk_count,
        file_path,
    }))
}

/// 将切片数据写入 LanceDB
async fn store_chunks(
    db: &lancedb::Connection,
    table_name: &str,
    file_id: &str,
    file_path: &str,
    chunks: &[TextChunk],
    embeddings: &[Vec<f32>],
) -> Result<(), (StatusCode, Json<ApiResponse>)> {
    let schema = make_chunks_schema();

    let table = db.open_table(table_name).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("打开切片表失败: {}", e),
            }),
        )
    })?;

    // 构建各列数据
    let ids: Vec<String> = chunks.iter().map(|_| Uuid::new_v4().to_string()).collect();
    let id_array = StringArray::from(ids.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let file_id_array = StringArray::from(vec![file_id; chunks.len()]);
    let text_array = StringArray::from(chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>());
    let file_path_array = StringArray::from(vec![file_path; chunks.len()]);
    let page_number_array =
        Int32Array::from(chunks.iter().map(|c| c.page_number).collect::<Vec<_>>());
    let chunk_index_array =
        Int32Array::from(chunks.iter().map(|c| c.chunk_index).collect::<Vec<_>>());

    // 构建向量列 (FixedSizeList)
    let all_values: Vec<f32> = embeddings.iter().flatten().copied().collect();
    let values_array = Float32Array::from(all_values);
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let vector_array = arrow_array::FixedSizeListArray::try_new(
        field,
        EMBEDDING_DIM,
        Arc::new(values_array),
        None,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("构建向量数据失败: {}", e),
            }),
        )
    })?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_array),
            Arc::new(file_id_array),
            Arc::new(text_array),
            Arc::new(vector_array),
            Arc::new(file_path_array),
            Arc::new(page_number_array),
            Arc::new(chunk_index_array),
        ],
    )
    .map_err(|e| {
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
                message: format!("写入切片数据失败: {}", e),
            }),
        )
    })?;

    Ok(())
}

/// 将文件元数据写入 LanceDB
#[allow(clippy::too_many_arguments)]
async fn store_metadata(
    db: &lancedb::Connection,
    table_name: &str,
    file_id: &str,
    file_name: &str,
    file_type: &str,
    file_path: &str,
    file_size: i64,
    chunk_count: i32,
    created_at: &str,
) -> Result<(), (StatusCode, Json<ApiResponse>)> {
    let schema = make_meta_schema();

    let table = db.open_table(table_name).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("打开元数据表失败: {}", e),
            }),
        )
    })?;

    let file_id_array = StringArray::from(vec![file_id]);
    let file_name_array = StringArray::from(vec![file_name]);
    let file_type_array = StringArray::from(vec![file_type]);
    let file_path_array = StringArray::from(vec![file_path]);
    let file_size_array = arrow_array::Int64Array::from(vec![file_size]);
    let chunk_count_array = Int32Array::from(vec![chunk_count]);
    let created_at_array = StringArray::from(vec![created_at]);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(file_id_array),
            Arc::new(file_name_array),
            Arc::new(file_type_array),
            Arc::new(file_path_array),
            Arc::new(file_size_array),
            Arc::new(chunk_count_array),
            Arc::new(created_at_array),
        ],
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("构建元数据 RecordBatch 失败: {}", e),
            }),
        )
    })?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
    table.add(Box::new(batches)).execute().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse {
                success: false,
                message: format!("写入元数据失败: {}", e),
            }),
        )
    })?;

    Ok(())
}
