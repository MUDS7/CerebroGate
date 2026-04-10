mod chat;
mod chunker;
mod embedding;
mod extractor;
mod handler;
mod sensitive;
mod sensitive_service;
mod upload;

use std::sync::Arc;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use lancedb::connect;
use tower_http::cors::CorsLayer;
use tracing::info;

use chat::{chat, chat_rag, delete_session, get_session_detail, get_sessions};
use handler::{
    AppState, add_item, create_folder, delete_file, delete_folder, get_file_content,
    get_folder_files, get_folders, health_check, init_table, rename_folder, search,
};
use sensitive::{
    add_sensitive_word, delete_sensitive_word, init_sensitive_table, list_sensitive_words,
    update_sensitive_word,
};
use upload::{init_default_folders, upload_file};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载 .env 环境变量
    dotenvy::dotenv().ok();

    // 初始化日志
    tracing_subscriber::fmt::init();

    let db_path = "./.lancedb".to_string();

    // 连接 LanceDB
    let db = connect(&db_path).execute().await?;
    info!("LanceDB 数据库连接成功 ({})", db_path);

    // 初始化表（如果不存在则创建）
    init_table(&db).await?;

    // 初始化默认文件夹（项目文件、其他文件）
    init_default_folders(&db).await?;

    // 初始化敏感词表
    init_sensitive_table(&db).await?;

    let state = Arc::new(AppState { db });

    // 构建路由
    let app = Router::new()
        .route("/", get(health_check))
        .route("/api/items", post(add_item))
        .route("/api/search", post(search))
        .route("/api/search_text", post(handler::search_text))
        .route("/api/upload", post(upload_file))
        .route("/api/folders", get(get_folders).post(create_folder))
        .route("/api/folders/rename", post(rename_folder))
        .route("/api/folders/delete", post(delete_folder))
        .route("/api/folders/files", post(get_folder_files))
        .route("/api/folders/files/content", post(get_file_content))
        .route("/api/folders/files/delete", post(delete_file))
        .route("/api/chat", post(chat))
        .route("/api/chat/rag", post(chat_rag))
        .route("/api/chat/sessions", get(get_sessions))
        .route("/api/chat/session", post(get_session_detail))
        .route("/api/chat/session/delete", post(delete_session))
        .route("/api/sensitive/list", get(list_sensitive_words))
        .route("/api/sensitive/create", post(add_sensitive_word))
        .route("/api/sensitive/update", post(update_sensitive_word))
        .route("/api/sensitive/delete", post(delete_sensitive_word))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("服务器启动在 http://127.0.0.1:3000 (监听所有接口 0.0.0.0:3000)");

    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chunk_text;
    use crate::embedding::embed_texts;
    use crate::extractor;
    use std::path::Path;
    use walkdir::WalkDir;

    #[tokio::test]
    async fn test_batch_vectorize_files() {
        // 加载 .env 环境变量
        dotenvy::dotenv().ok();
        // 初始化日志，避免重复初始化
        let _ = tracing_subscriber::fmt::try_init();

        // 由于底层已经改为本地离线向量化方案 (fastembed + BGE模型)，
        // 这里提供一个空的 api_key 占位符即可，无需再读取环境变量
        let api_key = "local_offline_mode".to_string();

        // 指定要扫描的路径（这里可以修改为您实际想要测试的路径）
        // 比如可以填绝对路径 "D:\\my_documents" 或相对路径 "./data/test_docs"
        let target_dir = "C:/Users/l/AppData/Roaming/E-Mobile/Downloads/重庆智能评审项目资料汇总20260310/项目履约/工程案例/工程案例【可研+初设】/可研/可研工程案例/重庆至万州高速铁路重庆汝溪河牵220千伏外部供电工程";

        println!("========= 开始扫描路径: {} =========", target_dir);
        if !Path::new(target_dir).exists() {
            println!("测试路径 {} 不存在，请指定一个有效的路径", target_dir);
            return;
        }

        let chunk_size = 1500;
        let chunk_overlap = 300;

        for entry in WalkDir::new(target_dir).into_iter().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    match ext_lower.as_str() {
                        "doc" | "docx" | "xls" | "xlsx" | "pdf" | "ofd" => {
                            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                            println!("--------------------------------------------------");
                            println!("找到支持的文件: {}", path.display());

                            // 提取文本
                            let extracted =
                                match extractor::extract_text(path.to_str().unwrap(), &ext_lower) {
                                    Ok(texts) => texts,
                                    Err(e) => {
                                        println!(">>> 文件 {} 文本提取失败: {}", file_name, e);
                                        continue;
                                    }
                                };

                            if extracted.is_empty() {
                                println!(">>> 文件 {} 未提取到文本内容", file_name);
                                continue;
                            }

                            // 切片
                            let chunks = chunk_text(&extracted, chunk_size, chunk_overlap);
                            println!(
                                ">>> 文件 {} 切分为 {} 个文本块(chunk)",
                                file_name,
                                chunks.len()
                            );

                            if chunks.is_empty() {
                                continue;
                            }

                            let texts: Vec<String> =
                                chunks.iter().map(|c| c.text.clone()).collect();

                            // 向量化
                            match embed_texts(&texts, &api_key).await {
                                Ok(embeddings) => {
                                    println!(
                                        ">>> 文件 {} 向量化成功! 获得 {} 个向量",
                                        file_name,
                                        embeddings.len()
                                    );
                                }
                                Err(e) => {
                                    println!(">>> 文件 {} 向量化失败: {}", file_name, e);
                                }
                            }
                        }
                        _ => {} // 忽略其他类型的文件
                    }
                }
            }
        }
        println!("========= 批量向量化测试完成 =========");
    }
}
