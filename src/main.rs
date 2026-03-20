mod handler;
mod extractor;
mod chunker;
mod embedding;
mod upload;
mod chat;

use std::sync::Arc;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    routing::{get, post},
};
use lancedb::connect;
use tower_http::cors::CorsLayer;
use tracing::info;

use handler::{
    AppState, add_item, create_folder, delete_folder, get_file_content, get_folder_files, get_folders,
    health_check, init_table, rename_folder, search, delete_file,
};
use upload::{init_default_folders, upload_file};
use chat::{chat, chat_rag, get_sessions, get_session_detail, delete_session};

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

    let state = Arc::new(AppState { db });

    // 构建路由
    let app = Router::new()
        .route("/", get(health_check))
        .route("/api/items", post(add_item))
        .route("/api/search", post(search))
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
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("服务器启动在 http://127.0.0.1:3000 (监听所有接口 0.0.0.0:3000)");

    axum::serve(listener, app).await?;

    Ok(())
}

