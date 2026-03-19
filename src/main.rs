mod handler;
mod extractor;
mod chunker;
mod embedding;
mod upload;

use std::sync::Arc;

use axum::{
    Router,
    routing::{get, post},
};
use lancedb::connect;
use tower_http::cors::CorsLayer;
use tracing::info;

use handler::{AppState, add_item, get_folders, health_check, init_table, search};
use upload::{init_default_folders, upload_file};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        .route("/api/folders", get(get_folders))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    info!("服务器启动在 http://127.0.0.1:3000 (监听所有接口 0.0.0.0:3000)");

    axum::serve(listener, app).await?;

    Ok(())
}
