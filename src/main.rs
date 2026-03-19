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
use upload::upload_file;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化日志
    tracing_subscriber::fmt::init();

    // 连接 LanceDB（使用本地目录存储）
    let db = connect("data/cerebro_gate.lancedb").execute().await?;
    info!("LanceDB 数据库连接成功");

    // 初始化表（如果不存在则创建）
    init_table(&db).await?;

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
    info!("服务器启动在 http://0.0.0.0:3000");

    axum::serve(listener, app).await?;

    Ok(())
}
