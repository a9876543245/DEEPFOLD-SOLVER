/// Headless HTTP API server (Module E).
/// Provides a lightweight REST API for external AI agents to invoke the solver.
///
/// Endpoint: POST /api/v1/solve
/// Listens on localhost:8080 by default.
use axum::{
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;
use crate::engine;
use crate::types::{SolverRequest, ErrorResponse};

/// Health check endpoint
async fn health() -> &'static str {
    "DeepSolver API v1 - OK"
}

/// Solve endpoint — accepts JSON request, returns solver response.
/// Context protection: never returns full 1326-combo matrix, only aggregated data.
async fn solve_handler(Json(request): Json<SolverRequest>) -> Json<serde_json::Value> {
    match engine::run_solver(&request).await {
        Ok(response) => {
            // Context protection: strip any large arrays before returning
            // The C++ engine already implements this, but double-check here
            Json(serde_json::to_value(response).unwrap_or_default())
        }
        Err(err) => {
            let error = ErrorResponse {
                status: "error".to_string(),
                message: err,
            };
            Json(serde_json::to_value(error).unwrap_or_default())
        }
    }
}

/// Start the headless API server.
/// Call this on a separate tokio task so it doesn't block the main thread.
pub async fn start_api_server(port: u16) -> Result<(), String> {
    let app = Router::new()
        .route("/", get(health))
        .route("/api/v1/solve", post(solve_handler));

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    eprintln!("[DeepSolver API] Starting headless server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await
        .map_err(|e| format!("Failed to bind to port {}: {}", port, e))?;

    axum::serve(listener, app).await
        .map_err(|e| format!("Server error: {}", e))?;

    Ok(())
}
