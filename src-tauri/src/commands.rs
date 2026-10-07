/// Tauri IPC commands exposed to the React frontend.
/// These are the bridge between the GUI and the C++ solver engine.
use tauri;
use tauri::AppHandle;
use crate::engine;
use crate::oauth::{self, OAuthSession};
use crate::types::{EstimateResponse, GpuInfo, SolverRequest, SolverResponse};

/// Solve a poker position. Called from React via `invoke('solve', { request })`.
///
/// AppHandle is taken so the engine can emit `engine-progress` events while
/// the solver runs (real iter / exploit values parsed from C++ stderr). The
/// frontend's useSolver hook listens for these and replaces the previous
/// fake setInterval-based progress animation that capped at 95%/iter 285.
#[tauri::command]
pub async fn solve(app: AppHandle, request: SolverRequest) -> Result<SolverResponse, String> {
    engine::run_solver(&request, Some(app)).await
}

/// v1.2.2: estimate solve time + memory BEFORE running the actual solve.
/// Calls `deepsolver_core --estimate-only` which builds tree + iso then
/// returns a SolveResources block (sub-second on most spots, ~100ms on
/// monotone iso). Frontend uses the result to show an ETA banner so users
/// can decide whether to commit before a 10-minute CPU wait.
#[tauri::command]
pub async fn estimate_solve(request: SolverRequest) -> Result<EstimateResponse, String> {
    engine::run_estimate(&request).await
}

/// v1.3.0: cancel the currently-running solve. Frontend Stop button calls
/// this. Returns true if a process was actually killed, false if there was
/// no active solve. Pure abort — does NOT preserve partial results (use
/// `time_budget_seconds` on SolverRequest for the "stop with what we have"
/// behavior; the budget is precise to within one in-flight iteration).
#[tauri::command]
pub fn cancel_solve() -> Result<bool, String> {
    engine::cancel_current_solve()
}

/// 2026-10-06: one node of the solve kept in memory (`--serve`): kind,
/// pot, stacks, to-call, action labels + amounts, strategies, EVs, ranges.
/// `history` is comma-separated engine action labels, "#<card>" on the
/// action that ends a street to pick the runout; a history ending a street
/// without a card returns the chance node (its `runouts` list the cards).
#[tauri::command]
pub async fn query_node(session_id: u64, history: String) -> Result<serde_json::Value, String> {
    let reply = engine::query_session(
        session_id, serde_json::json!({ "cmd": "node", "history": history })).await?;
    session_reply(reply)
}

/// 2026-10-06: per-combo ranges (+ pot, stack, board, initiative) at a node
/// of the in-memory solve — what a later-street re-solve starts from.
#[tauri::command]
pub async fn query_ranges(session_id: u64, history: String) -> Result<serde_json::Value, String> {
    let reply = engine::query_session(
        session_id, serde_json::json!({ "cmd": "ranges", "history": history })).await?;
    session_reply(reply)
}

/// Drop an in-memory solve (frees its RAM); no id = all of them.
#[tauri::command]
pub async fn close_session(session_id: Option<u64>) {
    engine::close_session(session_id).await;
}

fn session_reply(reply: serde_json::Value) -> Result<serde_json::Value, String> {
    if reply.get("status").and_then(|v| v.as_str()) == Some("error") {
        let msg = reply.get("message").and_then(|v| v.as_str()).unwrap_or("query failed");
        return Err(msg.to_string());
    }
    Ok(reply)
}

/// Get solver engine status / health check.
#[tauri::command]
pub fn engine_status() -> String {
    "ready".to_string()
}

/// Detect available CUDA GPU. Returns description + functional flag.
/// Called from React on app startup to populate the backend indicator.
#[tauri::command]
pub async fn get_gpu_info() -> Result<GpuInfo, String> {
    engine::detect_gpu().await
}

/// Start a Google OAuth flow via the system browser.
/// Returns the local callback port. Frontend must listen for the
/// `oauth-google-token` event to receive the id_token once the user
/// completes sign-in.
#[tauri::command]
pub async fn start_google_oauth(app: AppHandle) -> Result<OAuthSession, String> {
    oauth::start_google_oauth(app).await
}

/// Save a solution to a .dsolver file.
#[tauri::command]
pub async fn save_solution(path: String, result: SolverResponse) -> Result<(), String> {
    let json = serde_json::to_vec(&result)
        .map_err(|e| format!("Serialization error: {}", e))?;

    // Compress with flate2
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());

    // Write magic bytes + version
    encoder.write_all(b"DSLV").map_err(|e| e.to_string())?;
    encoder.write_all(&[0x01, 0x00]).map_err(|e| e.to_string())?; // version 1.0

    // Write JSON data
    encoder.write_all(&json).map_err(|e| e.to_string())?;

    let compressed = encoder.finish().map_err(|e| e.to_string())?;

    std::fs::write(&path, compressed)
        .map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(())
}

/// Load a solution from a .dsolver file.
#[tauri::command]
pub async fn load_solution(path: String) -> Result<SolverResponse, String> {
    let data = std::fs::read(&path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // Decompress with flate2
    use flate2::read::GzDecoder;
    use std::io::Read;

    let mut decoder = GzDecoder::new(&data[..]);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)
        .map_err(|e| format!("Decompression error: {}", e))?;

    // Skip magic bytes (4) + version (2)
    if decompressed.len() < 6 || &decompressed[..4] != b"DSLV" {
        return Err("Invalid .dsolver file format".to_string());
    }

    let json_data = &decompressed[6..];
    let result: SolverResponse = serde_json::from_slice(json_data)
        .map_err(|e| format!("Failed to parse solution data: {}", e))?;

    Ok(result)
}
