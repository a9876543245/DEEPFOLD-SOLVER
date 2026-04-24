/// Tauri IPC commands exposed to the React frontend.
/// These are the bridge between the GUI and the C++ solver engine.
use tauri;
use tauri::AppHandle;
use crate::engine;
use crate::oauth::{self, OAuthSession};
use crate::types::{GpuInfo, SolverRequest, SolverResponse};

/// Solve a poker position. Called from React via `invoke('solve', { request })`.
#[tauri::command]
pub async fn solve(request: SolverRequest) -> Result<SolverResponse, String> {
    engine::run_solver(&request).await
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
