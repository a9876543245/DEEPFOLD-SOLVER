mod commands;
mod engine;
mod gto_charts;
mod oauth;
mod types;
mod api_server;

use commands::{
    solve, engine_status, save_solution, load_solution, get_gpu_info, start_google_oauth,
};
use gto_charts::{list_gto_scenarios, load_gto_chart};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            solve,
            engine_status,
            save_solution,
            load_solution,
            get_gpu_info,
            start_google_oauth,
            list_gto_scenarios,
            load_gto_chart
        ])
        .setup(|_app| {
            // Optionally start headless API server
            // This runs on a background tokio task
            if std::env::args().any(|a| a == "--headless") {
                let port = std::env::args()
                    .skip_while(|a| a != "--api-port")
                    .nth(1)
                    .and_then(|p| p.parse::<u16>().ok())
                    .unwrap_or(8080);

                tauri::async_runtime::spawn(async move {
                    if let Err(e) = api_server::start_api_server(port).await {
                        eprintln!("[DeepSolver] API server error: {}", e);
                    }
                });
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
