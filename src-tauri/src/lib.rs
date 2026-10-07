mod commands;
mod engine;
mod gto_charts;
mod oauth;
mod types;
mod api_server;

use commands::{
    solve, estimate_solve, cancel_solve, engine_status, save_solution, load_solution, get_gpu_info, start_google_oauth,
    query_node, query_ranges, close_session,
};
use gto_charts::{list_gto_scenarios, load_gto_chart, read_bundled_presolve};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            solve,
            estimate_solve,
            cancel_solve,
            engine_status,
            save_solution,
            load_solution,
            get_gpu_info,
            start_google_oauth,
            list_gto_scenarios,
            load_gto_chart,
            read_bundled_presolve,
            query_node,
            query_ranges,
            close_session
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
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app, event| {
            // 2026-10-06 audit: a running solve or the in-memory session used
            // to outlive the window. (The engine also watches --parent-pid.)
            if let tauri::RunEvent::Exit = event {
                engine::shutdown();
            }
        });
}
