/// Engine subprocess bridge.
/// Spawns the C++ deepsolver_core executable and communicates via stdin/stdout.
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::time::{timeout, Duration};

use crate::types::{EstimateResponse, GpuInfo, ResolvedMemoryBudget, SolverRequest, SolverResponse};

/// v1.4.1: payload for the `engine-progress` event the frontend listens
/// for to drive the real progress bar. Emitted once per `[Iter N]` stderr
/// line written by the C++ engine. Replaces the old setInterval-based
/// fake animation that capped at 95% / iter 285.
#[derive(serde::Serialize, Clone)]
struct EngineProgress {
    /// 1-based iteration number that just completed (matches `[Iter N]`).
    iteration: u32,
    /// Exploitability % at this iteration. May be 0 if engine reported 0.00
    /// (early iterations / postsolve disabled at the running cadence).
    exploitability_pct: f32,
    /// Cumulative ms inside the iteration phase, as reported by the engine.
    elapsed_ms: f32,
}

/// Payload for the `engine-progress-decompose` event, emitted once per
/// `[decompose] … leaf I/L` stderr line during an Exact (runout
/// decomposition) run, plus a `phase: "finalize"` tick when the
/// `[decompose] outer=…` completion summary appears. Gives the UI real
/// per-subgame progress where the bar previously parked at 99% (the
/// monolithic pre-solve's last `[Iter]` line) for the whole decompose phase.
#[derive(serde::Serialize, Clone)]
struct DecomposeProgress {
    /// "sweep" | "final" | "finalize".
    phase: &'static str,
    /// 1-based sweep number. 0 outside the sweep phase.
    sweep: u32,
    /// Total outer sweeps. Carried into final/finalize payloads from the
    /// last seen sweep line so every event is self-contained.
    sweep_total: u32,
    /// Completed leaves (subgames) in the current pass. 0 for finalize.
    leaf: u32,
    /// Total leaves in the current pass. 0 for finalize.
    leaf_total: u32,
}

/// v1.3.0: PID of the currently-running solve subprocess (0 = none).
/// `cancel_solve` reads this and kills the process. Single global atomic
/// is fine because the UI only allows one solve at a time and the
/// `try_run_solver` path serialises through `run_solver`.
static CURRENT_SOLVE_PID: AtomicU32 = AtomicU32::new(0);

/// 2026-10-06 audit: set by the Stop button. The kill used to look like an
/// engine crash, so `run_solver` restarted the whole solve on the CPU.
static CANCELLED: AtomicBool = AtomicBool::new(false);

/// The error a cancelled solve returns; the UI treats it as neutral.
pub const CANCELLED_MESSAGE: &str = "Solve cancelled";

/// Line the engine writes after the result and after every --serve reply.
const SERVE_END: &str = "@@DEEPSOLVER_END@@";

/// 2026-10-06: an engine process that keeps a solve in memory (`--serve`)
/// and answers node / range queries, so navigating the tree never re-solves.
/// Several can be alive - a later-street re-solve keeps the earlier street's
/// solve queryable - but at most MAX_SESSIONS (least recently used dropped).
struct EngineSession {
    id: u64,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

const MAX_SESSIONS: usize = 3;
static NEXT_SESSION_ID: AtomicU32 = AtomicU32::new(1);
const SESSION_GONE: &str = "The solved spot is no longer in memory - solve it again.";

/// Sessions, least recently used first.
fn sessions() -> &'static tokio::sync::Mutex<Vec<EngineSession>> {
    static SLOT: OnceLock<tokio::sync::Mutex<Vec<EngineSession>>> = OnceLock::new();
    SLOT.get_or_init(|| tokio::sync::Mutex::new(Vec::new()))
}

async fn kill_session(mut s: EngineSession) {
    let _ = s.child.kill().await;
    let _ = s.child.wait().await;
}

/// End one in-memory session (frees its RAM); `None` ends all of them.
pub async fn close_session(id: Option<u64>) {
    let mut list = sessions().lock().await;
    let (gone, keep): (Vec<_>, Vec<_>) =
        list.drain(..).partition(|s| id.map_or(true, |i| i == s.id));
    *list = keep;
    for s in gone {
        kill_session(s).await;
    }
}

async fn add_session(session: EngineSession) {
    let mut list = sessions().lock().await;
    while list.len() >= MAX_SESSIONS {
        let oldest = list.remove(0);
        kill_session(oldest).await;
    }
    list.push(session);
}

/// Send one request to in-memory session `id` and return its JSON reply.
pub async fn query_session(id: u64, request: serde_json::Value) -> Result<serde_json::Value, String> {
    let mut list = sessions().lock().await;
    let pos = list.iter().position(|s| s.id == id).ok_or_else(|| SESSION_GONE.to_string())?;
    // Most recently used goes last.
    let session = list.remove(pos);
    list.push(session);
    let session = list.last_mut().unwrap();
    let line = format!("{}\n", request);
    let reply = timeout(Duration::from_secs(600), async {
        session.stdin.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
        session.stdin.flush().await.map_err(|e| e.to_string())?;
        read_until_end(&mut session.stdout).await
    })
    .await;
    match reply {
        Ok(Ok(Some(text))) => serde_json::from_str(&text)
            .map_err(|e| format!("Bad engine reply: {}", e)),
        _ => {
            // The process died or hung: drop it.
            if let Some(s) = list.pop() {
                kill_session(s).await;
            }
            Err(SESSION_GONE.to_string())
        }
    }
}

/// Read stdout up to the next SERVE_END line. None = EOF first (the engine
/// exited). Lossy UTF-8, so a stray byte can never stop the reader.
async fn read_until_end(reader: &mut BufReader<ChildStdout>) -> Result<Option<String>, String> {
    let mut out = String::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf).await.map_err(|e| e.to_string())?;
        if n == 0 {
            return Ok(None);
        }
        let line = String::from_utf8_lossy(&buf);
        let line = line.trim_end_matches(['\r', '\n']);
        if line == SERVE_END {
            return Ok(Some(out));
        }
        out.push_str(line);
        out.push('\n');
    }
}

/// Cancel the currently-running solve, if any. Returns true if a process
/// was killed. Called by the `cancel_solve` Tauri command bound to the
/// frontend Stop button. Uses `taskkill /F` on Windows; the kill_on_drop
/// path on the Rust side will also reap the child after the process dies.
pub fn cancel_current_solve() -> Result<bool, String> {
    CANCELLED.store(true, Ordering::SeqCst);
    let pid = CURRENT_SOLVE_PID.load(Ordering::SeqCst);
    if pid == 0 { return Ok(false); }
    kill_pid(pid)?;
    Ok(true)
}

fn kill_pid(pid: u32) -> Result<(), String> {
    #[cfg(windows)]
    {
        let mut cmd = std::process::Command::new("taskkill");
        cmd.args(["/F", "/PID", &pid.to_string()]);
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        let out = cmd.output().map_err(|e| format!("taskkill failed: {}", e))?;
        if !out.status.success() {
            // Already dead is fine — race between user click and natural exit.
            let stderr = String::from_utf8_lossy(&out.stderr);
            if !stderr.contains("not found") && !stderr.contains("找不到") {
                return Err(format!("taskkill {}: {}", pid, stderr));
            }
        }
    }
    #[cfg(not(windows))]
    {
        // SIGKILL via `kill -9` for unix portability (when we get there).
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .output();
    }
    Ok(())
}

/// App exit: stop the running solve and the in-memory session. (The engine
/// also watches --parent-pid, so a crash cannot leave it running either.)
pub fn shutdown() {
    let pid = CURRENT_SOLVE_PID.swap(0, Ordering::SeqCst);
    if pid != 0 {
        let _ = kill_pid(pid);
    }
    if let Ok(mut list) = sessions().try_lock() {
        for s in list.iter_mut() {
            let _ = s.child.start_kill();
        }
        list.clear();
    }
}

/// Diagnostic log file location. Writes CLI args and engine stderr on each
/// run so we can diagnose user-reported crashes without reproducing state.
fn engine_log_path() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join("co.deepfold.solver").join("engine.log")
}

fn log_to_file(msg: &str) {
    use std::io::Write;
    let path = engine_log_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(f, "[{}] {}", ts, msg);
    }
}

/// Find the path to the deepsolver_core binary.
/// In development, looks for it in the core/build/Release directory.
/// In production, looks for the sidecar binary.
fn find_engine_binary() -> PathBuf {
    // Try sidecar location first (production: binary next to the app executable)
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_default();

    let sidecar = exe_dir.join("deepsolver_core.exe");
    if sidecar.exists() {
        return sidecar;
    }

    // Try development locations relative to the executable directory
    let dev_candidates = [
        exe_dir.join("../core/build/Release/deepsolver_core.exe"),
        exe_dir.join("../../core/build/Release/deepsolver_core.exe"),
        exe_dir.join("../../../core/build/Release/deepsolver_core.exe"),
    ];
    for path in &dev_candidates {
        if path.exists() {
            return path.clone();
        }
    }

    // Try development locations relative to CWD
    let cwd_candidates = [
        PathBuf::from("core/build/Release/deepsolver_core.exe"),
        PathBuf::from("core/build_cpu/Release/deepsolver_core.exe"),
        PathBuf::from("../core/build/Release/deepsolver_core.exe"),
    ];
    for path in &cwd_candidates {
        if path.exists() {
            return path.clone();
        }
    }

    // Try to find workspace root by walking up from CWD until we find Cargo.toml
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = cwd.as_path();
        for _ in 0..5 {
            let candidate = dir.join("core/build/Release/deepsolver_core.exe");
            if candidate.exists() {
                return candidate;
            }
            let candidate_cpu = dir.join("core/build_cpu/Release/deepsolver_core.exe");
            if candidate_cpu.exists() {
                return candidate_cpu;
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
    }

    // Last resort: hope it's on PATH
    PathBuf::from("deepsolver_core.exe")
}

/// Run the solver engine with the given request, with automatic GPU→CPU fallback.
///
/// Behavior:
///   - If `request.backend` is "cpu" (or unset in a CPU-only context), run CPU directly.
///   - Otherwise try the requested backend (auto/gpu). On GPU-related failure
///     (CUDA/device/GPU errors), automatically retry with backend=cpu and
///     note it in `resources.fallback_reason` so the UI can show a toast.
///   - A cancelled solve returns CANCELLED_MESSAGE and is never retried.
///   - Non-GPU failures (timeout, bad input, etc.) bubble up as-is.
pub async fn run_solver(
    request: &SolverRequest,
    app: Option<AppHandle>,
) -> Result<SolverResponse, String> {
    CANCELLED.store(false, Ordering::SeqCst);
    let first_attempt = request.backend.as_deref().unwrap_or("auto");

    match try_run_solver(request, None, app.as_ref()).await {
        Ok(response) => Ok(response),
        Err(err) => {
            if CANCELLED.load(Ordering::SeqCst) {
                return Err(CANCELLED_MESSAGE.to_string());
            }
            let gpu_attempted = first_attempt != "cpu";

            // Known GPU-specific error messages (from structured engine JSON).
            let looks_like_gpu_err = err.contains("CUDA")
                || err.contains("cuda")
                || err.contains("GPU")
                || err.contains("Gpu")
                || err.contains("device")
                || err.contains("CUBLAS")
                || err.contains("out of memory");

            // Engine process died without emitting a structured error — usually a
            // native crash (access violation, stack overflow, heap corruption).
            // On Windows these appear as exit code `0xC0000xxx` in decimal form.
            // We can't prove it was the GPU path, but in AUTO/GPU mode the GPU
            // code is the riskiest (newer, larger surface) — falling back to
            // CPU is the right default. CPU is deterministic and well-tested.
            let is_process_crash = err.contains("Engine exited with code");

            // Timeouts are NOT a GPU-specific issue — CPU would likely also
            // time out if iterations are too high. Don't waste the user's
            // time retrying; surface the timeout message as-is.
            let is_timeout = err.contains("timed out");

            let should_fallback =
                gpu_attempted && !is_timeout && (looks_like_gpu_err || is_process_crash);

            if should_fallback {
                eprintln!(
                    "[DeepSolver] GPU run failed ({}); falling back to CPU",
                    err
                );
                let mut cpu_response = try_run_solver(request, Some("cpu"), app.as_ref())
                    .await
                    .map_err(|e| {
                        if CANCELLED.load(Ordering::SeqCst) { CANCELLED_MESSAGE.to_string() } else { e }
                    })?;
                // 2026-10-06 audit: keep `backend` the engine's own name (the
                // UI keys its badge on it) and say why in fallback_reason.
                cpu_response.resources.fallback_reason =
                    format!("GPU run failed ({}); solved on the CPU instead.", err);
                Ok(cpu_response)
            } else {
                Err(err)
            }
        }
    }
}

/// v1.2.2: build the CLI argv vector for `deepsolver_core` from a
/// SolverRequest. Extracted from `try_run_solver` so the estimate-only
/// path (`run_estimate`) uses byte-identical args — that way the ETA
/// reflects exactly what the real solve would do (same memory profile,
/// same iterations, same backend, same bet sizing).
fn build_solver_args(request: &SolverRequest, backend_override: Option<&str>) -> Vec<String> {
    let mut args = vec![
        "--pot".to_string(), request.pot_size.to_string(),
        "--stack".to_string(), request.effective_stack.to_string(),
        "--board".to_string(), request.board.clone(),
        "--iterations".to_string(), request.iterations.to_string(),
        "--exploitability".to_string(), request.exploitability.to_string(),
        // 2026-10-06: the engine ends itself when this app process is gone.
        "--parent-pid".to_string(), std::process::id().to_string(),
    ];

    if let Some(ref history) = request.history {
        args.push("--history".to_string());
        args.push(history.clone());
    }
    if let Some(ref target) = request.target_combo {
        args.push("--target".to_string());
        args.push(target.clone());
        if let Some(ref player) = request.target_player {
            args.push("--target-player".to_string());
            args.push(player.clone());
        }
    }
    if let Some(ref ip) = request.ip_range {
        args.push("--ip-range".to_string());
        args.push(ip.clone());
    }
    if let Some(ref oop) = request.oop_range {
        args.push("--oop-range".to_string());
        args.push(oop.clone());
    }
    if let Some(ref locks) = request.node_locks {
        args.push("--node-locks".to_string());
        args.push(locks.clone());
    }

    let effective_backend = backend_override
        .map(String::from)
        .or_else(|| request.backend.clone());
    if let Some(b) = effective_backend {
        args.push("--backend".to_string());
        args.push(b);
    }

    if let Some(oi) = request.oop_has_initiative {
        args.push("--oop-initiative".to_string());
        args.push(if oi { "1".to_string() } else { "0".to_string() });
    }
    if let Some(dk) = request.allow_donk_bet {
        args.push("--allow-donk-bet".to_string());
        args.push(if dk { "1".to_string() } else { "0".to_string() });
    }

    fn join_floats(v: &[f64]) -> String {
        v.iter().map(|f| format!("{}", f)).collect::<Vec<_>>().join(",")
    }
    if let Some(ref v) = request.flop_sizes {
        if !v.is_empty() {
            args.push("--flop-sizes".to_string());
            args.push(join_floats(v));
        }
    }
    if let Some(ref v) = request.turn_sizes {
        if !v.is_empty() {
            args.push("--turn-sizes".to_string());
            args.push(join_floats(v));
        }
    }
    if let Some(ref v) = request.river_sizes {
        if !v.is_empty() {
            args.push("--river-sizes".to_string());
            args.push(join_floats(v));
        }
    }
    // 2026-10-06: Pio-style per-player menus (JSON, see the engine's
    // --bet-sizing help). Overrides the per-street lists above.
    if let Some(ref spec) = request.bet_sizing {
        if !spec.is_empty() {
            args.push("--bet-sizing".to_string());
            args.push(spec.clone());
        }
    }
    if let Some(cap) = request.raise_cap {
        args.push("--raise-cap".to_string());
        args.push(cap.to_string());
    }
    if let Some(th) = request.allin_threshold {
        args.push("--allin-threshold".to_string());
        args.push(format!("{}", th));
    }
    if let Some(ref iso) = request.iso {
        if !iso.is_empty() {
            args.push("--iso".to_string());
            args.push(iso.clone());
        }
    }

    let mb = ResolvedMemoryBudget::resolve(request);
    if mb.host_mb > 0 {
        args.push("--host-memory-mb".to_string());
        args.push(mb.host_mb.to_string());
    }
    if mb.gpu_mb > 0 {
        args.push("--gpu-memory-mb".to_string());
        args.push(mb.gpu_mb.to_string());
    }
    if mb.json_mb > 0 {
        args.push("--json-memory-mb".to_string());
        args.push(mb.json_mb.to_string());
    }
    if mb.strategy_tree_max_nodes > 0 {
        args.push("--strategy-tree-max-nodes".to_string());
        args.push(mb.strategy_tree_max_nodes.to_string());
    }

    // v1.3.0: time budget — stops iteration phase at min(time, iter, exploit).
    // 0/None = no cap. UI mode presets fill this in based on Quick/Std/Deep.
    if let Some(budget) = request.time_budget_seconds {
        if budget > 0 {
            args.push("--time-budget-seconds".to_string());
            args.push(budget.to_string());
        }
    }

    // Stage 5: runout decomposition. Omit for "off"/None so the sidecar keeps
    // its default (off). The sidecar picks the backend (GPU-adaptive for
    // collapsed/rainbow boards, CPU for forced-on-enumerable) — no logic here.
    if let Some(mode) = &request.decompose_runouts {
        if mode == "auto" || mode == "on" {
            args.push("--decompose-runouts".to_string());
            args.push(mode.clone());
        }
    }
    // Roadmap ④: decomposition iteration presets. Pass-through only — the
    // preset → numbers mapping lives in poker.ts (DECOMPOSE_PRESETS), the
    // resolution order (defaults → CLI → env) in the engine.
    if let Some(v) = request.decompose_outer {
        if v > 0 {
            args.push("--decompose-outer".to_string());
            args.push(v.to_string());
        }
    }
    if let Some(v) = request.decompose_inner {
        if v > 0 {
            args.push("--decompose-inner".to_string());
            args.push(v.to_string());
        }
    }
    if let Some(v) = request.decompose_trunk_iters {
        if v > 0 {
            args.push("--decompose-trunk-iters".to_string());
            args.push(v.to_string());
        }
    }
    if let Some(v) = request.decompose_warm_start {
        args.push("--decompose-warmstart".to_string());
        args.push(if v { "1".to_string() } else { "0".to_string() });
    }

    // v1.4.0 Phase 2: CPU SIMD policy + thread count. Backend ignores these
    // when --backend gpu wins, but the engine still parses them so the UI
    // can pre-flight `--estimate-only` with the right CPU rate.
    if let Some(simd) = &request.cpu_simd {
        if !simd.is_empty() {
            args.push("--cpu-simd".to_string());
            args.push(simd.clone());
        }
    }
    if let Some(threads) = request.cpu_threads {
        args.push("--cpu-threads".to_string());
        args.push(threads.to_string());
    }
    if let Some(kind) = &request.cpu_backend {
        if !kind.is_empty() {
            args.push("--cpu-backend".to_string());
            args.push(kind.clone());
        }
    }

    args
}

/// One attempt at running the solver subprocess. Used by run_solver for the
/// initial call and for the CPU fallback retry.
async fn try_run_solver(
    request: &SolverRequest,
    backend_override: Option<&str>,
    app: Option<&AppHandle>,
) -> Result<SolverResponse, String> {
    let binary = find_engine_binary();
    let serve = request.serve.unwrap_or(false);
    let mut args = build_solver_args(request, backend_override);
    if serve {
        args.push("--serve".to_string());
    }

    // Log the command for post-hoc debugging of crashes.
    let quoted_args: Vec<String> = args.iter().map(|a| {
        if a.contains(' ') || a.contains('"') {
            format!("\"{}\"", a.replace('"', "\\\""))
        } else {
            a.clone()
        }
    }).collect();
    log_to_file(&format!(
        "SPAWN backend={} binary={:?} args={}",
        backend_override.unwrap_or("(none)"),
        binary,
        quoted_args.join(" ")
    ));

    let mut cmd = Command::new(&binary);
    cmd.args(&args)
        .stdin(if serve { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Phase 5 (10-point maturity plan): if the Tauri command future is dropped
    // (frontend reload, app shutdown, parent task cancelled), `kill_on_drop`
    // makes Tokio reap the child instead of leaving it as a 100% CPU/GPU
    // orphan. Without this a runaway solve survives the UI closing.
    cmd.kill_on_drop(true);

    // Hide the console window that Windows would otherwise pop up for every
    // subprocess call. CREATE_NO_WINDOW = 0x08000000.
    #[cfg(windows)]
    cmd.creation_flags(0x08000000);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn engine: {} (path: {:?})", e, binary))?;

    // v1.3.0: register PID so the cancel_solve Tauri command can kill us.
    // Cleared in every exit path below.
    if let Some(pid) = child.id() {
        CURRENT_SOLVE_PID.store(pid, Ordering::SeqCst);
    }
    // A Stop click that landed before the PID was registered.
    if CANCELLED.load(Ordering::SeqCst) {
        let _ = child.kill().await;
        let _ = child.wait().await;
        CURRENT_SOLVE_PID.store(0, Ordering::SeqCst);
        return Err(CANCELLED_MESSAGE.to_string());
    }

    let stdin = child.stdin.take();
    let stdout = child.stdout.take()
        .ok_or_else(|| "Failed to capture stdout".to_string())?;
    let stderr = child.stderr.take()
        .ok_or_else(|| "Failed to capture stderr".to_string())?;

    // Collect stderr (contains error JSON on failure, progress lines on success).
    //
    // v1.4.1: as we drain stderr we also parse `[Iter N] Exploitability: X%
    // (Yms)` progress lines and emit `engine-progress` events to the frontend.
    // The full string is still collected for error extraction in the failure
    // path. 2026-10-06: lossy UTF-8 — `lines()` stopped at the first invalid
    // byte, after which nobody drained the pipe and the engine could block.
    let app_for_progress = app.cloned();
    let stderr_handle = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut collected = String::new();
        let mut buf = Vec::new();
        // Outer-sweep count remembered across lines so "final pass" /
        // "outer=" events (which don't repeat it) still carry it.
        let mut last_sweep_total: u32 = 0;
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let text = String::from_utf8_lossy(&buf);
            let line = text.trim_end_matches(['\r', '\n']);
            // Parse: "[Iter N] Exploitability: X% (Yms)"
            if let Some(progress) = parse_iter_line(line) {
                if let Some(handle) = app_for_progress.as_ref() {
                    let _ = handle.emit("engine-progress", progress);
                }
            } else if let Some(dp) = parse_decompose_line(line, &mut last_sweep_total) {
                // Parse: "[decompose] sweep S/T leaf I/L" etc. (Exact mode).
                if let Some(handle) = app_for_progress.as_ref() {
                    let _ = handle.emit("engine-progress-decompose", dp);
                }
            }
            // Bounded: a long solve's progress lines are not needed for errors.
            if collected.len() < 1 << 20 {
                collected.push_str(line);
                collected.push('\n');
            }
        }
        collected
    });

    // Timeout scales with iteration count so bigger solves get more time, but
    // stays bounded so a runaway subprocess can't hang the app indefinitely.
    //
    //   100 iter → max(300, 260)       = 300 s
    //   300 iter → max(300, 720)       = 720 s
    //   500 iter → max(300, 1120)      =  900 s (capped)
    //
    // v1.3.1: when the user set a time_budget, Tauri's outer timeout MUST
    // be generous enough for the engine's internal budget to fire BEFORE
    // Tauri kills the subprocess. Allow 3× the budget plus 90s for postsolve,
    // capped at 30 min. Roadmap ④: Exact mode (runout decomposition) ignores
    // time_budget BY DESIGN — keep runaway protection at an Exact-scale
    // ceiling; the Stop button + kill_on_drop still cover interactive aborts.
    let decompose_on = matches!(
        request.decompose_runouts.as_deref(), Some("auto") | Some("on"));
    let timeout_secs = if decompose_on {
        6 * 3600
    } else if let Some(budget) = request.time_budget_seconds {
        if budget > 0 {
            std::cmp::min((budget as u64).saturating_mul(3).saturating_add(90), 1800)
        } else {
            std::cmp::min(std::cmp::max(300u64, (request.iterations as u64) * 2 + 120), 900)
        }
    } else {
        std::cmp::min(std::cmp::max(300u64, (request.iterations as u64) * 2 + 120), 900)
    };

    let mut stdout_reader = BufReader::new(stdout);
    let read_result = timeout(Duration::from_secs(timeout_secs), async {
        if serve {
            // The result ends at SERVE_END; None = the engine exited first.
            read_until_end(&mut stdout_reader).await
        } else {
            let mut bytes = Vec::new();
            stdout_reader.read_to_end(&mut bytes).await.map_err(|e| e.to_string())?;
            Ok(Some(String::from_utf8_lossy(&bytes).into_owned()))
        }
    }).await;

    let stdout_result = match read_result {
        Ok(Ok(Some(s))) => s,
        Ok(Ok(None)) | Ok(Err(_)) => String::new(),   // engine exited: see status below
        Err(_) => {
            // Phase 5: kill the child, reap it, drain stderr, log, report.
            let pid = child.id();
            log_to_file(&format!(
                "TIMEOUT killing engine pid={:?} backend={} iterations={} board={:?} timeout_secs={}",
                pid,
                backend_override.unwrap_or("(none)"),
                request.iterations,
                request.board,
                timeout_secs
            ));
            let _ = child.kill().await;
            let _ = child.wait().await;
            let _ = stderr_handle.await;
            CURRENT_SOLVE_PID.store(0, Ordering::SeqCst);
            let msg = if let Some(b) = request.time_budget_seconds {
                if b > 0 {
                    format!(
                        "Engine timed out after {}s. Your spot is too large for the {}-second budget on this hardware — \
                         a single iteration exceeded the {}s wall-clock allowance. \
                         Reduce iterations / bet sizes / range width, switch to a smaller spot, or use a faster machine.",
                        timeout_secs, b, timeout_secs
                    )
                } else {
                    format!(
                        "Engine timed out after {} seconds ({} iterations requested). Try reducing iterations.",
                        timeout_secs, request.iterations
                    )
                }
            } else {
                format!(
                    "Engine timed out after {} seconds ({} iterations requested). Try reducing iterations.",
                    timeout_secs, request.iterations
                )
            };
            return Err(msg);
        }
    };

    // A --serve result arrived and the engine stays up: keep it as the session.
    if serve && !stdout_result.is_empty() {
        CURRENT_SOLVE_PID.store(0, Ordering::SeqCst);
        let mut response: SolverResponse = serde_json::from_str(&stdout_result)
            .map_err(|e| format!(
                "Failed to parse engine output: {}. Raw: {}",
                e, stdout_result.chars().take(200).collect::<String>()
            ))?;
        match (response.session, stdin) {
            (true, Some(stdin)) => {
                let id = NEXT_SESSION_ID.fetch_add(1, Ordering::SeqCst) as u64;
                response.session_id = Some(id);
                add_session(EngineSession { id, child, stdin, stdout: stdout_reader }).await;
            }
            _ => {
                response.session = false;
                let _ = child.wait().await;
            }
        }
        log_to_file(&format!("OK backend={} session={}", backend_override.unwrap_or("(none)"), response.session));
        return Ok(response);
    }

    let status = child.wait().await
        .map_err(|e| format!("Engine process error: {}", e))?;
    // v1.3.0: child has been reaped — PID no longer valid for cancel.
    CURRENT_SOLVE_PID.store(0, Ordering::SeqCst);
    let stderr_str = stderr_handle.await.unwrap_or_default();

    if CANCELLED.load(Ordering::SeqCst) {
        return Err(CANCELLED_MESSAGE.to_string());
    }

    if !status.success() || stdout_result.trim().is_empty() {
        // Log the full stderr so we can see what engine said before crashing.
        let tail: String = {
            let chars: Vec<char> = stderr_str.chars().collect();
            chars[chars.len().saturating_sub(500)..].iter().collect()
        };
        log_to_file(&format!(
            "CRASH exit_code={:?} stderr_len={} stderr_tail={:?}",
            status.code(),
            stderr_str.len(),
            tail
        ));

        // Extract the error message from stderr JSON if present
        let err_msg = extract_engine_error(&stderr_str)
            .unwrap_or_else(|| format!("Engine exited with code: {:?}", status.code()));
        return Err(err_msg);
    }

    log_to_file(&format!(
        "OK backend={} stderr_len={}",
        backend_override.unwrap_or("(none)"),
        stderr_str.len()
    ));

    let response: SolverResponse = serde_json::from_str(&stdout_result)
        .map_err(|e| format!(
            "Failed to parse engine output: {}. Raw: {}",
            e, stdout_result.chars().take(200).collect::<String>()
        ))?;

    Ok(response)
}

/// v1.4.1: Parse an engine progress line like
/// `[Iter 50] Exploitability: 0.00% (1771.25ms)` into an `EngineProgress`.
/// Returns None for any line that doesn't match (regular engine logs, blank
/// lines, JSON error payloads, etc.). Line format is set by solver.h's
/// progress callback — keep them in sync if either side changes.
fn parse_iter_line(line: &str) -> Option<EngineProgress> {
    // Quick reject before doing real work.
    if !line.starts_with("[Iter ") { return None; }

    // "[Iter " consumed. Pull integer up to ']'.
    let rest = &line[6..];
    let end_iter = rest.find(']')?;
    let iter: u32 = rest[..end_iter].trim().parse().ok()?;

    // " Exploitability: " expected next.
    let after_bracket = &rest[end_iter + 1..];
    let exp_marker = "Exploitability:";
    let exp_start = after_bracket.find(exp_marker)? + exp_marker.len();
    let after_exp = after_bracket[exp_start..].trim_start();
    let pct_end = after_exp.find('%')?;
    let exploitability_pct: f32 = after_exp[..pct_end].trim().parse().ok()?;

    // "(Yms)" trailing.
    let after_pct = &after_exp[pct_end + 1..];
    let paren_open = after_pct.find('(')? + 1;
    let paren_after = &after_pct[paren_open..];
    let ms_end = paren_after.find("ms")?;
    let elapsed_ms: f32 = paren_after[..ms_end].trim().parse().ok()?;

    Some(EngineProgress { iteration: iter, exploitability_pct, elapsed_ms })
}

/// Parse an Exact-mode (runout decomposition) progress line:
///   `[decompose] sweep 1/2 leaf 37/245`   → phase "sweep"
///   `[decompose] final pass leaf 37/245`  → phase "final"
///   `[decompose] outer=2 inner=450 …`     → phase "finalize" (run summary;
///                                            ev/BR + nav stitch + JSON left)
/// The per-pass SUMMARY lines ("… solved N subgames in Xs") say "solved"
/// where these say "leaf", so they fall through to None along with all other
/// engine chatter. Line formats are set by
/// solver_decomposed.h::solve_all_subgames and main.cpp — keep in sync.
fn parse_decompose_line(line: &str, last_sweep_total: &mut u32) -> Option<DecomposeProgress> {
    let rest = line.strip_prefix("[decompose] ")?;
    if let Some(r) = rest.strip_prefix("sweep ") {
        let (sweep, r) = split_u32(r)?;
        let (sweep_total, r) = split_u32(r.strip_prefix('/')?)?;
        let (leaf, r) = split_u32(r.strip_prefix(" leaf ")?)?;
        let (leaf_total, _) = split_u32(r.strip_prefix('/')?)?;
        *last_sweep_total = sweep_total;
        return Some(DecomposeProgress { phase: "sweep", sweep, sweep_total, leaf, leaf_total });
    }
    if let Some(r) = rest.strip_prefix("final pass leaf ") {
        let (leaf, r) = split_u32(r)?;
        let (leaf_total, _) = split_u32(r.strip_prefix('/')?)?;
        return Some(DecomposeProgress {
            phase: "final", sweep: 0, sweep_total: *last_sweep_total, leaf, leaf_total,
        });
    }
    if rest.starts_with("outer=") {
        return Some(DecomposeProgress {
            phase: "finalize", sweep: 0, sweep_total: *last_sweep_total, leaf: 0, leaf_total: 0,
        });
    }
    None
}

/// Split a leading unsigned integer off `s`, returning (value, remainder).
fn split_u32(s: &str) -> Option<(u32, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if end == 0 { return None; }
    Some((s[..end].parse().ok()?, &s[end..]))
}

/// Parse the C++ engine's stderr for a JSON error payload and extract `.message`.
/// Returns None if no error JSON found.
fn extract_engine_error(stderr: &str) -> Option<String> {
    // Engine emits: {"status": "error", "message": "..."} on stderr on failure.
    // Find the first `{"status":` line and parse it.
    for line in stderr.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('{') && trimmed.contains("\"status\"") {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if let Some(msg) = val.get("message").and_then(|v| v.as_str()) {
                    return Some(msg.to_string());
                }
            }
        }
    }
    None
}

/// v1.2.2: Run the engine with `--estimate-only` to get a pre-solve cost
/// preview (memory + ETA). Short-lived subprocess (~50-300ms typically;
/// monotone iso boards ~1s). Frontend calls this before kicking off the
/// real solve so the UI can show "Estimated 12 minutes on CPU" before the
/// user commits.
///
/// Reuses the same arg-builder as run_solver so the estimate matches what
/// the real solve would actually do (same memory profile, same iterations,
/// same backend selection).
pub async fn run_estimate(request: &SolverRequest) -> Result<EstimateResponse, String> {
    let binary = find_engine_binary();

    let mut args = build_solver_args(request, None);
    args.push("--estimate-only".to_string());

    let mut cmd = Command::new(&binary);
    cmd.args(&args);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    // 2026-10-06 audit: on timeout the output future is dropped — kill the
    // process with it instead of leaving it running.
    cmd.kill_on_drop(true);

    #[cfg(windows)]
    cmd.creation_flags(0x08000000);

    // 30 s ceiling — estimate-only should complete in well under 1 s; this
    // is just a safety net for pathological cases.
    let output = match timeout(Duration::from_secs(30), cmd.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Err(format!("Failed to spawn engine for estimate: {}", e)),
        Err(_) => return Err("Engine estimate timed out (>30s)".to_string()),
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let msg = extract_engine_error(&stderr)
            .unwrap_or_else(|| stderr.lines().next().unwrap_or("").to_string());
        return Err(format!(
            "Engine estimate exited with code {:?}: {}",
            output.status.code(),
            msg
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse estimate JSON: {}. Raw: {}", e, stdout))
}

/// Run the engine with `--gpu-info` to detect CUDA GPU availability.
/// Short-lived subprocess (~100ms). Safe to call at startup and on demand.
pub async fn detect_gpu() -> Result<GpuInfo, String> {
    let binary = find_engine_binary();

    let mut cmd = Command::new(&binary);
    cmd.arg("--gpu-info")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    cmd.kill_on_drop(true);

    #[cfg(windows)]
    cmd.creation_flags(0x08000000);

    // 2026-10-06 audit: a hung driver query no longer hangs the app.
    let output = match timeout(Duration::from_secs(30), cmd.output()).await {
        Ok(r) => r.map_err(|e| format!(
            "Failed to run engine for GPU detection: {} (path: {:?})", e, binary))?,
        Err(_) => return Err("GPU detection timed out (>30s)".to_string()),
    };

    if !output.status.success() {
        return Err(format!("Engine exited with code: {:?}", output.status.code()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let info: GpuInfo = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse GPU info: {}. Raw: {}", e, stdout))?;

    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iter_line_parses() {
        let p = parse_iter_line("[Iter 50] Exploitability: 1.25% (1771.25ms)").unwrap();
        assert_eq!(p.iteration, 50);
        assert!((p.exploitability_pct - 1.25).abs() < 1e-6);
        assert!((p.elapsed_ms - 1771.25).abs() < 1e-3);
        assert!(parse_iter_line("[decompose] sweep 1/2 leaf 3/245").is_none());
    }

    #[test]
    fn decompose_sweep_line_parses() {
        let mut total = 0u32;
        let p = parse_decompose_line("[decompose] sweep 1/2 leaf 37/245", &mut total).unwrap();
        assert_eq!(p.phase, "sweep");
        assert_eq!((p.sweep, p.sweep_total, p.leaf, p.leaf_total), (1, 2, 37, 245));
        assert_eq!(total, 2);
    }

    #[test]
    fn decompose_final_line_carries_sweep_total() {
        let mut total = 0u32;
        parse_decompose_line("[decompose] sweep 2/2 leaf 245/245", &mut total).unwrap();
        let p = parse_decompose_line("[decompose] final pass leaf 5/245", &mut total).unwrap();
        assert_eq!(p.phase, "final");
        assert_eq!((p.sweep, p.sweep_total, p.leaf, p.leaf_total), (0, 2, 5, 245));
    }

    #[test]
    fn decompose_finalize_from_outer_summary() {
        let mut total = 3u32;
        let p = parse_decompose_line(
            "[decompose] outer=2 inner=450 trunk_iters=600 leaves=245 pinned=64 \
             warm_leaves=245 warm_start=on exploit=0.41%",
            &mut total,
        ).unwrap();
        assert_eq!(p.phase, "finalize");
        assert_eq!(p.sweep_total, 3);
    }

    #[test]
    fn decompose_summary_and_chatter_ignored() {
        let mut total = 0u32;
        for line in [
            "[decompose] sweep 1/2 solved 245 subgames in 437.5s",
            "[decompose] final pass solved 245 subgames in 757.8s",
            "[decompose] monolithic solve failed (x); attempting decomposed fallback",
            "[decompose] solve_decomposed threw (x); keeping the monolithic result.",
            "[Iter 50] Exploitability: 1.25% (1771.25ms)",
            "{\"status\": \"error\", \"message\": \"boom\"}",
        ] {
            assert!(parse_decompose_line(line, &mut total).is_none(), "matched: {}", line);
        }
        assert_eq!(total, 0);
    }
}
