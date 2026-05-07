/// Engine subprocess bridge.
/// Spawns the C++ deepsolver_core executable and communicates via stdin/stdout.
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::{timeout, Duration};

use crate::types::{EstimateResponse, GpuInfo, ResolvedMemoryBudget, SolverRequest, SolverResponse};

/// v1.3.0: PID of the currently-running solve subprocess (0 = none).
/// `cancel_solve` reads this and kills the process. Single global atomic
/// is fine because the UI only allows one solve at a time and the
/// `try_run_solver` path serialises through `run_solver`.
static CURRENT_SOLVE_PID: AtomicU32 = AtomicU32::new(0);

/// Cancel the currently-running solve, if any. Returns true if a process
/// was killed. Called by the `cancel_solve` Tauri command bound to the
/// frontend Stop button. Uses `taskkill /F` on Windows; the kill_on_drop
/// path on the Rust side will also reap the child after the process dies.
pub fn cancel_current_solve() -> Result<bool, String> {
    let pid = CURRENT_SOLVE_PID.load(Ordering::SeqCst);
    if pid == 0 { return Ok(false); }
    #[cfg(windows)]
    {
        let out = std::process::Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .output()
            .map_err(|e| format!("taskkill failed: {}", e))?;
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
    Ok(true)
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
///     annotate the response so the UI can show a toast.
///   - Non-GPU failures (timeout, bad input, etc.) bubble up as-is.
pub async fn run_solver(request: &SolverRequest) -> Result<SolverResponse, String> {
    let first_attempt = request.backend.as_deref().unwrap_or("auto");

    match try_run_solver(request, None).await {
        Ok(response) => Ok(response),
        Err(err) => {
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
                let mut cpu_response = try_run_solver(request, Some("cpu")).await?;
                // Annotate backend name so UI knows a fallback happened
                let original = cpu_response.backend.unwrap_or_else(|| "CPU".to_string());
                cpu_response.backend = Some(format!(
                    "{} (auto-fallback from GPU: {})",
                    original, err
                ));
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
    ];

    if let Some(ref history) = request.history {
        args.push("--history".to_string());
        args.push(history.clone());
    }
    if let Some(ref target) = request.target_combo {
        args.push("--target".to_string());
        args.push(target.clone());
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

    args
}

/// One attempt at running the solver subprocess. Used by run_solver for the
/// initial call and for the CPU fallback retry.
async fn try_run_solver(
    request: &SolverRequest,
    backend_override: Option<&str>,
) -> Result<SolverResponse, String> {
    let binary = find_engine_binary();
    let args = build_solver_args(request, backend_override);

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
    // Cleared in every exit path below — RAII would be nicer but the rest
    // of this function reads the child move-by-move so a guard struct
    // would tangle with the borrow checker.
    if let Some(pid) = child.id() {
        CURRENT_SOLVE_PID.store(pid, Ordering::SeqCst);
    }

    let stdout = child.stdout.take()
        .ok_or_else(|| "Failed to capture stdout".to_string())?;
    let stderr = child.stderr.take()
        .ok_or_else(|| "Failed to capture stderr".to_string())?;

    // Collect stderr (contains error JSON on failure, progress lines on success)
    let stderr_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        let mut collected = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            collected.push_str(&line);
            collected.push('\n');
        }
        collected
    });

    // Timeout scales with iteration count
    // Timeout scales with iteration count so bigger solves get more time, but
    // stays bounded so a runaway subprocess can't hang the app indefinitely.
    // Previous formula (iter/5 + 60) gave only 120s for 300 iter — too tight
    // for a CPU fallback on a complex spot with wide ranges. Bump it so the
    // envelope is ~2s per iteration + 2min baseline, capped at 15 min.
    //
    //   100 iter → max(300, 260)       = 300 s
    //   300 iter → max(300, 720)       = 720 s
    //   500 iter → max(300, 1120)      =  900 s (capped)
    //   1000 iter → 900 s (capped)
    //
    // v1.3.1: when the user set a time_budget, Tauri's outer timeout MUST
    // be generous enough for the engine's internal budget to fire BEFORE
    // Tauri kills the subprocess. The engine checks budget BETWEEN
    // iterations, so on slow hardware a single iter can exceed the budget
    // (Pascal CPU laptop on a 9k-node turn solve hits ~150-300s/iter).
    // Allow 3× the budget plus 90s for postsolve — that gives the engine
    // room to finish the in-flight iter and run finalize+postsolve before
    // Tauri's outer killswitch fires. Capped at 30 min to keep runaway
    // bounded.
    let timeout_secs = if let Some(budget) = request.time_budget_seconds {
        if budget > 0 {
            std::cmp::min((budget as u64).saturating_mul(3).saturating_add(90), 1800)
        } else {
            std::cmp::min(std::cmp::max(300u64, (request.iterations as u64) * 2 + 120), 900)
        }
    } else {
        std::cmp::min(std::cmp::max(300u64, (request.iterations as u64) * 2 + 120), 900)
    };

    let stdout_result = match timeout(Duration::from_secs(timeout_secs), async {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        let mut json_output = String::new();
        while let Ok(Some(line)) = lines.next_line().await {
            json_output.push_str(&line);
            json_output.push('\n');
        }
        json_output
    }).await {
        Ok(s) => s,
        Err(_) => {
            // Phase 5: previously we returned without killing the child, so the
            // engine kept burning CPU/GPU/RAM in the background after the user
            // saw a "timeout" toast. Now we explicitly:
            //   1. SIGKILL/TerminateProcess the child
            //   2. await child.wait so the OS reaps it (no zombie / locked exe)
            //   3. drain the stderr collector so its task ends cleanly
            //   4. log pid + backend + iters + board for post-hoc diagnosis
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
            // v1.3.1: clearer message when time_budget was active. The
            // engine should have stopped at the budget but didn't get
            // there before Tauri's killswitch — almost always means a
            // single iter exceeded our budget×3 envelope, i.e. spot is
            // too big for this hardware.
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

    let status = child.wait().await
        .map_err(|e| format!("Engine process error: {}", e))?;
    // v1.3.0: child has been reaped — PID no longer valid for cancel.
    // Clear here so cancel_solve doesn't try to taskkill an exited PID.
    CURRENT_SOLVE_PID.store(0, Ordering::SeqCst);
    let stderr_str = stderr_handle.await.unwrap_or_default();

    if !status.success() {
        // Log the full stderr so we can see what engine said before crashing.
        log_to_file(&format!(
            "CRASH exit_code={:?} stderr_len={} stderr_tail={:?}",
            status.code(),
            stderr_str.len(),
            &stderr_str[stderr_str.len().saturating_sub(500)..]
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
            e, &stdout_result[..stdout_result.len().min(200)]
        ))?;

    Ok(response)
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
/// same backend selection). Skips the streaming-progress / timeout
/// machinery — estimate-only is fast enough to await directly.
pub async fn run_estimate(request: &SolverRequest) -> Result<EstimateResponse, String> {
    let binary = find_engine_binary();

    let mut args = build_solver_args(request, None);
    args.push("--estimate-only".to_string());

    let mut cmd = Command::new(&binary);
    cmd.args(&args);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    #[cfg(windows)]
    cmd.creation_flags(0x08000000);

    // 30 s ceiling — estimate-only should complete in well under 1 s; this
    // is just a safety net for pathological cases (e.g. a broken binary
    // hanging at startup). If it ever gets close to this, something else
    // is wrong.
    let output = match timeout(Duration::from_secs(30), cmd.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => return Err(format!("Failed to spawn engine for estimate: {}", e)),
        Err(_) => return Err("Engine estimate timed out (>30s)".to_string()),
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Engine estimate exited with code {:?}: {}",
            output.status.code(),
            stderr.lines().next().unwrap_or("")
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
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    #[cfg(windows)]
    cmd.creation_flags(0x08000000);

    let output = cmd
        .output()
        .await
        .map_err(|e| format!("Failed to run engine for GPU detection: {} (path: {:?})", e, binary))?;

    if !output.status.success() {
        return Err(format!("Engine exited with code: {:?}", output.status.code()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let info: GpuInfo = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse GPU info: {}. Raw: {}", e, stdout))?;

    Ok(info)
}
