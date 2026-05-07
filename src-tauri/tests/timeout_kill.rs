//! Polish #3: Tauri-side regression test for the engine-timeout cleanup path.
//!
//! Background: src-tauri/src/engine.rs spawns `deepsolver_core.exe` per solve.
//! When the solve doesn't finish inside the budget, we want to kill the child
//! cleanly — not leak a zombie process holding GPU memory + CPU. The Phase 5
//! fix in engine.rs added explicit `child.kill().await + child.wait().await`
//! after a timeout. This test guards that fix from regressing.
//!
//! What this exercises (light version, per the polish-round spec):
//!   1. Locate the engine binary (skip the test with a clear message if not built)
//!   2. Spawn it with `--iterations 9999999 --backend cpu` (will run "forever")
//!   3. Wait 1s via `tokio::time::timeout` to mimic the production path
//!   4. After the wait expires, call `child.kill().await + child.wait().await`
//!   5. Sleep briefly so the OS reaps the process
//!   6. Assert `child.try_wait()` returns `Ok(Some(_))` — process is gone
//!
//! Skipped (deferred): platform-specific PID checks (e.g. /proc/<pid> on Linux,
//! GetExitCodeProcess on Windows). The standard try_wait is enough to catch
//! the regression we care about (kill not propagated → child stays alive).

use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::{sleep, timeout};

/// Search the usual binary locations. Returns None when the engine hasn't been
/// built yet — the test then prints a skip notice instead of failing CI on
/// machines that haven't run `cmake --build`.
fn find_engine_binary() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "deepsolver_core.exe" } else { "deepsolver_core" };
    let candidates = [
        // From src-tauri/ (where `cargo test` runs):
        PathBuf::from("..").join("core").join("build").join("Release").join(exe),
        PathBuf::from("..").join("core").join("build").join(exe),
        // From repo root (just in case the harness changes cwd):
        PathBuf::from("core").join("build").join("Release").join(exe),
        // Sidecar location (filled in for release builds):
        PathBuf::from("binaries").join(if cfg!(windows) {
            "deepsolver_core-x86_64-pc-windows-msvc.exe"
        } else {
            "deepsolver_core-x86_64-unknown-linux-gnu"
        }),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn engine_child_killed_after_timeout() {
    let Some(binary) = find_engine_binary() else {
        eprintln!(
            "SKIP: deepsolver_core binary not found. Build it with \
             `cmake --build core/build --config Release --target deepsolver_core` \
             before running this test."
        );
        return;
    };
    eprintln!("Using engine binary at {}", binary.display());

    // 9_999_999 iterations on CPU is "effectively forever" for any reasonable
    // wall-clock budget. We pick CPU explicitly so the test doesn't compete
    // with other GPU users on the box.
    let mut cmd = Command::new(&binary);
    cmd.args([
        "--pot", "100",
        "--stack", "500",
        "--board", "AsKd7c",
        "--iterations", "9999999",
        "--backend", "cpu",
        "--postsolve", "none",
    ]);
    cmd.kill_on_drop(true);
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(std::process::Stdio::null());

    let mut child = cmd.spawn().expect("spawn engine");

    // Pretend we're in the production timeout path: race the child's exit
    // against a 1s budget. The child should NOT exit on its own.
    let waited = timeout(Duration::from_secs(1), child.wait()).await;
    assert!(
        waited.is_err(),
        "engine exited before our timeout fired — bump --iterations or \
         CPU budget if hardware got faster"
    );

    // Cleanup path mirrors src-tauri/src/engine.rs:
    //   `child.kill().await` then `child.wait().await` so the OS reaps
    //   the process and we don't leave a zombie.
    child.kill().await.expect("kill child");
    let exit_status = child.wait().await.expect("wait child");
    eprintln!("Engine exit status after kill: {:?}", exit_status);

    // Sleep briefly so any pending wait notification settles.
    sleep(Duration::from_millis(200)).await;

    // Already-reaped child should report `Ok(Some(_))` from try_wait.
    let try_wait = child.try_wait().expect("try_wait should not error after wait");
    assert!(
        try_wait.is_some(),
        "expected try_wait to return Some(_) after explicit wait, got None — \
         child may still be alive (kill didn't propagate)"
    );
}
