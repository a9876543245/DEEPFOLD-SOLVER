/// GTO preflop chart library — loader for the bundled `gto_output/` dataset.
///
/// The dataset is ~2550 JSON files covering Cash 6max/8max + MTT scenarios,
/// each with per-action mixed strategies in PioSolver UPI range format. The
/// frontend uses these for two purposes:
///   1. Default IP/OOP ranges when the user picks a position matchup
///   2. A standalone chart browser for studying preflop strategy
///
/// Listing is metadata-only (cheap, ~1MB total response). Individual charts
/// are loaded on demand via `load_gto_chart`.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One scenario's metadata. Returned by `list_gto_scenarios` (no range
/// strings — keeps the listing payload small).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtoScenario {
    /// Path-based unique ID, e.g. "cash/6max_100bb/BB_11bb_N19".
    pub id: String,
    /// "Cash" or "MTT".
    pub game_type: String,
    /// Folder name, e.g. "6max_100bb", "vs_open_3b".
    pub scenario_type: String,
    /// "BB", "BTN", "CO", etc.
    pub hero_position: String,
    /// Effective stack in BB (None for charts that don't pin a stack).
    pub effective_bb: Option<i32>,
    pub description: String,
}

/// One chart's full data. Returned by `load_gto_chart`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GtoChart {
    pub id: String,
    pub context: GtoScenario,
    /// Action labels in display order, e.g. ["Raise 2.5", "Call", "Fold"].
    pub actions: Vec<String>,
    /// action label -> PioSolver range string ("AA:0.93,AKs:0.96,...").
    pub ranges: HashMap<String, String>,
}

/// Raw JSON shape that `gto_output/*.json` files use. Internal — we
/// translate to the public types above.
#[derive(Debug, Deserialize)]
struct RawChart {
    context: RawContext,
    strategy: RawStrategy,
}

#[derive(Debug, Deserialize)]
struct RawContext {
    game_type: Option<String>,
    scenario_type: Option<String>,
    hero_position: Option<String>,
    effective_bb: Option<i32>,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Deserialize)]
struct RawStrategy {
    actions: Vec<RawAction>,
    upi_ranges: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawAction {
    name: String,
}

/// Locate the bundled `gto_output/` directory.
///
/// Search order — covers every layout we care about:
///   1. `<exe>/resources/gto_output/`
///      Tauri 2 production with `"resources": {"../gto_output": "gto_output"}`
///      object-form mapping (the clean install layout).
///   2. `<exe>/resources/_up_/gto_output/`
///      Tauri 2 production with array-form `["../gto_output/**/*"]` glob —
///      Tauri prefixes parent-relative paths with `_up_` for safety.
///   3. `<exe>/gto_output/`
///      Sidecar-style co-location, or a manual install drop.
///   4. Walk up from cwd looking for `gto_output/` — development mode.
fn find_gto_root() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let candidates = [
                parent.join("resources").join("gto_output"),
                parent.join("resources").join("_up_").join("gto_output"),
                parent.join("gto_output"),
            ];
            for c in candidates.iter() {
                if c.exists() && c.is_dir() {
                    return Some(c.clone());
                }
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = cwd.as_path();
        for _ in 0..6 {
            let candidate = dir.join("gto_output");
            if candidate.exists() && candidate.is_dir() {
                return Some(candidate);
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
    }
    None
}

/// Read one chart's metadata WITHOUT loading the (potentially large)
/// per-action ranges. Used for the listing pass.
fn read_metadata(path: &Path, root: &Path) -> Option<GtoScenario> {
    let bytes = std::fs::read(path).ok()?;
    let raw: RawChart = serde_json::from_slice(&bytes).ok()?;
    let rel = path.strip_prefix(root).ok()?;
    let mut id = rel.with_extension("").to_string_lossy().replace('\\', "/");
    // ID always uses forward slashes regardless of platform.
    if id.starts_with("./") { id = id[2..].to_string(); }

    Some(GtoScenario {
        id,
        game_type: raw.context.game_type.unwrap_or_else(|| "Unknown".into()),
        scenario_type: raw.context.scenario_type.unwrap_or_else(|| "Unknown".into()),
        hero_position: raw.context.hero_position.unwrap_or_else(|| "?".into()),
        effective_bb: raw.context.effective_bb,
        description: raw.context.description,
    })
}

/// Recursively walk `gto_output/` collecting all .json files.
/// Skips files whose name doesn't match the chart pattern (e.g. progress
/// files, .range artifacts).
fn collect_json_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            // Skip the top-level progress file — it's not a chart.
            if path.file_name().and_then(|n| n.to_str())
                == Some("progress_hybrid.json") {
                continue;
            }
            out.push(path);
        }
    }
}

/// List every scenario in the bundled dataset. Metadata only — call
/// `load_gto_chart(id)` for the full ranges.
#[tauri::command]
pub async fn list_gto_scenarios() -> Result<Vec<GtoScenario>, String> {
    let root = find_gto_root()
        .ok_or_else(|| "gto_output/ directory not found".to_string())?;

    let mut paths = Vec::new();
    collect_json_files(&root, &mut paths);
    paths.sort();

    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        if let Some(s) = read_metadata(&p, &root) {
            out.push(s);
        }
    }
    Ok(out)
}

/// Load one full chart by ID (the path returned in `list_gto_scenarios`).
#[tauri::command]
pub async fn load_gto_chart(id: String) -> Result<GtoChart, String> {
    let root = find_gto_root()
        .ok_or_else(|| "gto_output/ directory not found".to_string())?;

    // Re-attach extension and read.
    let rel = format!("{}.json", id);
    let path = root.join(&rel);
    if !path.exists() {
        return Err(format!("Chart not found: {}", id));
    }

    let bytes = std::fs::read(&path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let raw: RawChart = serde_json::from_slice(&bytes)
        .map_err(|e| format!("Invalid JSON in {}: {}", path.display(), e))?;

    let scenario = GtoScenario {
        id: id.clone(),
        game_type: raw.context.game_type.unwrap_or_else(|| "Unknown".into()),
        scenario_type: raw.context.scenario_type.unwrap_or_else(|| "Unknown".into()),
        hero_position: raw.context.hero_position.unwrap_or_else(|| "?".into()),
        effective_bb: raw.context.effective_bb,
        description: raw.context.description,
    };

    Ok(GtoChart {
        id,
        context: scenario,
        actions: raw.strategy.actions.into_iter().map(|a| a.name).collect(),
        ranges: raw.strategy.upi_ranges,
    })
}

// ============================================================================
// v1.8.3+ pre-solved spot bundle (PRESOLVE_BUNDLE_PLAN Phase 3)
// ============================================================================
//
// The bulk-presolve pipeline (scripts/bulk-presolve.mjs +
// scripts/compact-presolved.mjs) emits one `<spot_key>.json.gz` per bundled
// spot under `gto_output/presolved/`. Filename scheme:
//   m<matchupIdx>_b<boardIdx>_<sizingKey>_<stackLabel>bb.json.gz
// e.g. `m0_b0_standard_defaultbb.json.gz`.
//
// The frontend's SpotLibrary calls `read_bundled_presolve(spot_key)`. We
// locate the file under the same `gto_output/` resource root used by the
// preflop charts above, gunzip it, and return the JSON string. The
// frontend then JSON.parses + version-checks against its CURRENT_SOLVER_VERSION
// constant — mismatches fall through to a live solve.

/// Like `find_gto_root()` but additionally requires that the resolved
/// `gto_output/` contains a `presolved/` subdirectory. Reason: in dev mode,
/// Tauri may have ALREADY copied an older snapshot of `gto_output/` (without
/// `presolved/`) into `target/debug/gto_output/`, while the source-of-truth
/// `gto_output/presolved/` lives one level up. `find_gto_root()` would
/// (correctly) return the exe-relative path because that's right for chart
/// listing; but for presolve lookup we need the dir that ACTUALLY has the
/// presolved bundles. This helper iterates the same candidate list but skips
/// any candidate that lacks `presolved/`.
fn find_presolve_dir() -> Option<PathBuf> {
    let mut all_candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            all_candidates.push(parent.join("resources").join("gto_output"));
            all_candidates.push(parent.join("resources").join("_up_").join("gto_output"));
            all_candidates.push(parent.join("gto_output"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut dir = cwd.as_path();
        for _ in 0..6 {
            all_candidates.push(dir.join("gto_output"));
            match dir.parent() {
                Some(p) => dir = p,
                None => break,
            }
        }
    }
    for c in all_candidates.iter() {
        let presolved = c.join("presolved");
        if presolved.exists() && presolved.is_dir() {
            return Some(c.clone());
        }
    }
    None
}

#[tauri::command]
pub async fn read_bundled_presolve(spot_key: String) -> Result<String, String> {
    use std::io::Read;

    // Defensive — never let a frontend pass `..` or absolute paths.
    if spot_key.contains("..") || spot_key.contains('/') || spot_key.contains('\\') {
        return Err(format!("Invalid spot_key (path traversal): {}", spot_key));
    }
    if !spot_key.ends_with(".json.gz") {
        return Err(format!("spot_key must end with .json.gz, got: {}", spot_key));
    }

    // Use the presolve-aware finder so dev mode doesn't pick a stale Tauri
    // resource snapshot that happens to have `gto_output/` but not
    // `gto_output/presolved/`.
    let root = find_presolve_dir()
        .ok_or_else(|| "gto_output/presolved/ directory not found".to_string())?;
    let path = root.join("presolved").join(&spot_key);

    if !path.exists() {
        // NotFound is the common case — frontend treats it as "no bundle, fall
        // through to live solve". Return a structured marker so the frontend
        // can distinguish from real I/O errors.
        return Err(format!("NOT_FOUND:{}", spot_key));
    }

    let bytes = std::fs::read(&path)
        .map_err(|e| format!("Read failed for {}: {}", path.display(), e))?;
    let mut decoder = flate2::read::GzDecoder::new(&bytes[..]);
    let mut json = String::new();
    decoder
        .read_to_string(&mut json)
        .map_err(|e| format!("Gunzip failed for {}: {}", path.display(), e))?;
    Ok(json)
}
