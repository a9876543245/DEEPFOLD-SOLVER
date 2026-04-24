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

/// Locate the bundled `gto_output/` directory. Tries (in order):
///   1. `gto_output/` next to the executable (production sidecar)
///   2. Project root (development: walk up from cwd looking for it)
fn find_gto_root() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.join("gto_output");
            if p.exists() && p.is_dir() {
                return Some(p);
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
