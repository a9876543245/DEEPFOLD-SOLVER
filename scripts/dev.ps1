# Local dev launcher for D:\DEEPFOLD-SOLVER-repo
#
# `npm run tauri dev` invokes `cargo run` which reads DEEPFOLD_GOOGLE_CLIENT_SECRET
# at COMPILE time via option_env!() in src-tauri/src/oauth.rs. Without it,
# Google sign-in returns "client_secret is missing" at runtime, and you can't
# get past the login screen to test SpotLibrary / pre-solved bundles / etc.
#
# This script:
#   1. Loads .env.local into the current PowerShell session
#   2. Verifies DEEPFOLD_GOOGLE_CLIENT_SECRET is now set
#   3. Runs `npm run tauri dev`
#
# Usage:
#   pwsh scripts/dev.ps1            # if you have pwsh
#   powershell -ExecutionPolicy Bypass -File scripts/dev.ps1  # builtin powershell
#
# .env.local is searched in:
#   1. ./.env.local                 (repo-local)
#   2. D:\DEEPFOLD-SOLVER\.env.local (sandbox sibling — original location per memory)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path $PSScriptRoot -Parent
$envCandidates = @(
    Join-Path $repoRoot ".env.local"
    "D:\DEEPFOLD-SOLVER\.env.local"
)

$loaded = $false
foreach ($envFile in $envCandidates) {
    if (-not (Test-Path $envFile)) { continue }
    Write-Host "Loading env from $envFile" -ForegroundColor DarkGray
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) { return }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { return }
        $key = $line.Substring(0, $eq).Trim()
        $val = $line.Substring($eq + 1).Trim()
        if ($val.StartsWith('"') -and $val.EndsWith('"')) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        Set-Item -Path "Env:$key" -Value $val
    }
    $loaded = $true
    break
}

if (-not $loaded) {
    Write-Host "[!] No .env.local found in any of:" -ForegroundColor Yellow
    $envCandidates | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    Write-Host "    Google sign-in will fail at runtime." -ForegroundColor Yellow
}

if ($env:DEEPFOLD_GOOGLE_CLIENT_SECRET) {
    Write-Host "DEEPFOLD_GOOGLE_CLIENT_SECRET set (length: $($env:DEEPFOLD_GOOGLE_CLIENT_SECRET.Length))" -ForegroundColor Green
} else {
    Write-Host "[!] DEEPFOLD_GOOGLE_CLIENT_SECRET still not set. Sign-in will return" -ForegroundColor Yellow
    Write-Host "    'client_secret is missing'. Check .env.local file format:" -ForegroundColor Yellow
    Write-Host "    DEEPFOLD_GOOGLE_CLIENT_SECRET=GOCSPX-xxxxx" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Starting npm run tauri dev (Ctrl+C to stop) ===" -ForegroundColor Cyan
Write-Host ""

# IMPORTANT: cargo's incremental cache does NOT detect option_env! changes
# automatically. If you previously built without the secret set, you must
# rebuild for the new env to take effect. The cleanest way:
#   cargo clean --manifest-path src-tauri/Cargo.toml -p deepsolver-app
# Run that ONCE if you switched from a no-secret build, then re-run dev.

Push-Location $repoRoot
try {
    npm run tauri dev
} finally {
    Pop-Location
}
