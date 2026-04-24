# =============================================================================
# DEEPFOLD-SOLVER — Release preparation script
# =============================================================================
#
# Produces signed update artifacts for `tauri-plugin-updater` + emits the
# `latest.json` manifest clients poll for.
#
# Usage:
#   pwsh scripts/prepare-release.ps1                     # uses version from package.json
#   pwsh scripts/prepare-release.ps1 -Version 1.0.1      # override version
#   pwsh scripts/prepare-release.ps1 -SkipBuild          # manifest only
#
# Prerequisites:
#   - Tauri signing private key at $env:TAURI_SIGNING_PRIVATE_KEY_PATH
#     (default: $HOME\.tauri\deepfold_updater.key)
#   - Key password via $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD (empty if none)
#   - Public key in src-tauri\tauri.conf.json (already set — don't change
#     after the first release or existing installs will reject updates)
#
# Output:
#   release\
#     latest.json                                       (upload to endpoint)
#     DEEPFOLD - SOLVER_<ver>_x64-setup.exe             (NSIS installer)
#     DEEPFOLD - SOLVER_<ver>_x64-setup.exe.sig         (signature)
#     DEEPFOLD - SOLVER_<ver>_x64_en-US.msi             (MSI installer)
#
# The updater endpoint (configured in tauri.conf.json) should serve
# latest.json at the URL pattern:
#   https://<host>/solver/{{target}}/{{current_version}}
# Server should return:
#   204 No Content  — when installed version >= latest
#   200 + latest.json — when update is available
# =============================================================================

param(
    [string]$Version = "",
    [switch]$SkipBuild,
    [string]$Notes = "",
    [string]$BaseUrl = "https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/download",
    [string]$TagPrefix = "v"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# ----- 1. Resolve version ---------------------------------------------------
if (-not $Version) {
    $pkg = Get-Content "package.json" | ConvertFrom-Json
    $Version = $pkg.version
}
Write-Host ""
Write-Host "=== DEEPFOLD-SOLVER release v$Version ===" -ForegroundColor Cyan

# Verify version matches across all manifests
$cargoToml = Get-Content "src-tauri\Cargo.toml" -Raw
$tauriConf = Get-Content "src-tauri\tauri.conf.json" | ConvertFrom-Json
$pkgJson   = Get-Content "package.json" | ConvertFrom-Json

if ($tauriConf.version -ne $Version) {
    Write-Host "  [!] tauri.conf.json version is $($tauriConf.version), expected $Version" -ForegroundColor Yellow
    Write-Host "      Bump tauri.conf.json + Cargo.toml + package.json together before release." -ForegroundColor Yellow
    throw "Version mismatch"
}

# ----- 2. Verify signing key ------------------------------------------------
$keyPath = $env:TAURI_SIGNING_PRIVATE_KEY_PATH
if (-not $keyPath) {
    $keyPath = Join-Path $HOME ".tauri\deepfold_updater.key"
    $env:TAURI_SIGNING_PRIVATE_KEY_PATH = $keyPath
}
if (-not (Test-Path $keyPath)) {
    throw "Signing key not found at $keyPath. Regenerate with:`n  npx tauri signer generate -w $keyPath"
}
# Tauri v2 reads TAURI_SIGNING_PRIVATE_KEY (a path OR the base64 secret).
# Pass the path — passing the file content includes the rsign comment
# header line which fails base64 decode. Empty password is set explicitly
# so the signer doesn't block on a stdin prompt during a non-interactive
# build (this is the pattern that hung the build before).
$env:TAURI_SIGNING_PRIVATE_KEY = $keyPath
if (-not $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD) {
    $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = ""
}
Write-Host "Signing key: $keyPath" -ForegroundColor DarkGray

# ----- 3. Build (unless skipped) --------------------------------------------
if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "Building (cargo tauri build)..." -ForegroundColor Cyan
    npm run tauri build
    if ($LASTEXITCODE -ne 0) { throw "Tauri build failed" }
} else {
    Write-Host "Skipping build (-SkipBuild)" -ForegroundColor DarkGray
}

# ----- 4. Collect artifacts -------------------------------------------------
$bundleDir = "src-tauri\target\release\bundle"
$nsisExe  = Get-ChildItem "$bundleDir\nsis\*$Version*setup.exe" -File | Select-Object -First 1
$nsisSig  = Get-ChildItem "$bundleDir\nsis\*$Version*setup.exe.sig" -File | Select-Object -First 1
$msi      = Get-ChildItem "$bundleDir\msi\*$Version*.msi" -File | Select-Object -First 1

if (-not $nsisExe) { throw "NSIS installer not found in $bundleDir\nsis\" }
if (-not $nsisSig) { throw "NSIS signature (.sig) not found. Is `createUpdaterArtifacts: true` in tauri.conf.json?" }

Write-Host ""
Write-Host "Artifacts:" -ForegroundColor Cyan
Write-Host "  NSIS exe : $($nsisExe.FullName) ($([math]::Round($nsisExe.Length/1MB,2)) MB)"
Write-Host "  NSIS sig : $($nsisSig.FullName)"
if ($msi) { Write-Host "  MSI      : $($msi.FullName) ($([math]::Round($msi.Length/1MB,2)) MB)" }

# ----- 5. Build latest.json -------------------------------------------------
$signature = Get-Content $nsisSig.FullName -Raw

if (-not $Notes) {
    # Try to grab a note from git log since the last tag
    try {
        $lastTag = git describe --tags --abbrev=0 2>$null
        if ($lastTag) {
            $Notes = git log "$lastTag..HEAD" --pretty=format:"- %s" 2>$null | Out-String
        }
    } catch {}
    if (-not $Notes) { $Notes = "DEEPFOLD-SOLVER v$Version" }
}

$pubDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

# Tauri's build names files like "DEEPFOLD - SOLVER_<ver>_x64-setup.exe" because
# productName has spaces. GitHub release asset URLs don't cope well with spaces,
# so rename to the dash-form "DEEPFOLD-SOLVER_<ver>..." when staging.
# Minisign signs file *content*, not filename, so renaming is safe.
function Get-DistName([string]$original) {
    return $original -replace "DEEPFOLD - SOLVER", "DEEPFOLD-SOLVER"
}
$nsisDistName = Get-DistName $nsisExe.Name
$nsisSigDistName = Get-DistName $nsisSig.Name
$msiDistName  = if ($msi) { Get-DistName $msi.Name } else { $null }

$tag = "$TagPrefix$Version"
$nsisUrl = "$BaseUrl/$tag/$nsisDistName"

$manifest = [ordered]@{
    version    = $Version
    notes      = $Notes.Trim()
    pub_date   = $pubDate
    platforms  = [ordered]@{
        "windows-x86_64" = [ordered]@{
            signature = $signature.Trim()
            url       = $nsisUrl
        }
    }
}

# ----- 6. Emit release directory --------------------------------------------
$releaseDir = "release\$Version"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

$manifestJson = $manifest | ConvertTo-Json -Depth 6 -Compress:$false
$manifestPath = Join-Path $releaseDir "latest.json"
# Write UTF-8 WITHOUT BOM. Windows PowerShell 5.1's Set-Content -Encoding UTF8
# adds a BOM (EF BB BF) which makes Tauri v2's updater plugin fail with
# "error decoding response body" — serde_json rejects the leading BOM byte.
# Use raw WriteAllBytes with UTF8Encoding(false) instead.
[System.IO.File]::WriteAllBytes(
    $manifestPath,
    (New-Object System.Text.UTF8Encoding $false).GetBytes($manifestJson)
)

Copy-Item $nsisExe.FullName -Destination (Join-Path $releaseDir $nsisDistName) -Force
Copy-Item $nsisSig.FullName -Destination (Join-Path $releaseDir $nsisSigDistName) -Force
if ($msi) { Copy-Item $msi.FullName -Destination (Join-Path $releaseDir $msiDistName) -Force }

Write-Host ""
Write-Host "=== Release staged at $releaseDir ===" -ForegroundColor Green
Write-Host ""
Write-Host "Manifest ($manifestPath):"
Write-Host "---------------------------------------------------"
Get-Content $manifestPath
Write-Host "---------------------------------------------------"
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Create GitHub release:"
Write-Host "     cd `"$releaseDir`""
Write-Host "     gh release create $tag ``"
Write-Host "       `"$nsisDistName`" ``"
Write-Host "       `"$nsisSigDistName`" ``"
if ($msi) { Write-Host "       `"$msiDistName`" ``" }
Write-Host "       latest.json ``"
Write-Host "       --repo a9876543245/DEEPFOLD-SOLVER ``"
Write-Host "       --title `"DEEPFOLD-SOLVER $tag`" ``"
Write-Host "       --notes-file -  # paste notes from latest.json or CHANGELOG"
Write-Host ""
Write-Host "  2. Once the release is public, the Tauri endpoint"
Write-Host "     (https://github.com/a9876543245/DEEPFOLD-SOLVER/releases/latest/download/latest.json)"
Write-Host "     will auto-resolve to latest.json in this release."
Write-Host ""
Write-Host "  3. Test: install an older version (e.g. v1.0.0), launch app,"
Write-Host "     confirm the update banner appears within a few seconds."
Write-Host ""
