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
#   - Public key in src-tauri\tauri.conf.json (already set — don't change
#     after the first release or existing installs will reject updates)
#
# Signing strategy:
#   The build is run with `createUpdaterArtifacts: false` to skip Tauri's
#   in-build signing (it hangs on stdin password prompts in PowerShell 5.1 +
#   npm.cmd in some setups, even when TAURI_SIGNING_PRIVATE_KEY_PASSWORD is
#   set to ""). The .sig file is then created manually post-build via
#   `tauri-cli signer sign --password ""` which is reliable across shells.
#   `createUpdaterArtifacts: true` in tauri.conf.json is kept so the runtime
#   updater plugin still verifies signatures normally on end-user machines.
#
# PowerShell quirks handled:
#   - `git fetch --tags` runs early so the changelog step can resolve the
#     previous release tag (gh release create pushes tags but doesn't pull
#     them back locally).
#   - The `npm run tauri build` invocation runs with `$ErrorActionPreference
#     = 'Continue'` and stderr merged via `2>&1`. PowerShell 5.1 wraps every
#     line on a native command's stderr as an ErrorRecord, and npm/tauri
#     log benign progress on stderr ("Info Looking up installed tauri
#     packages...") that would otherwise crash the script under the
#     top-level `Stop` preference even when the build itself succeeded.
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

# Fetch tags from origin so the changelog step (`git log v<prev>..HEAD`)
# can find the previous release tag. `gh release create` pushes the tag to
# the remote but doesn't pull it back to local refs — so a fresh clone (or
# a machine that hasn't fetched recently) will silently produce empty
# release notes if we don't do this. `--quiet` suppresses progress chatter,
# `2>$null` swallows the "couldn't reach remote" case (offline / detached).
& git fetch --tags --quiet origin 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [!] git fetch --tags failed (offline?). Notes may be incomplete." -ForegroundColor Yellow
    $global:LASTEXITCODE = 0  # don't poison the rest of the script
}

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
# NOTE: don't set TAURI_SIGNING_PRIVATE_KEY here. Newer tauri-cli versions
# (≥ 2.x) reject the combination of `-f <path>` (which step 3.5 passes via
# CLI) AND the env var being set with the same path — error: "the argument
# '--private-key-path' cannot be used with '--private-key'". The variable
# was originally set to feed the in-build signer; we disabled in-build
# signing back in v1.2.0, so the env var is no longer load-bearing.
# Also unset it preemptively in case the calling shell exported it (the
# error above appeared on a session that had it pre-exported).
Remove-Item Env:TAURI_SIGNING_PRIVATE_KEY -ErrorAction SilentlyContinue
if (-not $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD) {
    $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = ""
}
Write-Host "Signing key: $keyPath" -ForegroundColor DarkGray

# ----- 2.5 Load build-time secrets from .env.local --------------------------
#
# The OAuth client_secret needed for Google Desktop sign-in (post-2024
# clients require it even though Desktop secrets are not "true" secrets per
# Google's own docs) is read at compile time via `option_env!` in
# src-tauri/src/oauth.rs. If `DEEPFOLD_GOOGLE_CLIENT_SECRET` isn't set when
# `cargo build` runs, the constant compiles to "" and every released install
# fails sign-in with "Token exchange failed: Google Token endpoint returned
# 400 bad request".
#
# v1.4.0 / v1.4.1 / v1.5.0 all shipped with an empty secret because this
# loader didn't exist — the file lived at D:\DEEPFOLD-SOLVER\.env.local and
# the operator was supposed to source it manually before running this
# script. Easy to forget. Doing it inline here removes the foot-gun.
#
# Format expected: KEY=value (one per line), comments with `#`, blank lines
# allowed. Values may be quoted with single or double quotes (stripped).
# $repoRoot is computed later for releaseDir staging; do it eagerly here.
if (-not $repoRoot) { $repoRoot = Split-Path $PSScriptRoot -Parent }
$envFiles = @(
    Join-Path $repoRoot ".env.local"
    Join-Path (Split-Path -Parent $repoRoot) "DEEPFOLD-SOLVER\.env.local"
)
$loadedSecret = $false
foreach ($envFile in $envFiles) {
    if (-not (Test-Path $envFile)) { continue }
    Write-Host "Loading build-time env from $envFile" -ForegroundColor DarkGray
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) { return }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { return }
        $key = $line.Substring(0, $eq).Trim()
        $val = $line.Substring($eq + 1).Trim()
        if ($val.StartsWith('"') -and $val.EndsWith('"')) {
            $val = $val.Substring(1, $val.Length - 2)
        } elseif ($val.StartsWith("'") -and $val.EndsWith("'")) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        Set-Item -Path "Env:$key" -Value $val
        if ($key -eq "DEEPFOLD_GOOGLE_CLIENT_SECRET") {
            $script:loadedSecret = $true
        }
    }
    break
}
if (-not $env:DEEPFOLD_GOOGLE_CLIENT_SECRET) {
    Write-Host "  [!] DEEPFOLD_GOOGLE_CLIENT_SECRET is not set. The build" `
               "will compile but Google sign-in will fail at runtime with" `
               "'Token exchange failed: 400 bad request'. Add it to" `
               ".env.local in the repo root or as a shell env var." `
               -ForegroundColor Yellow
} else {
    Write-Host "  DEEPFOLD_GOOGLE_CLIENT_SECRET is set (length: $($env:DEEPFOLD_GOOGLE_CLIENT_SECRET.Length))" -ForegroundColor DarkGray
}

# ----- 3. Build (unless skipped) --------------------------------------------
#
# We DISABLE Tauri's in-build minisign signing here (`--config` override) and
# do it manually in step 3.5 below. Reason: Tauri's bundler reads
# TAURI_SIGNING_PRIVATE_KEY_PASSWORD from the env, but in some Windows shells
# (notably PowerShell 5.1 invoking npm.cmd) the empty-string password doesn't
# propagate cleanly through the npm → cargo → tauri-bundler → signer chain.
# When that fails the signer prompts on stdin and the build hangs forever
# (bundling completes, .exe lands in target/release/bundle/nsis/, then nothing).
# Doing signing as an explicit post-build step with the `--password` CLI flag
# is reliable across all shells. The override only affects build-time signing;
# `createUpdaterArtifacts: true` in tauri.conf.json stays so the runtime
# updater plugin still verifies signatures normally.
if (-not $SkipBuild) {
    Write-Host ""
    Write-Host "Building (cargo tauri build, in-build signing disabled)..." -ForegroundColor Cyan
    # Pass --config via a temp JSON file rather than an inline argument —
    # PowerShell quoting + npm.cmd argument forwarding mangle inline JSON
    # ("{\"bundle\":...}") in subtle ways that vary by shell version.
    $tmpConfig = New-TemporaryFile
    try {
        Set-Content -Path $tmpConfig.FullName -Value '{"bundle":{"createUpdaterArtifacts":false}}' -Encoding ASCII -NoNewline

        # PowerShell 5.1 wraps every line on a native command's stderr as
        # an ErrorRecord, and `$ErrorActionPreference = 'Stop'` (set at the
        # top of this script) treats those wrappers as fatal. npm/tauri
        # routinely log benign progress on stderr ("Info Looking up
        # installed tauri packages..."), which would crash the script even
        # though the build itself succeeds (.exe lands in bundle/nsis/).
        #
        # Two-part fix:
        #   1. Relax $ErrorActionPreference to 'Continue' around the call so
        #      stderr-wrapped ErrorRecords don't terminate the script.
        #   2. Capture stdout AND stderr to a log file via cmd.exe — this
        #      bypasses PowerShell's `2>&1 | ForEach-Object` which can mangle
        #      argv on npm.cmd batch files (saw "Unknown command: pm" — the
        #      'n' got eaten by the redirection wrapper).
        # Trust $LASTEXITCODE for the actual pass/fail decision.
        $prevPref = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $buildLog = Join-Path $env:TEMP "deepfold_tauri_build_$Version.log"
        try {
            cmd.exe /c "npm run tauri build -- --config `"$($tmpConfig.FullName)`" > `"$buildLog`" 2>&1"
            $exitCode = $LASTEXITCODE
            # Echo the captured log so the operator sees the build progress
            # in the terminal regardless of pass/fail.
            if (Test-Path $buildLog) {
                Get-Content $buildLog | Out-Default
            }
            if ($exitCode -ne 0) { throw "Tauri build failed (exit $exitCode); log at $buildLog" }
        } finally {
            $ErrorActionPreference = $prevPref
            # Clean up log on success; leave it on failure for diagnostics.
            if ($LASTEXITCODE -eq 0 -and (Test-Path $buildLog)) {
                Remove-Item $buildLog -ErrorAction SilentlyContinue
            }
        }
    } finally {
        Remove-Item $tmpConfig.FullName -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Skipping build (-SkipBuild)" -ForegroundColor DarkGray
}

# ----- 3.5 Sign NSIS installer (manual, post-build) -------------------------
$bundleDir = "src-tauri\target\release\bundle"
$nsisExeForSigning = Get-ChildItem "$bundleDir\nsis\*$Version*setup.exe" -File | Select-Object -First 1
if (-not $nsisExeForSigning) { throw "NSIS installer not found in $bundleDir\nsis\ — build must have failed" }

$existingSig = Get-ChildItem "$bundleDir\nsis\*$Version*setup.exe.sig" -File -ErrorAction SilentlyContinue | Select-Object -First 1
# v1.5.1: re-sign when the .exe has been rebuilt since the last .sig was
# produced, even at the same version number. Previous logic reused a stale
# .sig from an older build and shipped it alongside a fresh .exe — the
# signature wouldn't verify, so the auto-updater would refuse to apply the
# update. Compare timestamps; only skip when both files exist AND the .sig
# is at least as new as the .exe.
$sigIsStale = $false
if ($existingSig) {
    $sigIsStale = ($nsisExeForSigning.LastWriteTimeUtc -gt $existingSig.LastWriteTimeUtc)
    if ($sigIsStale) {
        Write-Host "Existing .sig is older than .exe — re-signing." -ForegroundColor Yellow
        Remove-Item -LiteralPath $existingSig.FullName -Force
        $existingSig = $null
    }
}
if (-not $existingSig -or $SkipBuild) {
    Write-Host ""
    Write-Host "Signing $($nsisExeForSigning.Name) (manual, --password CLI flag)..." -ForegroundColor Cyan
    # Wrap via cmd.exe for the same reason the build call does (above):
    # PowerShell 5.1's `& npx @tauri-apps/cli ... -p "" ...` mangles args
    # — the `@` prefix can be misread as splat, the empty-string `""` for
    # password gets stripped, and PS wraps npx's stderr lines as
    # ErrorRecords that trip $ErrorActionPreference. cmd.exe handles
    # native command quoting and stderr cleanly.
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        cmd.exe /c "npx --yes @tauri-apps/cli signer sign -f `"$keyPath`" -p `"`" `"$($nsisExeForSigning.FullName)`""
        if ($LASTEXITCODE -ne 0) { throw "Manual signer failed (exit $LASTEXITCODE)" }
    } finally {
        $ErrorActionPreference = $prevPref
    }
} else {
    Write-Host "Existing signature found — skipping manual sign" -ForegroundColor DarkGray
}

# ----- 4. Collect artifacts -------------------------------------------------
$nsisExe  = $nsisExeForSigning
$nsisSig  = Get-ChildItem "$bundleDir\nsis\*$Version*setup.exe.sig" -File | Select-Object -First 1
$msi      = Get-ChildItem "$bundleDir\msi\*$Version*.msi" -File -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $nsisSig) { throw "NSIS signature (.sig) not found after manual signing — check signer output above" }

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
# Anchor on the script's parent (= repo root) regardless of any CWD drift
# that happened during build/sign. Got bitten by this in v1.3.0: relative
# `release\$Version` was resolving against D:\DEEPFOLD-SOLVER (work dir,
# inherited from the calling shell) instead of D:\DEEPFOLD-SOLVER-repo,
# so WriteAllBytes blew up on a non-existent parent.
$repoRoot = Split-Path $PSScriptRoot -Parent
$releaseDir = Join-Path $repoRoot "release\$Version"
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
