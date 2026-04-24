/// Google OAuth for Tauri desktop app — Authorization Code + PKCE flow.
///
/// Desktop OAuth clients do NOT accept `response_type=id_token` (implicit).
/// They must use the Authorization Code flow with PKCE, per Google's native
/// app guidance (https://developers.google.com/identity/protocols/oauth2/native-app).
///
/// Flow:
/// 1. Frontend invokes `start_google_oauth`.
/// 2. Rust binds an ephemeral localhost port, generates a PKCE verifier +
///    SHA256 challenge + CSRF state token.
/// 3. Rust opens the system browser to Google's auth URL with
///    `response_type=code` and the PKCE challenge.
/// 4. User signs in in the real browser. Google redirects to
///    `http://127.0.0.1:<port>/callback?code=…&state=…` (query string — the
///    code arrives at our server directly, no JS extraction needed).
/// 5. Our /callback handler validates state, exchanges the code for tokens
///    at `https://oauth2.googleapis.com/token` (POST with
///    code + code_verifier + client_id + redirect_uri — no client secret,
///    that's what PKCE is for), extracts `id_token`, emits Tauri event
///    `oauth-google-token`, and shuts down the local server.
/// 6. Frontend posts the id_token to DEEPFOLD's `/api/auth/google`, gets a
///    JWT, and proceeds.

use axum::{
    extract::{Query, State},
    response::{Html, IntoResponse},
    routing::get,
    Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tauri_plugin_opener::OpenerExt;
use tokio::sync::{oneshot, Mutex};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Google OAuth 2.0 Client ID (Desktop application type).
///
/// Lives in the same Google Cloud Project as the deepfold.co Web client
/// (project number 230259880631), so id_tokens issued here share the same
/// `sub` and `email` with website logins. The DEEPFOLD API accepts this
/// Client ID's `aud` value — see `marketing/BACKEND-AUDIENCE-WHITELIST.md`.
const GOOGLE_CLIENT_ID: &str =
    "230259880631-vgghi4kg2vniisjrvegdicl1k5nissk8.apps.googleusercontent.com";

/// Google OAuth 2.0 Client Secret (Desktop client).
///
/// Yes, this is in the binary. Yes, that's correct. Per Google's own
/// documentation on native apps (Protocols/oauth2/native-app):
///
///   "Native apps cannot keep secrets confidential. The OAuth 2.0
///    client_secret for a desktop app should be considered an identifier,
///    not a true secret."
///
/// Google introduced mandatory secrets for *all* new OAuth clients (Web +
/// Desktop) in 2024, which is why PKCE alone is no longer sufficient. The
/// threat model for a Desktop client secret is: an attacker who extracts it
/// from the binary can impersonate *the DEEPFOLD-SOLVER login UI* — they
/// still cannot bypass the backend's `aud` whitelist to forge PRO access,
/// since every id_token is verified against Google's JWKS server-side.
///
/// If this secret is ever rotated in Google Cloud Console, update the
/// `DEEPFOLD_GOOGLE_CLIENT_SECRET` env var at build time and ship a new
/// release; older installers will continue to work until the old secret
/// is explicitly revoked.
///
/// BUILD-TIME INJECTION: the real value is compiled in via env var
/// `DEEPFOLD_GOOGLE_CLIENT_SECRET`, which must be set before `cargo build`
/// / `npm run tauri build`. Without it, sign-in fails cleanly at runtime.
/// Keeping the secret out of source prevents GitHub's secret scanner from
/// blocking pushes; the value is still extractable from the compiled
/// binary, which is the accepted threat model for Desktop OAuth clients.
const GOOGLE_CLIENT_SECRET: &str = match option_env!("DEEPFOLD_GOOGLE_CLIENT_SECRET") {
    Some(s) => s,
    None => "",
};

const GOOGLE_AUTH_URL: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// Server self-destructs if no callback arrives within this many seconds.
const SESSION_TIMEOUT_SECS: u64 = 300;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
pub struct OAuthSession {
    pub port: u16,
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct SharedState {
    app: AppHandle,
    expected_state: String,
    code_verifier: String,
    redirect_uri: String,
    shutdown: Option<oneshot::Sender<()>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_bytes(n: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen::<u8>()).collect()
}

/// Base64url (no padding) of N random bytes. Length of output is ~ceil(4N/3).
fn random_url_safe_b64(byte_count: usize) -> String {
    URL_SAFE_NO_PAD.encode(random_bytes(byte_count))
}

/// PKCE S256 code challenge: base64url(sha256(code_verifier)).
fn pkce_s256_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

/// Minimal URL component encoder. Avoids adding `urlencoding` just for one
/// parameter. Matches `encodeURIComponent` (unreserved set: A-Z a-z 0-9 - _ . ~).
fn encode_uri_component(uri: &str) -> String {
    let mut out = String::with_capacity(uri.len() * 3);
    for b in uri.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub async fn start_google_oauth(app: AppHandle) -> Result<OAuthSession, String> {
    // Bind ephemeral port on localhost (127.0.0.1 is Google's preferred form
    // for desktop OAuth clients — both are allowed but 127.0.0.1 avoids any
    // hosts-file weirdness with "localhost").
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Failed to bind OAuth callback server: {}", e))?;
    let port = listener
        .local_addr()
        .map_err(|e| e.to_string())?
        .port();

    let redirect_uri = format!("http://127.0.0.1:{}/callback", port);

    // 32 random bytes → ~43 URL-safe chars. PKCE RFC requires 43–128 chars.
    let code_verifier = random_url_safe_b64(32);
    let code_challenge = pkce_s256_challenge(&code_verifier);
    let state_token = random_url_safe_b64(16);

    // Build Google auth URL (Authorization Code + PKCE).
    let auth_url = format!(
        "{base}\
         ?response_type=code\
         &client_id={client_id}\
         &redirect_uri={redirect}\
         &scope=openid%20email%20profile\
         &state={state}\
         &code_challenge={challenge}\
         &code_challenge_method=S256\
         &prompt=select_account",
        base = GOOGLE_AUTH_URL,
        client_id = GOOGLE_CLIENT_ID,
        redirect = encode_uri_component(&redirect_uri),
        state = encode_uri_component(&state_token),
        challenge = code_challenge,
    );

    // Shared state so the callback handler can recover the verifier and
    // signal shutdown once done.
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let shared = Arc::new(Mutex::new(SharedState {
        app: app.clone(),
        expected_state: state_token,
        code_verifier,
        redirect_uri: redirect_uri.clone(),
        shutdown: Some(shutdown_tx),
    }));

    let router = Router::new()
        .route("/callback", get(callback_handler))
        .with_state(shared.clone());

    // Spawn server — graceful shutdown on token exchange success OR timeout.
    tokio::spawn(async move {
        let shutdown_future = async move {
            tokio::select! {
                _ = shutdown_rx => {},
                _ = tokio::time::sleep(Duration::from_secs(SESSION_TIMEOUT_SECS)) => {},
            }
        };
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(shutdown_future)
            .await;
    });

    // Open in system browser.
    if let Err(err) = app.opener().open_url(auth_url.clone(), None::<&str>) {
        eprintln!("[oauth] Failed to open system browser: {}", err);
        let _ = app.emit("oauth-google-manual", auth_url);
    }

    Ok(OAuthSession { port })
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CallbackParams {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

async fn callback_handler(
    State(shared): State<Arc<Mutex<SharedState>>>,
    Query(params): Query<CallbackParams>,
) -> impl IntoResponse {
    // --- Error from Google (user cancelled, consent denied, etc.) ---
    if let Some(err) = params.error {
        let msg = params
            .error_description
            .unwrap_or_else(|| "Google returned an error".to_string());
        shutdown_server(&shared).await;
        return Html(error_page(&format!("{}: {}", err, msg)));
    }

    // --- Missing required params ---
    let code = match params.code {
        Some(c) if !c.is_empty() => c,
        _ => {
            shutdown_server(&shared).await;
            return Html(error_page("Missing authorization code"));
        }
    };
    let state = match params.state {
        Some(s) if !s.is_empty() => s,
        _ => {
            shutdown_server(&shared).await;
            return Html(error_page("Missing state parameter"));
        }
    };

    // --- Validate state + capture data for token exchange ---
    let (verifier, redirect_uri, app) = {
        let g = shared.lock().await;
        if state != g.expected_state {
            drop(g);
            shutdown_server(&shared).await;
            return Html(error_page(
                "State mismatch — possible CSRF, refusing to continue",
            ));
        }
        (
            g.code_verifier.clone(),
            g.redirect_uri.clone(),
            g.app.clone(),
        )
    };

    // --- Exchange code for tokens ---
    match exchange_code(&code, &verifier, &redirect_uri).await {
        Ok(id_token) => {
            if let Err(err) = app.emit("oauth-google-token", id_token) {
                eprintln!("[oauth] emit failed: {}", err);
                shutdown_server(&shared).await;
                return Html(error_page(&format!("Event emit failed: {}", err)));
            }
            shutdown_server(&shared).await;
            Html(success_page())
        }
        Err(err) => {
            eprintln!("[oauth] token exchange failed: {}", err);
            shutdown_server(&shared).await;
            Html(error_page(&format!("Token exchange failed: {}", err)))
        }
    }
}

async fn shutdown_server(shared: &Arc<Mutex<SharedState>>) {
    let mut g = shared.lock().await;
    if let Some(tx) = g.shutdown.take() {
        let _ = tx.send(());
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    id_token: String,
}

async fn exchange_code(code: &str, verifier: &str, redirect_uri: &str) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("HTTP client init: {}", e))?;

    let params = [
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", GOOGLE_CLIENT_ID),
        ("client_secret", GOOGLE_CLIENT_SECRET),
        ("code_verifier", verifier),
    ];

    let resp = client
        .post(GOOGLE_TOKEN_URL)
        .form(&params)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!(
            "Google token endpoint returned {}: {}",
            status, body
        ));
    }

    let tokens: TokenResponse = resp
        .json()
        .await
        .map_err(|e| format!("Parse token response: {}", e))?;

    if tokens.id_token.is_empty() {
        return Err("Google returned empty id_token".to_string());
    }

    Ok(tokens.id_token)
}

// ---------------------------------------------------------------------------
// HTML pages (shown to user in the browser tab after OAuth completes)
// ---------------------------------------------------------------------------

fn success_page() -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Signed in · DEEPFOLD-SOLVER</title>
<style>{style}</style></head>
<body><div class="card">
  <div class="logo">D</div>
  <h1 class="ok">Signed in</h1>
  <p>You can close this window and return to DEEPFOLD-SOLVER.</p>
</div>
<script>setTimeout(() => {{ try {{ window.close(); }} catch (_) {{}} }}, 1500);</script>
</body></html>"#,
        style = PAGE_STYLE
    )
}

fn error_page(msg: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Sign-in error · DEEPFOLD-SOLVER</title>
<style>{style}</style></head>
<body><div class="card">
  <div class="logo">D</div>
  <h1 class="err">Sign-in error</h1>
  <p>{msg}</p>
  <p class="small">You can close this window and try again from the app.</p>
</div></body></html>"#,
        style = PAGE_STYLE,
        msg = html_escape(msg)
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

const PAGE_STYLE: &str = r#"
body { font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
       background: #0b0b0d; color: #f2f2f7; margin: 0;
       min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.card { max-width: 480px; padding: 40px; text-align: center;
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; }
h1 { font-size: 18px; margin: 0 0 8px; }
p  { font-size: 14px; color: #a9a9af; line-height: 1.6; margin: 8px 0; }
.small { font-size: 12px; opacity: 0.7; }
.ok  { color: #32D74B; }
.err { color: #FF453A; }
.logo { width: 48px; height: 48px; border-radius: 12px; margin: 0 auto 16px;
        background: linear-gradient(135deg,#0A84FF,#BF5AF2);
        display:flex;align-items:center;justify-content:center;
        font-weight:800;color:#fff;font-size:22px;}
"#;
