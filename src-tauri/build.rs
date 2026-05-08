fn main() {
    // Re-run cargo build whenever the OAuth client_secret env var changes.
    // The constant in src/oauth.rs is filled via `option_env!` at compile
    // time, but cargo's incremental cache doesn't track `option_env!`
    // dependencies automatically — without this hint, flipping the env
    // var between builds would silently keep the old compiled value.
    // (v1.4.x / v1.5.0 shipped with an empty secret because of exactly
    // this gap, breaking Google sign-in with a 400 Token exchange error.)
    println!("cargo:rerun-if-env-changed=DEEPFOLD_GOOGLE_CLIENT_SECRET");

    tauri_build::build()
}
