# Sourced by the other scripts (Linux / WSL / macOS). Project-local build and test environment.
#
# Machine-specific settings (a separate target dir, a data drive for the Hugging Face cache, the
# full golden references, the build-job limit...) go in scripts/env.local.sh, which is not
# committed. Example:
#   export CARGO_TARGET_DIR=$HOME/target/hf-processors-rs
#   export CARGO_BUILD_JOBS=4
#   export HF_PROCESSORS_GOLDEN_FULL=/data/hf-processors-rs/golden-full
#   export HF_HOME=/data/hf-processors-rs/hf-home
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
export PATH=$HOME/.local/bin:$PATH
_hfp_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_hfp_scripts/env.local.sh" ] && source "$_hfp_scripts/env.local.sh"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$_hfp_scripts/../target}
export RUST_TEST_THREADS=${RUST_TEST_THREADS:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Python venv with the pinned reference stack (created by scripts/setup_venv.sh).
export HF_PROCESSORS_VENV=${HF_PROCESSORS_VENV:-$HOME/venvs/hf-processors-rs}
