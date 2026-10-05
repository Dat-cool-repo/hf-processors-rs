#!/usr/bin/env bash
# Run cargo inside WSL with the project's environment. Usage: scripts/cargo.sh test --release
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec cargo "$@"
