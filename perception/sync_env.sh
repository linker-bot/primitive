#!/bin/bash
# Sync conda environment with environment.yml
# - Export: writes current env to environment.yml (captures new packages)
# - Update: installs missing packages from environment.yml (after git pull)
#
# Usage:
#   ./sync_env.sh export   # after installing new packages
#   ./sync_env.sh update   # after git pull on another machine
#   ./sync_env.sh          # auto-detect: update if env differs, else export

set -e

ENV_NAME="robot_perception"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[sync]${NC} $1"; }
warn()  { echo -e "${YELLOW}[sync]${NC} $1"; }
error() { echo -e "${RED}[sync]${NC} $1"; exit 1; }

# Check conda is available
if ! command -v conda &>/dev/null; then
    error "conda not found. Please activate conda first."
fi

# Ensure environment exists
if ! conda env list | grep -qw "^${ENV_NAME}"; then
    if [[ -f "$ENV_FILE" ]]; then
        info "Environment '${ENV_NAME}' not found. Creating from ${ENV_FILE}..."
        conda env create -f "$ENV_FILE"
        info "Done. Run: conda activate ${ENV_NAME}"
        exit 0
    else
        error "Environment '${ENV_NAME}' not found and no ${ENV_FILE} exists."
    fi
fi

do_export() {
    info "Exporting '${ENV_NAME}' to ${ENV_FILE}..."
    conda env export -n "$ENV_NAME" --no-builds > "$ENV_FILE"
    info "Exported. Remember to commit environment.yml"
}

do_update() {
    if [[ ! -f "$ENV_FILE" ]]; then
        error "No ${ENV_FILE} found. Run './sync_env.sh export' first."
    fi
    info "Updating '${ENV_NAME}' from ${ENV_FILE}..."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    info "Environment synced."
}

case "${1:-auto}" in
    export)
        do_export
        ;;
    update)
        do_update
        ;;
    auto)
        if [[ ! -f "$ENV_FILE" ]]; then
            warn "No environment.yml found, exporting current env..."
            do_export
        else
            # Compare: check if current env has packages not in yml or vice versa
            CURRENT=$(conda list -n "$ENV_NAME" --export 2>/dev/null | grep -v "^#" | sort)
            SAVED=$(conda env export -n "$ENV_NAME" --no-builds 2>/dev/null | md5sum)
            ONDISK=$(md5sum < "$ENV_FILE")

            if [[ "$SAVED" != "$ONDISK" ]]; then
                warn "environment.yml differs from current env."
                echo "  [1] update env from yml (install missing packages)"
                echo "  [2] export env to yml (save new packages)"
                read -rp "  Choose [1/2]: " choice
                case "$choice" in
                    1) do_update ;;
                    2) do_export ;;
                    *) error "Invalid choice." ;;
                esac
            else
                info "Environment already in sync."
            fi
        fi
        ;;
    *)
        echo "Usage: $0 [export|update|auto]"
        echo "  export  - Save current env to environment.yml"
        echo "  update  - Sync env from environment.yml (install/remove packages)"
        echo "  auto    - Detect difference and prompt for action"
        exit 1
        ;;
esac
