#!/usr/bin/env bash
# Apply M2.7 patches onto the vendor/llama.cpp submodule.
#
# Usage:
#   bash evals/aloepri-attacks/m2_7/apply-patches.sh           # apply
#   bash evals/aloepri-attacks/m2_7/apply-patches.sh --revert  # restore clean
#   bash evals/aloepri-attacks/m2_7/apply-patches.sh --check   # dry-run
#
# Run this once after `git submodule update --init` and before
# `docker build -f .../vulkan-m2_7.Dockerfile`. The submodule
# pointer in the parent repo stays at the clean upstream commit;
# the patches live in evals/aloepri-attacks/m2_7/patches/ and are
# applied to the working tree only.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
SUBMODULE=$REPO_ROOT/vendor/llama.cpp
PATCH_DIR=$REPO_ROOT/evals/aloepri-attacks/m2_7/patches

if [[ ! -e $SUBMODULE/.git ]]; then
    echo "error: $SUBMODULE is not initialised. Run:"
    echo "  git submodule update --init --recursive vendor/llama.cpp"
    exit 1
fi

PATCHES=("$PATCH_DIR"/*.patch)
if [[ ${#PATCHES[@]} -eq 0 || ! -e ${PATCHES[0]} ]]; then
    echo "no patches found in $PATCH_DIR"
    exit 1
fi

mode=${1:-apply}
case $mode in
    apply)
        for p in "${PATCHES[@]}"; do
            echo "applying $(basename "$p")"
            git -C "$SUBMODULE" apply --whitespace=nowarn "$p"
        done
        echo "done. Submodule working tree is now patched; submodule pointer"
        echo "in parent repo is unchanged (still clean upstream)."
        ;;
    --check|check)
        for p in "${PATCHES[@]}"; do
            echo "checking $(basename "$p")"
            git -C "$SUBMODULE" apply --check "$p"
        done
        echo "all patches apply cleanly"
        ;;
    --revert|revert)
        for p in "${PATCHES[@]}"; do
            echo "reverting $(basename "$p")"
            git -C "$SUBMODULE" apply -R --whitespace=nowarn "$p"
        done
        echo "done. Submodule working tree restored to clean upstream."
        ;;
    *)
        echo "usage: $0 [apply|--check|--revert]"
        exit 2
        ;;
esac
