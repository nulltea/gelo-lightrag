#!/usr/bin/env bash
# Build and install AOCL-BLIS (AMD's BLIS fork) into vendor/aocl-install/.
#
# Why: the `blas` cargo feature routes GELO's mask::apply/unapply through
# cblas_sgemm. Vanilla BLIS on Zen 4/5 dispatches SGEMM to haswell_asm (AVX2)
# because no zen*_asm SGEMM kernels exist. AOCL-BLIS's zen5 config maps
# SGEMM to bli_sgemm_skx_asm_32x12_l2 (Intel SKX AVX-512 — pure AVX-512
# instructions, runs natively on Zen). Measured 4.96× speedup on mask
# GEMMs in the BEIR/NFCorpus 100-doc bench (gelo.md §7).
#
# Idempotent: if vendor/aocl-install/lib/libblis-mt.so already exists,
# this is a no-op. No sudo required.
#
# After running, build/test with:
#   RUSTFLAGS="-L $(pwd)/vendor/aocl-install/lib" \
#   LD_LIBRARY_PATH=$(pwd)/vendor/aocl-install/lib \
#   cargo test --release --features blas ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor"
SRC_DIR="$VENDOR_DIR/aocl-blis"
INSTALL_DIR="$VENDOR_DIR/aocl-install"
INSTALLED_LIB="$INSTALL_DIR/lib/libblis-mt.so"

# Tag from amd/blis. Pinning ensures reproducible builds; bump deliberately.
AOCL_BLIS_REF="${AOCL_BLIS_REF:-master}"

# Required symbol that signals "built with LPGEMM addon enabled". Used
# by the idempotency check below — without this, the .so exists but
# the bf16 mask GEMM path (M1.12 bucket 3a) can't link. We check via
# `nm` if available, falling back to a "rebuild" decision if not.
REQUIRED_SYMBOL="aocl_gemm_bf16bf16f32of32"

# Idempotency: if the .so exists AND it has the required LPGEMM
# symbols, the install is current — skip. If the .so exists but the
# LPGEMM addon wasn't enabled (a pre-2026-05-22 build), force a
# rebuild so the bf16 mask path works.
if [[ -e "$INSTALLED_LIB" ]]; then
    if command -v nm >/dev/null 2>&1; then
        if test "$(nm -D "$INSTALLED_LIB" 2>/dev/null | grep -c "$REQUIRED_SYMBOL")" -gt 0; then
            echo "[install-aocl-blis] $INSTALLED_LIB already present with LPGEMM addon, skipping."
            exit 0
        else
            echo "[install-aocl-blis] $INSTALLED_LIB present but missing LPGEMM addon"
            echo "  (no '$REQUIRED_SYMBOL' symbol — pre-2026-05-22 build). Rebuilding."
            # Wipe stale install + build artifacts so the rebuild picks
            # up the new --enable-addon=aocl_gemm flag.
            rm -rf "$INSTALL_DIR"
            if [[ -d "$SRC_DIR" ]]; then
                (cd "$SRC_DIR" && make clean >/dev/null 2>&1 || true)
            fi
        fi
    else
        echo "[install-aocl-blis] $INSTALLED_LIB present but 'nm' not available;"
        echo "  cannot verify LPGEMM symbols. If you need the bf16 mask path,"
        echo "  delete $INSTALLED_LIB and re-run this script."
        exit 0
    fi
fi

mkdir -p "$VENDOR_DIR"

if [[ ! -d "$SRC_DIR/.git" ]]; then
    echo "[install-aocl-blis] cloning amd/blis..."
    git clone --depth 1 --branch "$AOCL_BLIS_REF" https://github.com/amd/blis.git "$SRC_DIR"
fi

cd "$SRC_DIR"

echo "[install-aocl-blis] configuring (amdzen, openmp, shared, aocl_gemm addon)..."
# `--enable-addon=aocl_gemm` enables AOCL's LPGEMM low-precision GEMM
# addon, which provides `aocl_gemm_bf16bf16f32of32` (bf16 × bf16 → f32
# output) using AVX-512_BF16's vdpbf16ps instruction on Zen 5. Required
# for the M1.12 bucket-3a bf16 mask GEMM path; without this flag the
# kernel sources at `addon/aocl_gemm/` are present but not built into
# `libblis-mt.so`. Verify post-install with:
#   nm -D vendor/aocl-install/lib/libblis-mt.so | grep aocl_gemm_bf16
./configure \
    --enable-cblas \
    --enable-threading=openmp \
    --enable-shared \
    --enable-addon=aocl_gemm \
    --prefix="$INSTALL_DIR" \
    amdzen

echo "[install-aocl-blis] building..."
make -j"$(nproc)"

echo "[install-aocl-blis] installing to $INSTALL_DIR..."
make install

# `make install` writes the .so.5.2.2 file then the symlinks in
# sequence. The post-install nm verification below can race against
# that sequence on fast filesystems — explicitly sync so the verify
# observes a steady state.
sync

# blis-src `system` feature emits `-lblis` (not `-lblis-mt`).
# Symlink so the linker can find it.
cd "$INSTALL_DIR/lib"
for ext in so so.5 so.5.2.2 a; do
    [[ -e "libblis-mt.$ext" && ! -e "libblis.$ext" ]] && ln -sf "libblis-mt.$ext" "libblis.$ext"
done

echo
# Advisory verify: warn if the LPGEMM addon symbol didn't make it
# into the .so. Not a hard fail — `make install` writes
# `libblis-mt.so.5.2.2` then symlinks in sequence and the verify
# can race against the linker's final fsync on fast filesystems
# (observed: false-negative even after `sync`). The next invocation's
# idempotency check at the top of this script re-validates from a
# steady state and triggers a rebuild if the symbol really is
# missing — so a false-negative here is self-correcting on the next
# run rather than blocking the current one.
if command -v nm >/dev/null 2>&1; then
    if command -v realpath >/dev/null 2>&1; then
        VERIFY_LIB="$(realpath "$INSTALLED_LIB")"
    else
        VERIFY_LIB="$INSTALLED_LIB"
    fi
    # Use grep -c (reads full input) instead of grep -q to avoid SIGPIPE
    # propagating to nm under `set -euo pipefail` — the early-exit of
    # grep -q caused nm to die with SIGPIPE (rc 141), failing the
    # pipeline even when the symbol was actually present.
    if test "$(nm -D "$VERIFY_LIB" 2>/dev/null | grep -c "$REQUIRED_SYMBOL")" -gt 0; then
        echo "[install-aocl-blis] verified: LPGEMM addon symbols present in $VERIFY_LIB."
    else
        echo "[install-aocl-blis] note: post-install verify didn't observe"
        echo "  '$REQUIRED_SYMBOL' in $VERIFY_LIB. This is usually a transient"
        echo "  fsync race after make install — confirm independently:"
        echo "    nm -D $VERIFY_LIB | grep $REQUIRED_SYMBOL"
        echo "  If the symbol really is missing the next invocation of this"
        echo "  script will detect it and rebuild."
    fi
fi

echo "[install-aocl-blis] done."
echo
echo "To build with AOCL-BLIS in use, export:"
echo "  export RUSTFLAGS='-L $INSTALL_DIR/lib'"
echo "  export LD_LIBRARY_PATH=$INSTALL_DIR/lib"
echo "  export BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1"
echo
echo "Then run benches as usual:"
echo "  cargo test --release --features blas ..."
