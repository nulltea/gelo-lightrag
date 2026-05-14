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

if [[ -e "$INSTALLED_LIB" ]]; then
    echo "[install-aocl-blis] $INSTALLED_LIB already present, skipping."
    exit 0
fi

mkdir -p "$VENDOR_DIR"

if [[ ! -d "$SRC_DIR/.git" ]]; then
    echo "[install-aocl-blis] cloning amd/blis..."
    git clone --depth 1 --branch "$AOCL_BLIS_REF" https://github.com/amd/blis.git "$SRC_DIR"
fi

cd "$SRC_DIR"

echo "[install-aocl-blis] configuring (amdzen, openmp, shared)..."
./configure \
    --enable-cblas \
    --enable-threading=openmp \
    --enable-shared \
    --prefix="$INSTALL_DIR" \
    amdzen

echo "[install-aocl-blis] building..."
make -j"$(nproc)"

echo "[install-aocl-blis] installing to $INSTALL_DIR..."
make install

# blis-src `system` feature emits `-lblis` (not `-lblis-mt`).
# Symlink so the linker can find it.
cd "$INSTALL_DIR/lib"
for ext in so so.5 so.5.2.2 a; do
    [[ -e "libblis-mt.$ext" && ! -e "libblis.$ext" ]] && ln -sf "libblis-mt.$ext" "libblis.$ext"
done

echo
echo "[install-aocl-blis] done."
echo
echo "To build with AOCL-BLIS in use, export:"
echo "  export RUSTFLAGS='-L $INSTALL_DIR/lib'"
echo "  export LD_LIBRARY_PATH=$INSTALL_DIR/lib"
echo "  export BLIS_NUM_THREADS=1 OMP_NUM_THREADS=1"
echo
echo "Then run benches as usual:"
echo "  cargo test --release --features blas ..."
