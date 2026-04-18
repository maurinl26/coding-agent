#!/bin/bash
# test_gpu.sh — Déploie et compile les kernels GPU sur la VM Azure A100
#
# Usage :
#   bash scripts/test_gpu.sh                    # copie output/ + compile
#   bash scripts/test_gpu.sh --check            # vérifie l'environnement GPU distant
#   bash scripts/test_gpu.sh --run              # compile ET lance la simulation
#   bash scripts/test_gpu.sh --bench <orig.f90> # CPU vs GPU speedup benchmark
#   AZURE_GPU_HOST=<ip> bash scripts/test_gpu.sh
#
# Variables d'environnement (ou .env) :
#   AZURE_GPU_HOST     IP publique de la VM GPU (ex: 20.x.x.x)
#   AZURE_GPU_USER     Utilisateur SSH (défaut: azureuser)
#   AZURE_GPU_KEY      Chemin vers la clé privée SSH (défaut: ~/.ssh/id_rsa)
#   GPU_REMOTE_DIR     Répertoire distant (défaut: ~/seismic_gpu)

set -e

# ── Config ───────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep AZURE_GPU | xargs) 2>/dev/null || true
fi

GPU_HOST="${AZURE_GPU_HOST:-}"
GPU_USER="${AZURE_GPU_USER:-azureuser}"
GPU_KEY="${AZURE_GPU_KEY:-${HOME}/.ssh/id_rsa}"
REMOTE_DIR="${GPU_REMOTE_DIR:-~/seismic_gpu}"
LOCAL_OUTPUT="./output"

# ── Helpers ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"
[ -f "$GPU_KEY" ] && ssh_opts="$ssh_opts -i $GPU_KEY"

# ── Validate args ─────────────────────────────────────────────────────────────
if [ -z "$GPU_HOST" ]; then
    echo ""
    echo "Usage: AZURE_GPU_HOST=<ip> bash scripts/test_gpu.sh [--check|--run]"
    echo ""
    echo "Get the GPU VM IP from Azure:"
    echo "  az vm list-ip-addresses -g rg-total-seismic-agent -o table"
    echo ""
    echo "Or set it in .env:"
    echo "  echo 'AZURE_GPU_HOST=20.x.x.x' >> .env"
    exit 1
fi

if [ "$1" = "--bench" ]; then
    ORIG="${2:-}"
    [ -z "$ORIG" ] && { echo "Usage: bash scripts/test_gpu.sh --bench <original.f90>"; exit 1; }
    exec bash scripts/bench_gpu.sh "$ORIG"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  GPU Test — ${GPU_USER}@${GPU_HOST}  →  ${REMOTE_DIR}"
echo "═══════════════════════════════════════════════════════════════"

SSH="ssh $ssh_opts ${GPU_USER}@${GPU_HOST}"
SCP="scp $ssh_opts"

# ── --check : vérifier l'environnement GPU distant ───────────────────────────
if [ "$1" = "--check" ]; then
    echo ""
    echo "--- Remote environment check ---"
    $SSH bash << 'REMOTE'
        echo "=== OS ==="
        uname -a
        echo ""
        echo "=== GPU ==="
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
            || echo "WARNING: nvidia-smi not found"
        echo ""
        echo "=== nvfortran ==="
        which nvfortran && nvfortran --version | head -1 || echo "WARNING: nvfortran not found"
        echo ""
        echo "=== Python + Cython ==="
        python3 --version && python3 -c "import Cython; print('Cython', Cython.__version__)" 2>/dev/null \
            || echo "WARNING: Cython not found"
REMOTE
    ok "Check complete."
    exit 0
fi

# ── Check local output/ ───────────────────────────────────────────────────────
if [ ! -d "$LOCAL_OUTPUT" ]; then
    fail "output/ directory not found. Run agent-gpu first:\n  uv run agent-gpu translate /path/to/kernel.f90"
fi

if [ ! -f "$LOCAL_OUTPUT/Makefile" ] && [ ! -f "$LOCAL_OUTPUT/compile_gpu.sh" ]; then
    warn "No Makefile or compile_gpu.sh in output/ — did the agent run completely?"
fi

# ── Step 1 : Copy output/ to remote ──────────────────────────────────────────
echo ""
echo "Step 1 — Copying output/ to ${GPU_USER}@${GPU_HOST}:${REMOTE_DIR}/"
$SSH "mkdir -p ${REMOTE_DIR}"
$SCP -r "$LOCAL_OUTPUT"/* "${GPU_USER}@${GPU_HOST}:${REMOTE_DIR}/"
ok "Files copied."

# ── Step 2 : Compile on GPU node ─────────────────────────────────────────────
echo ""
echo "Step 2 — Compiling on GPU node ..."
$SSH bash << REMOTE
    set -e
    cd "${REMOTE_DIR}"
    echo "--- Working directory: \$(pwd) ---"
    ls -lh fortran_gpu/*.f90 2>/dev/null || echo "WARNING: no .f90 files found"
    echo ""
    bash compile_gpu.sh
REMOTE
ok "Compilation complete."

# ── Step 3 : Run simulation (optional) ───────────────────────────────────────
if [ "$1" = "--run" ]; then
    echo ""
    echo "Step 3 — Running GPU simulation ..."
    BINARY=$(ls output/fortran_gpu/*.f90 2>/dev/null | head -1 | xargs -I{} basename {} .f90)_gpu
    BINARY="${BINARY:-seismic_cpml_2d_isotropic_second_order_gpu}"
    $SSH bash << REMOTE
        set -e
        cd "${REMOTE_DIR}"
        echo "--- Running: ./${BINARY} ---"
        ./${BINARY} 2>&1 | tail -20
REMOTE
    ok "Simulation complete. Check ${REMOTE_DIR}/energy.dat and image*.gif"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. SSH into the GPU node to inspect results:"
echo "  ssh ${GPU_USER}@${GPU_HOST}"
echo "  cd ${REMOTE_DIR} && ls -lh"
echo "═══════════════════════════════════════════════════════════════"
echo ""
