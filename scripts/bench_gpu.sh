#!/bin/bash
# bench_gpu.sh — CPU vs GPU speedup benchmark (Fortran seismic CPML)
#
# Usage:
#   AZURE_GPU_HOST=<ip> bash scripts/bench_gpu.sh <original.f90>
#   AZURE_GPU_HOST=<ip> bash scripts/bench_gpu.sh <original.f90> --nsys
#
# Variables d'environnement (ou .env) :
#   AZURE_GPU_HOST     IP publique de la VM GPU
#   AZURE_GPU_USER     Utilisateur SSH (défaut: azureuser)
#   AZURE_GPU_KEY      Chemin vers la clé privée SSH (défaut: ~/.ssh/id_rsa)
#   GPU_REMOTE_DIR     Répertoire distant (défaut: ~/seismic_gpu)

set -e

if [ -f .env ]; then
    export $(grep -v '^#' .env | grep AZURE_GPU | xargs) 2>/dev/null || true
fi

GPU_HOST="${AZURE_GPU_HOST:-}"
GPU_USER="${AZURE_GPU_USER:-azureuser}"
GPU_KEY="${AZURE_GPU_KEY:-${HOME}/.ssh/id_rsa}"
REMOTE_DIR="${GPU_REMOTE_DIR:-~/seismic_gpu}"
ORIGINAL_F90="${1:-}"
NSYS_FLAG="${2:-}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $*"; }

if [ -z "$GPU_HOST" ]; then
    echo "Usage: AZURE_GPU_HOST=<ip> bash scripts/bench_gpu.sh <original.f90> [--nsys]"
    echo ""
    echo "Get the GPU VM IP:"
    echo "  bash scripts/get_gpu_ip.sh"
    exit 1
fi

if [ -z "$ORIGINAL_F90" ] || [ ! -f "$ORIGINAL_F90" ]; then
    fail "Original Fortran file not found: '${ORIGINAL_F90}'"
fi

if [ ! -f "output/fortran_gpu/kernel_gpu.f90" ]; then
    fail "GPU kernel not found. Run the pipeline first:\n  uv run agent-gpu translate ${ORIGINAL_F90}"
fi

ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"
[ -f "$GPU_KEY" ] && ssh_opts="$ssh_opts -i $GPU_KEY"
SSH="ssh $ssh_opts ${GPU_USER}@${GPU_HOST}"
SCP="scp $ssh_opts"

ORIG_BASENAME=$(basename "$ORIGINAL_F90")
GPU_BASENAME="kernel_gpu.f90"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  CPU vs GPU Benchmark — ${GPU_USER}@${GPU_HOST}"
echo "  CPU source : ${ORIG_BASENAME}"
echo "  GPU source : ${GPU_BASENAME}"
echo "════════════════════════════════════════════════════════════"

# ── Copy files ────────────────────────────────────────────────────────────────
info "Copying sources to ${REMOTE_DIR}/"
$SSH "mkdir -p ${REMOTE_DIR}"
$SCP "$ORIGINAL_F90" "${GPU_USER}@${GPU_HOST}:${REMOTE_DIR}/${ORIG_BASENAME}"
$SCP "output/fortran_gpu/kernel_gpu.f90" "${GPU_USER}@${GPU_HOST}:${REMOTE_DIR}/${GPU_BASENAME}"
ok "Files copied."

# ── Compile and benchmark on GPU node ─────────────────────────────────────────
DO_NSYS="false"
[ "$NSYS_FLAG" = "--nsys" ] && DO_NSYS="true"

$SSH bash << REMOTE
set -e
cd "${REMOTE_DIR}"

# ── Detect GPU architecture ──────────────────────────────────────────────────
GPU_NAME=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
case "\$GPU_NAME" in
    *A100*)  GPU_ARCH="cc80" ;;
    *H100*)  GPU_ARCH="cc90" ;;
    *V100*)  GPU_ARCH="cc70" ;;
    *T4*)    GPU_ARCH="cc75" ;;
    *3090*|*3080*|*3070*) GPU_ARCH="cc86" ;;
    *)       GPU_ARCH="cc70" ;;
esac
echo "[INFO] GPU: \${GPU_NAME:-unknown}  arch: \${GPU_ARCH}"

# ── Compile CPU version (no OpenACC) ─────────────────────────────────────────
echo ""
echo "--- Compiling CPU version (nvfortran -O3, no OpenACC) ---"
nvfortran -O3 -o seismic_cpu "${ORIG_BASENAME}" 2>&1 || {
    echo "[WARN] nvfortran failed for CPU, trying gfortran..."
    gfortran -O3 -o seismic_cpu "${ORIG_BASENAME}"
}
echo "[OK] CPU binary ready."

# ── Compile GPU version (OpenACC) ────────────────────────────────────────────
echo ""
echo "--- Compiling GPU version (nvfortran -acc -gpu=\${GPU_ARCH}) ---"
nvfortran -acc -gpu=\${GPU_ARCH} -Minfo=accel -O3 -o seismic_gpu "${GPU_BASENAME}" 2>&1
echo "[OK] GPU binary ready."

# ── Warmup GPU (avoid first-launch overhead) ─────────────────────────────────
echo ""
echo "--- Warmup GPU (1 run discarded) ---"
./seismic_gpu > /dev/null 2>&1 || true

# ── Benchmark CPU ─────────────────────────────────────────────────────────────
echo ""
echo "=== BENCHMARK CPU ==="
T_CPU_NS=\$(date +%s%N)
./seismic_cpu > cpu_output.txt 2>&1
T_CPU_END=\$(date +%s%N)
T_CPU_MS=\$(( (T_CPU_END - T_CPU_NS) / 1000000 ))
echo "CPU time: \${T_CPU_MS} ms"

# ── Benchmark GPU ─────────────────────────────────────────────────────────────
echo ""
echo "=== BENCHMARK GPU ==="
T_GPU_NS=\$(date +%s%N)
./seismic_gpu > gpu_output.txt 2>&1
T_GPU_END=\$(date +%s%N)
T_GPU_MS=\$(( (T_GPU_END - T_GPU_NS) / 1000000 ))
echo "GPU time: \${T_GPU_MS} ms"

# ── Speedup ───────────────────────────────────────────────────────────────────
SPEEDUP=\$(echo "scale=1; \${T_CPU_MS} / \${T_GPU_MS}" | bc 2>/dev/null || echo "N/A")
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  GPU : \${GPU_NAME}"
echo "  CPU time : \${T_CPU_MS} ms"
echo "  GPU time : \${T_GPU_MS} ms"
echo "  SPEEDUP  : \${SPEEDUP}×"
echo "════════════════════════════════════════════════════════════"

# ── Numerical correctness check ───────────────────────────────────────────────
echo ""
echo "--- Numerical correctness check (velocnorm comparison) ---"
grep -i "velocnorm\|Time step" cpu_output.txt > cpu_velocnorm.txt 2>/dev/null || true
grep -i "velocnorm\|Time step" gpu_output.txt > gpu_velocnorm.txt 2>/dev/null || true

if [ -s cpu_velocnorm.txt ] && [ -s gpu_velocnorm.txt ]; then
    if diff -q cpu_velocnorm.txt gpu_velocnorm.txt > /dev/null 2>&1; then
        echo "[OK] CPU and GPU outputs are numerically identical."
    else
        echo "[WARN] Differences detected in velocnorm output:"
        diff cpu_velocnorm.txt gpu_velocnorm.txt | head -20
    fi
else
    echo "[INFO] No velocnorm output found — skipping numerical check."
fi

# ── NSight Systems profiling (optional) ──────────────────────────────────────
if [ "${DO_NSYS}" = "true" ]; then
    echo ""
    echo "--- NSight Systems profiling ---"
    if command -v nsys &>/dev/null; then
        nsys profile --stats=true -o bench_nsys ./seismic_gpu > nsys_stats.txt 2>&1
        echo "[OK] NSight report: ${REMOTE_DIR}/bench_nsys.nsys-rep"
        echo "[OK] Stats summary: ${REMOTE_DIR}/nsys_stats.txt"
        head -50 nsys_stats.txt
    else
        echo "[WARN] nsys not found — skipping profiling."
    fi
fi

# ── Write benchmark log ───────────────────────────────────────────────────────
cat > benchmark.log << LOG
GPU: \${GPU_NAME}
GPU arch: \${GPU_ARCH}
CPU time (ms): \${T_CPU_MS}
GPU time (ms): \${T_GPU_MS}
Speedup: \${SPEEDUP}x
LOG
echo ""
echo "[OK] Results saved to ${REMOTE_DIR}/benchmark.log"
REMOTE

ok "Benchmark complete."
