#!/bin/bash
# =================================================================
# 🚀 Multi-Core Scientific Auditor v2.6.1
# Parallel Fortran→JAX translation for the seismic_cpml repository
# Uses local Loki fork at ./loki via sys.path injection in the agent
# =================================================================

set -euo pipefail

SOURCE_DIR="/Users/loicmaurin/PycharmProjects/seismic_cpml"
AGENT_DIR="$(cd "$(dirname "$0")" && pwd)"
AUDIT_DIR="$AGENT_DIR/audit_results"
PIPELINE="$AGENT_DIR/.venv/bin/agent-pipeline"
PARALLELISM=2    # 2 concurrent LLM calls — adjust based on API rate limits
LOG_SUFFIX=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$AUDIT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  🔬 Scientific Auditor v2.6.1 — $(date)"
echo "  📂 Source  : $SOURCE_DIR"
echo "  📁 Results : $AUDIT_DIR"
echo "  🔄 Workers : $PARALLELISM concurrent"
echo "════════════════════════════════════════════════════════════════"

# Collect all .f90 files
TMPLIST=$(mktemp)
find "$SOURCE_DIR" -name "*.f90" -maxdepth 1 | sort > "$TMPLIST"
TOTAL=$(wc -l < "$TMPLIST")
echo "  📋 Found $TOTAL Fortran files to process"
echo ""

process_file() {
    FILE="$1"
    AGENT_DIR="$2"
    AUDIT_DIR="$3"
    PIPELINE="$4"

    FILENAME=$(basename "$FILE")
    FILE_BASE="${FILENAME%.*}"
    RUN_DIR="$AUDIT_DIR/$FILE_BASE"
    LOG="$RUN_DIR/pipeline.log"

    mkdir -p "$RUN_DIR"
    echo "  ⚙️  START: $FILENAME"

    # Run in a subshell with its own working directory copy
    # We create a temp symlink to share .venv without copying it
    WORK_DIR=$(mktemp -d "/tmp/audit_${FILE_BASE}_XXXXXX")
    ln -sf "$AGENT_DIR/.venv"       "$WORK_DIR/.venv"
    ln -sf "$AGENT_DIR/local_code_agent" "$WORK_DIR/local_code_agent"
    ln -sf "$AGENT_DIR/loki"        "$WORK_DIR/loki"
    cp -f  "$AGENT_DIR/.env"        "$WORK_DIR/.env" 2>/dev/null || true
    cp -f  "$AGENT_DIR/pyproject.toml" "$WORK_DIR/pyproject.toml" 2>/dev/null || true

    cd "$WORK_DIR"

    EXIT_CODE=0
    "$PIPELINE" translate "$FILE" > "$LOG" 2>&1 || EXIT_CODE=$?

    # Move generated output to audit results
    if [ -d "$WORK_DIR/output" ]; then
        cp -r "$WORK_DIR/output/." "$RUN_DIR/"
    fi

    # Cleanup temp
    rm -rf "$WORK_DIR"

    STATUS="✅ DONE"
    [ "$EXIT_CODE" -ne 0 ] && STATUS="❌ FAIL (exit=$EXIT_CODE)"
    echo "  $STATUS : $FILENAME"
}

export -f process_file

# Run in parallel using xargs
cat "$TMPLIST" | xargs -I {} -P "$PARALLELISM" bash -c \
    'process_file "$@"' _ {} "$AGENT_DIR" "$AUDIT_DIR" "$PIPELINE"

rm -f "$TMPLIST"

# Build consolidated audit report
REPORT="$AUDIT_DIR/AUDIT_REPORT_${LOG_SUFFIX}.md"
echo "# 🔬 Audit Report — $(date)" > "$REPORT"
echo "" >> "$REPORT"
echo "| File | Status | Report |" >> "$REPORT"
echo "| :--- | :--- | :--- |" >> "$REPORT"

for DIR in "$AUDIT_DIR"/*/; do
    NAME=$(basename "$DIR")
    LOG="$DIR/pipeline.log"
    REPRO="$DIR/tests/reproducibility/*/REPRODUCIBILITY_REPORT.md"
    
    if [ -f "$LOG" ]; then
        if grep -q "Pipeline Complete" "$LOG" 2>/dev/null; then
            STATUS="✅ Complete"
        elif grep -q "Error\|Traceback" "$LOG" 2>/dev/null; then
            STATUS="❌ Error"
        else
            STATUS="⏳ Partial"
        fi
    else
        STATUS="❔ No log"
    fi
    
    echo "| $NAME | $STATUS | [log]($LOG) |" >> "$REPORT"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Audit Complete!"
echo "  📊 Report: $REPORT"
echo "════════════════════════════════════════════════════════════════"
