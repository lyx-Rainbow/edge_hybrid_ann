#!/bin/bash
# ============================================================
# One-click full-dataset parameter sweep runner.
#
# Runs all 8 sweeps (4 indexes x 2 datasets) sequentially
# with timing, progress, and error recovery.
#
# Usage:
#   bash run_all_sweeps.sh              # Run all sweeps
#   bash run_all_sweeps.sh --dry-run    # Print commands without running
#   bash run_all_sweeps.sh --skip curator  # Skip a specific index
#
# Output:
#   - stdout: color-coded progress
#   - logs/sweep_YYYYMMDD_HHMMSS.log: full transcript
# ============================================================
set -o pipefail

# ---- Config ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sweep_${TIMESTAMP}.log"
FAILED_TASKS="$LOG_DIR/failed_${TIMESTAMP}.txt"
rm -f "$FAILED_TASKS"
touch "$FAILED_TASKS"

# ---- Color helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ---- Sweep definitions ----
# Format: "index_name|config|sweep_config"
# Order: fastest first, so we get quick feedback
declare -a SWEEPS=(
    "Pre-Filtering yfcc100m|Pre-Filtering/run_prefiltering_sweep.py --config 3_Config/Pre-Filtering/prefiltering_yfcc100m.json --sweep 3_Config/Pre-Filtering/sweep.json"
    "Pre-Filtering arxiv|Pre-Filtering/run_prefiltering_sweep.py --config 3_Config/Pre-Filtering/prefiltering_arxiv.json --sweep 3_Config/Pre-Filtering/sweep.json"
    "DiskIVF yfcc100m|DiskIVF-PostFiltering/run_diskivf_sweep.py --config 3_Config/DiskIVF-PostFiltering/diskivf_yfcc100m.json --sweep 3_Config/DiskIVF-PostFiltering/sweep.json"
    "DiskIVF arxiv|DiskIVF-PostFiltering/run_diskivf_sweep.py --config 3_Config/DiskIVF-PostFiltering/diskivf_arxiv.json --sweep 3_Config/DiskIVF-PostFiltering/sweep.json"
    "SPANN yfcc100m|SPANN-PostFiltering/run_spann_sweep.py --config 3_Config/SPANN-PostFiltering/spann_yfcc100m.json --sweep 3_Config/SPANN-PostFiltering/sweep.json"
    "SPANN arxiv|SPANN-PostFiltering/run_spann_sweep.py --config 3_Config/SPANN-PostFiltering/spann_arxiv.json --sweep 3_Config/SPANN-PostFiltering/sweep.json"
    "Curator yfcc100m|Curator/run_curator_sweep.py --config 3_Config/Curator/curator_yfcc100m.json --sweep 3_Config/Curator/sweep_full.json"
    "Curator arxiv|Curator/run_curator_sweep.py --config 3_Config/Curator/curator_arxiv.json --sweep 3_Config/Curator/sweep_full.json"
)

DRY_RUN=false
SKIP_INDEXES=""

# ---- Parse args ----
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --skip)
            shift
            SKIP_INDEXES="$SKIP_INDEXES $1"
            shift
            ;;
        --skip=*)
            SKIP_INDEXES="$SKIP_INDEXES ${arg#--skip=}"
            ;;
    esac
done

# ---- Helpers ----
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

run_with_timing() {
    local label="$1"
    local cmd="$2"

    # Check skip
    for skip in $SKIP_INDEXES; do
        if echo "$label" | grep -qi "$skip"; then
            log "  ${YELLOW}[SKIP]${NC} $label (matched --skip $skip)"
            return 0
        fi
    done

    log ""
    log "${BOLD}${CYAN}┌──────────────────────────────────────────────────────────────┐${NC}"
    log "${BOLD}${CYAN}│${NC} ${BOLD}[$((DONE + 1))/${TOTAL}]${NC} $label"
    log "${BOLD}${CYAN}└──────────────────────────────────────────────────────────────┘${NC}"
    log "  Command: python $cmd"
    log "  Start:   $(date '+%Y-%m-%d %H:%M:%S')"

    if [ "$DRY_RUN" = true ]; then
        log "  ${YELLOW}[DRY-RUN]${NC} Would execute: python $cmd"
        return 0
    fi

    local t0
    t0=$(date +%s)

    if python $cmd >> "$LOG_FILE" 2>&1; then
        local t1
        t1=$(date +%s)
        local elapsed=$((t1 - t0))
        local h=$((elapsed / 3600))
        local m=$(((elapsed % 3600) / 60))
        local s=$((elapsed % 60))
        log ""
        log "  ${GREEN}${BOLD}[OK]${NC} $label ${GREEN}completed in ${h}h ${m}m ${s}s${NC}"
        return 0
    else
        local t1
        t1=$(date +%s)
        local elapsed=$((t1 - t0))
        local h=$((elapsed / 3600))
        local m=$(((elapsed % 3600) / 60))
        local s=$((elapsed % 60))
        log ""
        log "  ${RED}${BOLD}[FAIL]${NC} $label ${RED}failed after ${h}h ${m}m ${s}s${NC}"
        echo "$label" >> "$FAILED_TASKS"
        return 1
    fi
}

# ================================================================
# Main
# ================================================================

TOTAL=${#SWEEPS[@]}
DONE=0
PASSED=0
FAILED=0
OVERALL_START=$(date +%s)

log ""
log "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
log "${BOLD}${BLUE}║${NC}  ${BOLD}Full-Dataset Parameter Sweep Runner${NC}"
log "${BOLD}${BLUE}║${NC}  Datasets: yfcc100m (800K train), arxiv (1.6M train)"
log "${BOLD}${BLUE}║${NC}  Indexes:  Pre-Filtering, DiskIVF, SPANN, Curator"
log "${BOLD}${BLUE}║${NC}  Queries:  500 SL + 50 filters x 40 CP (yfcc100m only)"
log "${BOLD}${BLUE}║${NC}  ${TOTAL} tasks total"
log "${BOLD}${BLUE}║${NC}  Start:    $(date '+%Y-%m-%d %H:%M:%S')"
log "${BOLD}${BLUE}║${NC}  Log:      $LOG_FILE"
if [ "$DRY_RUN" = true ]; then
    log "${BOLD}${BLUE}║${NC}  ${YELLOW}Mode:     DRY-RUN (no commands executed)${NC}"
fi
log "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

for entry in "${SWEEPS[@]}"; do
    LABEL="${entry%%|*}"
    CMD="${entry#*|}"

    run_with_timing "$LABEL" "$CMD"
    status=$?

    DONE=$((DONE + 1))
    if [ $status -eq 0 ]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
    fi

    # Print progress
    log "  ${BLUE}Progress:${NC} $DONE/$TOTAL done ($PASSED passed, $FAILED failed)"
done

# ================================================================
# Summary
# ================================================================
OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END - OVERALL_START))
H=$((OVERALL_ELAPSED / 3600))
M=$(((OVERALL_ELAPSED % 3600) / 60))
S=$((OVERALL_ELAPSED % 60))

log ""
log "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
log "${BOLD}${BLUE}║${NC}  ${BOLD}Sweep Complete${NC}"
log "${BOLD}${BLUE}║${NC}  Total time: ${H}h ${M}m ${S}s"
log "${BOLD}${BLUE}║${NC}  Passed: ${GREEN}$PASSED${NC} / $TOTAL"
if [ $FAILED -gt 0 ]; then
    log "${BOLD}${BLUE}║${NC}  Failed: ${RED}$FAILED${NC} / $TOTAL"
    log "${BOLD}${BLUE}║${NC}  Failed tasks:"
    while IFS= read -r task; do
        log "${BOLD}${BLUE}║${NC}    ${RED}- $task${NC}"
    done < "$FAILED_TASKS"
else
    log "${BOLD}${BLUE}║${NC}  ${GREEN}All tasks passed!${NC}"
fi
log "${BOLD}${BLUE}║${NC}  Log: $LOG_FILE"
log "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
log ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi
exit 0
