#!/bin/bash
# Training Monitor Agent - Quick Launch Script
# Monitors SLURM/Python jobs and launches small object detection training
# MAX 50 EPOCHS for vulcan cluster

set -e

cd "$(dirname "$0")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================================${NC}"
    echo -e "${BLUE}  TRAINING MONITOR AGENT - Small Object Detection${NC}"
    echo -e "${BLUE}  MAX 50 EPOCHS - Vulcan Cluster${NC}"
    echo -e "${BLUE}========================================================${NC}"
}

print_usage() {
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --interactive    Interactive mode (default)"
    echo "  -a, --auto           Auto-start first available config"
    echo "  -c, --config NAME    Run specific configuration"
    echo "  -s, --status         Show current training status"
    echo "  -l, --list           List available configurations"
    echo "  --no-tmux            Use SLURM instead of tmux"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Available configurations (all max 50 epochs):"
    echo "  rf_detr_nano_beads     - RF-DETR Nano with 800 queries (50 epochs)"
    echo "  rf_detr_small_beads    - RF-DETR Small with 800 queries (50 epochs)"
    echo "  rf_detr_medium_beads   - RF-DETR Medium balanced (50 epochs)"
    echo "  rt_detr_v2_beads       - RT-DETR v2 with 800 queries (50 epochs)"
    echo "  rt_detr_dinov2_beads   - RT-DETR v2 DINOv2 backbone (50 epochs)"
    echo "  rf_detr_no300_beads    - RF-DETR with filtered dataset (50 epochs)"
    echo ""
}

check_slurm() {
    if command -v squeue &> /dev/null; then
        echo -e "${GREEN}SLURM available${NC}"
        return 0
    else
        echo -e "${YELLOW}SLURM not available - will use tmux mode${NC}"
        return 1
    fi
}

check_tmux() {
    if command -v tmux &> /dev/null; then
        echo -e "${GREEN}tmux available${NC}"
        return 0
    else
        echo -e "${YELLOW}tmux not available${NC}"
        return 1
    fi
}

show_status() {
    print_header
    echo ""
    echo -e "${YELLOW}Current Training Status:${NC}"
    echo ""

    # Check SLURM jobs
    if command -v squeue &> /dev/null; then
        echo "SLURM Jobs:"
        squeue -u "$USER" --format="%j %i %T %N" 2>/dev/null || echo "  No jobs found"
    else
        echo "SLURM: not available"
    fi

    echo ""
    echo "Training Processes:"
    ps aux | grep -E "train_(rf_detr|rt_detr|yol)" | grep -v grep || echo "  No training processes"

    echo ""
    echo -e "${YELLOW}Note: All training limited to 50 epochs max${NC}"
}

list_configs() {
    print_header
    echo ""
    echo -e "${YELLOW}Available Small Object Detection Configurations (MAX 50 EPOCHS):${NC}"
    echo ""
    echo "1. rf_detr_nano_beads"
    echo "   - RF-DETR Nano with 800 queries for tiny objects"
    echo "   - Best for: Very small beads, fast training"
    echo "   - Epochs: 50 (max)"
    echo ""
    echo "2. rf_detr_small_beads"
    echo "   - RF-DETR Small with 800 queries"
    echo "   - Best for: Small beads, balanced speed/accuracy"
    echo "   - Epochs: 50 (max)"
    echo ""
    echo "3. rf_detr_medium_beads"
    echo "   - RF-DETR Medium with 600 queries"
    echo "   - Best for: General small object detection"
    echo "   - Epochs: 50 (max)"
    echo ""
    echo "4. rt_detr_v2_beads"
    echo "   - RT-DETR v2 ResNet50 with 800 queries"
    echo "   - Best for: Real-time small object detection"
    echo "   - Epochs: 50 (max)"
    echo ""
    echo "5. rt_detr_dinov2_beads"
    echo "   - RT-DETR v2 DINOv2 backbone"
    echo "   - Best for: Best features for small objects"
    echo "   - Epochs: 50 (max)"
    echo ""
    echo "6. rf_detr_no300_beads"
    echo "   - RF-DETR with filtered dataset (no >300 bbox images)"
    echo "   - Best for: Cleaner training data"
    echo "   - Epochs: 50 (max)"
    echo ""
    echo -e "${YELLOW}Note: All configurations limited to 50 epochs for vulcan cluster${NC}"
    echo ""
}

# Parse arguments
MODE="interactive"
CONFIG=""
USE_TMUX=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interactive)
            MODE="interactive"
            shift
            ;;
        -a|--auto)
            MODE="auto"
            shift
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -s|--status)
            show_status
            exit 0
            ;;
        -l|--list)
            list_configs
            exit 0
            ;;
        --no-tmux)
            USE_TMUX=false
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check environment
echo ""
print_header
echo ""

# Activate uv environment
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found. Please install uv first.${NC}"
    exit 1
fi

# Build command args
PYTHON_ARGS=()

if [ "$MODE" = "auto" ]; then
    PYTHON_ARGS+=("--auto")
fi

if [ -n "$CONFIG" ]; then
    PYTHON_ARGS+=("--config" "$CONFIG")
fi

if [ "$USE_TMUX" = "false" ]; then
    PYTHON_ARGS+=("--no-tmux")
fi

# Run the agent
echo -e "${GREEN}Starting Training Monitor Agent...${NC}"
echo -e "${YELLOW}Note: All training runs limited to 50 epochs max${NC}"
echo ""

exec uv run python training_monitor_agent.py "${PYTHON_ARGS[@]}"
