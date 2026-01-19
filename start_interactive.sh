#!/bin/bash

# Usage: ./start_interactive.sh [account_name]

# Priority: 1. Argument, 2. Environment Variable, 3. Hardcoded Default
ACCOUNT=${1:-${CC_ACCOUNT:-def-youruser}}

echo "Requesting interactive session on Compute Canada cluster..."
echo "Account: $ACCOUNT"
echo "Resources: 1 Node, 1 GPU, 4 CPUs, 32GB Mem, 3 Hours"

salloc --account=$ACCOUNT \
       --nodes=1 \
       --gpus-per-node=1 \
       --cpus-per-task=4 \
       --mem=32000M \
       --time=3:00:00 \
       --x11
