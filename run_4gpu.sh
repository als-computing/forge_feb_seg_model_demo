#!/bin/bash
# ============================================
# 4-GPU SAM3 Inference (NERSC)
# ============================================

echo "Starting SAM3 multi-GPU inference..."

# ---- Activate environment ----
source ~/.bashrc
conda activate sam3_env

# ---- NCCL / CUDA settings ----
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ---- Paths ----
INPUT_DIR="/global/cfs/cdirs/als/synaps/rec20211222_125057_petiole4"
OUT_DIR="/pscratch/sd/e/elavarpa/sam3_vid_data"

BPE_PATH="/pscratch/sd/e/elavarpa/sam3_checkpoints/sam3_checkpoints/sam3_models_data/bpe_simple_vocab_16e6.txt.gz"
SAM3_PT="/pscratch/sd/e/elavarpa/sam3_checkpoints/sam3_checkpoints/sam3_models_data/sam3-002.pt"
FT_CKPT="/pscratch/sd/e/elavarpa/sam3_checkpoints/sam3_checkpoints/sam3_models_data/checkpoint-001.pt"

# ---- Prompts ----
PROMPTS=(
  "cortex"
  "Phloem Fibers"
  "Xylem vessels"
  "Pith cells"
  "outer cells"
)

# ---- Run ----
torchrun --nproc_per_node=4 run_sam3_4gpu.py \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_DIR" \
  --bpe_path "$BPE_PATH" \
  --sam3_pt "$SAM3_PT" \
  --finetune_ckpt "$FT_CKPT" \
  --confidence 0.5 \
  --amp \
  --prompts "${PROMPTS[@]}"

echo "Inference finished."
