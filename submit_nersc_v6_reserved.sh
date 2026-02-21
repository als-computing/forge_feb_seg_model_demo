#!/bin/bash
#SBATCH -q regular
#SBATCH -A amsc006
#SBATCH --reservation=_CAP_March_ModCon_Dry_Run_GPU
#SBATCH -N 2
#SBATCH -C gpu&hbm80g
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# ============================================================
# MASTER NODE
# ============================================================
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# ============================================================
# PATHS
# ============================================================
INPUT_DIR="/pscratch/sd/x/xchong/sam3_finetune/forge_feb_seg_model_demo/sample_data/0220"
RESULTS_BASE="./results/submit_nersc_7classes_0220all_envtest_$(date +%Y%m%d_%H%M%S)"
#RESULTS_BASE="./results/submit_nersc_7classes_petiole50_small_envtest_20260220_133047"  # Uncomment to resume

SAM3_RESULTS="${RESULTS_BASE}/sam3"
CELLPOSE_RESULTS="${RESULTS_BASE}/cellpose"
DINO_RESULTS="${RESULTS_BASE}/dino"
COMBINED_OUTPUT="${RESULTS_BASE}/combined"

SAM3_ORIGINAL_CHECKPOINT="/pscratch/sd/x/xchong/sam3_finetune/sam3/.cache/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
SAM3_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/sam3_finetune/sam3/sam3_logs_plants_with_seg_patches_cleaned_640_25_42tiff_7classes_20/checkpoints/checkpoint.pt"
CELLPOSE_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/cellpose/models/petiole_model_flow0"
DINO_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/lightly/out_slurm_vitl16/my_experiment_petiole_42tiff/checkpoints/best.ckpt"
BPE_PATH="/pscratch/sd/x/xchong/sam3_finetune/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

# ============================================================
# SETUP
# ============================================================
mkdir -p logs
mkdir -p "${SAM3_RESULTS}" "${CELLPOSE_RESULTS}" "${DINO_RESULTS}"
mkdir -p "${COMBINED_OUTPUT}/cellpose_dino" "${COMBINED_OUTPUT}/sam_dino" "${COMBINED_OUTPUT}/extract_regions"

module load conda

NUM_IMAGES=$(ls ${INPUT_DIR}/*.tif* 2>/dev/null | wc -l)

echo "============================================================"
echo "JOB STARTED: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Total nodes: $SLURM_NNODES  |  Total GPUs: $((SLURM_NNODES * 4))"
echo "Images to process: ${NUM_IMAGES}"
echo "Results base: ${RESULTS_BASE}"
echo "============================================================"

TOTAL_START=$(date +%s)

# ============================================================
# STEP 1: SAM3 Inference
# ============================================================
echo ""
echo ">>> STEP 1/5: SAM3 Inference [$(date)]"
STEP_START=$(date +%s)

conda activate sam3_demo

srun --ntasks-per-node=1 --gpus-per-task=4 bash -c "
torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=\${SLURM_JOB_ID}1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m src.inference_v6 \
    --input-dir '$INPUT_DIR' \
    --output-dir '$SAM3_RESULTS' \
    --patch-size 640 \
    --batch-size 1 \
    --confidence 0.1 \
    --overlap-ratio 0.5 \
    --prompts 'Cortex' 'Phloem Fibers' 'Phloem' 'Hydrated Xylem vessels' 'Air-based Pith cells' 'Water-based Pith cells' 'Dehydrated Xylem vessels' \
    --bpe-path '$BPE_PATH' \
    --finetuned-checkpoint '$SAM3_FINETUNED_CHECKPOINT' \
    --original-checkpoint '$SAM3_ORIGINAL_CHECKPOINT'
"

echo "<<< STEP 1 DONE: $(($(date +%s) - STEP_START))s elapsed"

# # ============================================================
# # STEP 2: Cellpose Inference
# # ============================================================
# echo ""
# echo ">>> STEP 2/5: Cellpose Inference [$(date)]"
# STEP_START=$(date +%s)

# conda activate dino_demo

# srun --ntasks-per-node=1 --gpus-per-task=4 bash -c "
# torchrun \
#     --nnodes=\$SLURM_NNODES \
#     --nproc_per_node=4 \
#     --rdzv_id=\${SLURM_JOB_ID}2 \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     -m src.inference_cellpose_v1 \
#     --input-dir '$INPUT_DIR' \
#     --output-dir '$CELLPOSE_RESULTS' \
#     --finetuned-checkpoint '$CELLPOSE_FINETUNED_CHECKPOINT' \
#     --save-overlay
# "

# echo "<<< STEP 2 DONE: $(($(date +%s) - STEP_START))s elapsed"

# ============================================================
# STEP 3: DINO Inference
# ============================================================
echo ""
echo ">>> STEP 3/5: DINO Inference [$(date)]"
STEP_START=$(date +%s)

srun --ntasks-per-node=1 --gpus-per-task=4 bash -c "
torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=\${SLURM_JOB_ID}3 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m src.inference_dino_v1 \
    --input-dir '$INPUT_DIR' \
    --output-dir '$DINO_RESULTS' \
    --finetuned-checkpoint '$DINO_FINETUNED_CHECKPOINT' \
    --save-overlay
"

echo "<<< STEP 3 DONE: $(($(date +%s) - STEP_START))s elapsed"

# # ============================================================
# # STEP 4: Combine Cellpose + DINO  (single node, multi-CPU)
# # ============================================================
# echo ""
# echo ">>> STEP 4/5: Combine Cellpose + DINO [$(date)]"
# STEP_START=$(date +%s)

# srun --nodes=1 --ntasks=1 --cpus-per-task=128 \
#     python -m src.combine_cellpose_dino \
#     --input-dir "$INPUT_DIR" \
#     --instance-masks-dir "$CELLPOSE_RESULTS/instance_masks" \
#     --semantic-masks-dir "$DINO_RESULTS/semantic_masks" \
#     --output-dir "$COMBINED_OUTPUT/cellpose_dino" \
#     --num-workers 128

# echo "<<< STEP 4 DONE: $(($(date +%s) - STEP_START))s elapsed"

# # ============================================================
# # STEP 5a: Combine SAM3 + DINO  (single node, multi-CPU)
# # ============================================================
# echo ""
# echo ">>> STEP 5a/5: Combine SAM3 + DINO [$(date)]"
# STEP_START=$(date +%s)

# srun --nodes=1 --ntasks=1 --cpus-per-task=128 \
#     python -m src.combine_sam_dino_v2 \
#     --input-dir "$INPUT_DIR" \
#     --instance-masks-dir "$SAM3_RESULTS" \
#     --semantic-masks-dir "$DINO_RESULTS/semantic_masks" \
#     --output-dir "$COMBINED_OUTPUT/sam_dino" \
#     --num-workers 128

# echo "<<< STEP 5a DONE: $(($(date +%s) - STEP_START))s elapsed"

# # ============================================================
# # STEP 5b: Extract Xylem Regions  (single node, multi-CPU)
# # ============================================================
# echo ""
# echo ">>> STEP 5b/5: Extract Xylem Regions [$(date)]"
# STEP_START=$(date +%s)

# srun --nodes=1 --ntasks=1 --cpus-per-task=128 \
#     python -m src.extract_region \
#     --input-dir "$INPUT_DIR" \
#     --semantic-masks-dir "$DINO_RESULTS/semantic_masks" \
#     --instance-masks-dir "$SAM3_RESULTS" \
#     --output-dir "$COMBINED_OUTPUT/extract_regions" \
#     --xylem-mode all \
#     --dilate-px 5 \
#     --num-workers 128

# echo "<<< STEP 5b DONE: $(($(date +%s) - STEP_START))s elapsed"


# ============================================================
# STEP 5: Filter, Combine & Extract  (single node, multi-CPU)
# ============================================================
echo ""
echo ">>> STEP 5/5: Filter + Combine + Extract [$(date)]"
STEP_START=$(date +%s)
srun --nodes=1 --ntasks=1 --cpus-per-task=128 \
    python -m src.combine_sam_dino_v3 \
    --input-dir "$INPUT_DIR" \
    --instance-masks-dir "$SAM3_RESULTS" \
    --semantic-masks-dir "$DINO_RESULTS/semantic_masks" \
    --output-dir "$COMBINED_OUTPUT/sam_dino" \
    --dilate-px 5 \
    --save-extracted \
    --dino-trust Cortex Phloem_Fibers Phloem Air-based_Pith_cells Water-based_Pith_cells
echo "<<< STEP 5 DONE: $(($(date +%s) - STEP_START))s elapsed"


# ============================================================
# SUMMARY
# ============================================================
TOTAL_DURATION=$(($(date +%s) - TOTAL_START))
echo ""
echo "============================================================"
echo "ALL STEPS COMPLETED: $(date)"
echo "Total wall time: $((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s"
echo "Images processed: ${NUM_IMAGES}"
echo "Results saved to: ${RESULTS_BASE}"
echo "============================================================"