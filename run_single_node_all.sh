# Common configuration
module load conda


# Define base paths with timestamp
INPUT_DIR="/pscratch/sd/x/xchong/sam3_finetune/seg_annotation_pipeline2/data4/images"
RESULTS_BASE="./results/data4_envtest_$(date +%Y%m%d_%H%M%S)"
#RESULTS_BASE="./results/data4_envtest_20260215_232400"
SAM3_RESULTS="${RESULTS_BASE}/sam3"
CELLPOSE_RESULTS="${RESULTS_BASE}/cellpose"
DINO_RESULTS="${RESULTS_BASE}/dino"
COMBINED_OUTPUT="${RESULTS_BASE}/combined"

# Define checkpoint paths
SAM3_ORIGINAL_CHECKPOINT="/pscratch/sd/x/xchong/sam3_finetune/sam3/.cache/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
SAM3_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/sam3_finetune/sam3/sam3_logs_plants_with_seg_patches_cleaned_640_25_42tiff_20/checkpoints/checkpoint.pt"
CELLPOSE_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/cellpose/models/petiole_model_flow0"
DINO_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/lightly/out_slurm_vitl16/my_experiment_petiole_42tiff/checkpoints/best.ckpt"
BPE_PATH="/pscratch/sd/x/xchong/sam3_finetune/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


conda activate sam3
# SAM3 inference
torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_v5 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$SAM3_RESULTS" \
    --patch-size 640 \
    --batch-size 1 \
    --confidence 0.1 \
    --overlap-ratio 0.5 \
    --prompts "Cortex" "Phloem" "Phloem Fibers" "Air-based Pith cells" "Water-based Pith cells" "Xylem vessels" \
    --bpe-path "$BPE_PATH" \
    --finetuned-checkpoint "$SAM3_FINETUNED_CHECKPOINT" \
    --original-checkpoint "$SAM3_ORIGINAL_CHECKPOINT"

conda activate dino_demo

# Cellpose inference
torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_cellpose_v1 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$CELLPOSE_RESULTS" \
    --finetuned-checkpoint "$CELLPOSE_FINETUNED_CHECKPOINT" \
    --save-overlay


# DINO inference
torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_dino_v1 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$DINO_RESULTS" \
    --finetuned-checkpoint "$DINO_FINETUNED_CHECKPOINT" \
    --save-overlay

# Combine Cellpose + DINO
python -m src.combine_cellpose_dino \
    --input-dir "$INPUT_DIR" \
    --instance-masks-dir "$CELLPOSE_RESULTS/instance_masks" \
    --semantic-masks-dir "$DINO_RESULTS/semantic_masks" \
    --output-dir "$COMBINED_OUTPUT/cellpose_dino"

# Combine SAM + DINO
python -m src.combine_sam_dino \
    --input-dir "$INPUT_DIR" \
    --instance-masks-dir "$SAM3_RESULTS" \
    --semantic-masks-dir "$DINO_RESULTS/semantic_masks" \
    --output-dir "$COMBINED_OUTPUT/sam_dino"