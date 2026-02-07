#!/bin/bash
#SBATCH -q regular
#SBATCH -A m4880_g
#SBATCH -N 16                          # 4 nodes = 16 GPUs total
#SBATCH -C gpu
#SBATCH --time=00:20:00               # Reduce to 1 hour (500 images takes ~10 min)
#SBATCH --ntasks-per-node=1           
#SBATCH --gpus-per-node=4             
#SBATCH --cpus-per-task=128           
#SBATCH --output=logs/job_%j.out      # %j = job ID
#SBATCH --error=logs/job_%j.err

# Get master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Set your parameters
export OUTPUT_DIR="./results/rec20211222_125057_petiole4_$(date +%Y%m%d_%H%M%S)"
export BATCH_SIZE=16
export INPUT_DIR="./sample_data/rec20211222_125057_petiole4"

# Create output and log directories
mkdir -p ${OUTPUT_DIR}
mkdir -p logs

# Load your conda environment
module load conda
conda activate /pscratch/sd/x/xchong/envs/sam3

echo "============================================================"
echo "JOB STARTED: $(date)"
echo "============================================================"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $((SLURM_NNODES * 4))"

# Count actual images
NUM_IMAGES=$(ls ${INPUT_DIR}/*.tif* 2>/dev/null | wc -l)
echo "Images to process: ${NUM_IMAGES}"
echo "============================================================"

# Record start time
START_TIME=$(date +%s)

# Run inference (no nsys for production)
srun --ntasks-per-node=1 --gpus-per-task=4 bash -c "
torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=\${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    src/inference_v3.py \
    --input-dir ${INPUT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --patch-size 640 \
    --batch-size ${BATCH_SIZE} \
    --confidence 0.5 \
    --prompts 'Cortex' 'Phloem Fibers' 'Air-based Pith cells' 'Water-based Pith cells' 'Xylem vessels' \
    --bpe-path '/pscratch/sd/x/xchong/sam3_finetune/sam3/assets/bpe_simple_vocab_16e6.txt.gz' \
    --original-checkpoint '/pscratch/sd/x/xchong/sam3_finetune/sam3/.cache/huggingface/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt' \
    --finetuned-checkpoint '/pscratch/sd/x/xchong/sam3_finetune/sam3/sam3_logs_plants_with_seg_patches_cleaned_640_0_20tiff_20/checkpoints/checkpoint.pt' \
"

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))
TIME_PER_IMAGE=$(echo "scale=3; $DURATION / $NUM_IMAGES" | bc)
THROUGHPUT=$(echo "scale=2; $NUM_IMAGES / $DURATION * 60" | bc)

echo ""
echo "============================================================"
echo "JOB COMPLETED: $(date)"
echo "============================================================"
echo "Total time: ${MINUTES}m ${SECONDS}s (${DURATION}s)"
echo "Images processed: ${NUM_IMAGES}"
echo "Time per image: ${TIME_PER_IMAGE}s"
echo "Throughput: ${THROUGHPUT} images/minute"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"