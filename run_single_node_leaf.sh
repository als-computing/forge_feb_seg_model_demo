# Common configuration
module load conda


# Define base paths with timestamp
#INPUT_DIR="/pscratch/sd/x/xchong/3RSE/832/harry/Icy_Regolith_Training_Data/LHS_SparseIce_-162_Tue_1510_8bit_CBC"
INPUT_DIR="/pscratch/sd/x/xchong/3RSE/832/leaf/Walnut leaves/images"
#RESULTS_BASE="./results/moon_LHS_envtest_$(date +%Y%m%d_%H%M%S)"
RESULTS_BASE="./results/Walnut_leaves_envtest_$(date +%Y%m%d_%H%M%S)"
DINO_RESULTS="${RESULTS_BASE}/dino"

DINO_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/3RSE/832/leaf_new/out_vitl16-eomt-cityscapes_5k/leaf_experiment/checkpoints/best.ckpt"



conda activate dino_demo
# DINO inference
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_dino_v4 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$DINO_RESULTS" \
    --finetuned-checkpoint "$DINO_FINETUNED_CHECKPOINT" \
    --save-overlay \
    --project leaf


