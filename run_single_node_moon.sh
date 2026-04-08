# Common configuration
module load conda


# Define base paths with timestamp
#INPUT_DIR="/pscratch/sd/x/xchong/3RSE/832/harry/Icy_Regolith_Training_Data/LHS_SparseIce_-162_Tue_1510_8bit_CBC"
INPUT_DIR="/pscratch/sd/x/xchong/3RSE/832/harry/Icy_Regolith_Training_Data/MGS_SparseIce_-158_Mon_2310_8bit_CBC"
#RESULTS_BASE="./results/moon_LHS_envtest_$(date +%Y%m%d_%H%M%S)"
RESULTS_BASE="./results/moon_MGS_envtest_$(date +%Y%m%d_%H%M%S)"
DINO_RESULTS="${RESULTS_BASE}/dino"

DINO_FINETUNED_CHECKPOINT="/pscratch/sd/x/xchong/3RSE/832/harry/dragonfly_seg_lightly/out/vits16-eomt-cityscapes/icy_regolith_LHS_MGS/checkpoints/best.ckpt"



conda activate dino_demo
# DINO inference
torchrun --nproc_per_node=4 --nnodes=1 -m src.inference_dino_v2 \
    --input-dir "$INPUT_DIR" \
    --output-dir "$DINO_RESULTS" \
    --finetuned-checkpoint "$DINO_FINETUNED_CHECKPOINT" \
    --save-overlay \
    --project moon


