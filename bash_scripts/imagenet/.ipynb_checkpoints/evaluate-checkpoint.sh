# =================== ImageNet LS experiments =======================

# Step 1: Check nvidia and nvcc versions
nvidia-smi
nvcc --version


# Step 2: Copy the above tar files to the following point
INPUT_FOLDER=/datasets/
#INPUT_FOLDER=/mnt/data/imagenet/


# Step 3: Create train and val folders
INPUT_FOLDER_TRAIN=$INPUT_FOLDER/train/ # Downloaded zip file should be 138GB
INPUT_FOLDER_VAL=$INPUT_FOLDER/val/ # # Downloaded zip file should be 6.7GB


# Step 4: Create output directory. I will store the output in this directory. 
OUTPUT_FOLDER=results/imagenet/


# Step 5 : Workspace directory (Place the code here)
WORKSPACE_DIR=/workspace/
#WORKSPACE_DIR=/mnt/workspace/projects/ls-kd-compatibility/


# Step 7: Define hyper-parameters
NUM_GPUS=-1
BATCH_SIZE=256 # per GPU
WORKERS=1
ALPHAS=(0.1) # mixture parameter $\alpha$ in LS
TEMPERATURES=(1.0)
MODELS=(resnet50)
WANDB_PROJECT_NAME=test-icml
CHECKPOINT_TEACHER=pretrained_models/imagenet/teacher-models/teacher=resnet50-0.1-best.pth.tar
# CHECKPOINT_STUDENT=pretrained_models/imagenet/student-models/resnet18/alpha\=0.0/imagenet-student\=resnet18-teacher=resnet50\(0.0\)-T\=1.0-best.pth.tar
#CHECKPOINT_STUDENT=pretrained_models/imagenet/student-models

# >> From this step, python code will run. 
cd $WORKSPACE_DIR/
mkdir -p $OUTPUT_FOLDER


# for MODEL in ${MODELS[*]}; do
#     for ALPHA in ${ALPHAS[*]}; do
#         for T in ${TEMPERATURES[*]}; do
#             python src/image_classification/imagenet/official_evaluation.py $INPUT_FOLDER --gpus=$NUM_GPUS \
#             -a=$MODEL -b=$BATCH_SIZE --resume $CHECKPOINT_STUDENT/$MODEL/alpha\=$ALPHA/imagenet-student\=$MODEL-teacher=resnet50\($ALPHA\)-T\=$T-best.pth.tar \
#             -j=$WORKERS --dist-url 'tcp://127.0.0.1:8000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --evaluate --seed=2022
#         done
#     done
# done


for MODEL in ${MODELS[*]}; do
    for ALPHA in ${ALPHAS[*]}; do
        python src/image_classification/imagenet/official_evaluation.py $INPUT_FOLDER --gpus=$NUM_GPUS \
        -a=$MODEL -b=$BATCH_SIZE --resume $CHECKPOINT_TEACHER \
        -j=$WORKERS --dist-url 'tcp://127.0.0.1:8000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --evaluate --seed=2022
    done
done