# =================== ImageNet KD experiments =======================

# Step 1: Check nvidia and nvcc versions
nvidia-smi
nvcc --version

# Step 2: Install additional requirements
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
pip install wandb==0.12.1
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ../

# Step 3: Copy the above tar files to the following point
INPUT_FOLDER=/mnt/data/imagenet/

# Step 4: Create train and val folders
INPUT_FOLDER_TRAIN=$INPUT_FOLDER/train/ # Downloaded zip file should be 138GB
INPUT_FOLDER_VAL=$INPUT_FOLDER/val/ # # Downloaded zip file should be 6.7GB


# Step 5: Create output directory. I will store the output in this directory. 
OUTPUT_FOLDER=results/imagenet/


# Step 6 : Workspace directory (Place the code here)
WORKSPACE_DIR=/mnt/workspace/projects/ls-kd-compatibility/


# Step 7: Define hyper-parameters
NUM_GPUS=8
BATCH_SIZE=256 # This should be NUM_GPUS * 256
WORKERS=64
TEMPARATURES=(1.0 2.0 3.0 64.0)
MODELS=(resnet18 resnet50)
TEACHER_ALPHAS=(0.0 0.1)
WEIGHTS_PARENT_PATH=pretrained_models/imagenet/teacher-models/
EPOCHS=200
WANDB_PROJECT_NAME=test-icml


# >> From this step, python code will run. 
cd $WORKSPACE_DIR/
mkdir -p $OUTPUT_FOLDER

for MODEL in ${MODELS[*]}; do
    for T in ${TEMPARATURES[*]}; do
        for TEACHER_ALPHA in ${TEACHER_ALPHAS[*]}; do
            echo $MODEL/$ALPHA
            python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS src/image_classification/imagenet/train_student_dali.py -a $MODEL --dali_cpu \
                --b $BATCH_SIZE --loss-scale 128.0 --workers $WORKERS --lr=0.1 --opt-level O2 $INPUT_FOLDER \
                --temperature=$T --teacher_alpha=$TEACHER_ALPHA \
                --teacher_weights_path=$WEIGHTS_PARENT_PATH/teacher\=resnet50-$TEACHER_ALPHA-best.pth.tar \
                --exp_name=$WANDB_PROJECT_NAME --epochs=$EPOCHS --output_dir=$OUTPUT_FOLDER
        done
    done
done