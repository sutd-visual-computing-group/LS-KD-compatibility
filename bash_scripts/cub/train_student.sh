# =================== CUB200-2011 KD experiments =======================


# Step 1: Install requirements. (Comment out if you run inside Docker container)
pip3 install -r requirements.txt


# Step 2: Check nvidia and nvcc versions
nvidia-smi
nvcc --version


# Step 3: Copy the above tar files to the following point
INPUT_FOLDER=datasets/


# Step 4: Create output directory.
OUTPUT_PARENT=results/
OUTPUT_FOLDER=$OUTPUT_PARENT/cub/


# Step 5 : Workspace directory (I assume the code submitted is placed here)
WORKSPACE_DIR=/mnt/workspace/projects/ls-kd-compatibility/


# Step 6: Define hyper-parameters
NUM_GPUS=1
BATCH_SIZE=256
WORKERS=8
TEMPARATURES=(1.0 2.0 3.0 64.0)
MODELS=(resnet18 mobilenet_v2 convnext_tiny)
TEACHER_ALPHAS=(0.0 0.1)
WEIGHTS_PARENT_PATH=release_this/CUB/teacher-models/
USE_AMP=1 # 1 if need to use amp, 0 if need to use fp32
WANDB_PROJECT_NAME=test-icml
SEED=2021

# >> From this step, python code will run. 
cd $WORKSPACE_DIR/
mkdir -p $OUTPUT_FOLDER

for MODEL in ${MODELS[*]}; do
    for T in ${TEMPARATURES[*]}; do
        for TEACHER_ALPHA in ${TEACHER_ALPHAS[*]}; do
            python src/image_classification/cub/train_student.py $INPUT_FOLDER --output_dir=$OUTPUT_FOLDER --gpus=$NUM_GPUS -a=$MODEL -b=$BATCH_SIZE \
                -j=$WORKERS --dist-url 'tcp://127.0.0.1:3081' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
                --teacher_weights_path=$WEIGHTS_PARENT_PATH/cub-teacher\=resnet50-$TEACHER_ALPHA-best.pth.tar \
                --temperature=$T --teacher_alpha=$TEACHER_ALPHA --pretrained --use_amp=$USE_AMP \
                --exp_name=$WANDB_PROJECT_NAME --seed=$SEED
        done
    done
done