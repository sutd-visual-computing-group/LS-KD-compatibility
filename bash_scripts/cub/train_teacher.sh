# =================== CUB LS experiments =======================


# Step 1: Install requirements. (Comment out if you run inside Docker container)
pip3 install -r requirements.txt


# Step 2: Check nvidia and nvcc versions
nvidia-smi
nvcc --version


# Step 3: Copy the above tar files to the following point
INPUT_FOLDER=datasets/


# Step 4: Create output directory
OUTPUT_PARENT=results/
OUTPUT_FOLDER=$OUTPUT_PARENT/cub/


# Step 5 : Workspace directory (I assume the code submitted is placed here)
WORKSPACE_DIR=/mnt/workspace/projects/ls-kd-compatibility/


# Step 6: Define hyper-parameters
NUM_GPUS=1
BATCH_SIZE=256
WORKERS=8
ALPHAS=(0.1 0.1)
MODELS=(resnet50)
WEIGHTS_PARENT_PATH=pretrained_models/ImageNet-1K/
USE_AMP=1 # Set 1 if need to use AMP, 0 if need to use FP32
WANDB_PROJECT_NAME=test-icml


# >> From this step, python code will run. 
cd $WORKSPACE_DIR/
mkdir -p $OUTPUT_FOLDER

for MODEL in ${MODELS[*]}; do
    for ALPHA in ${ALPHAS[*]}; do
        echo $MODEL/$ALPHA
        python src/image_classification/cub/train_teacher.py $INPUT_FOLDER --output_dir=$OUTPUT_FOLDER --gpus=$NUM_GPUS --alpha="$ALPHA" -a=$MODEL -b=$BATCH_SIZE \
            -j=$WORKERS --dist-url 'tcp://127.0.0.1:8080' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
            --pretrained --imagenet_weights_path=$WEIGHTS_PARENT_PATH/teacher-models/teacher\=resnet50-$ALPHA-best.pth.tar --use_amp=$USE_AMP \
            --exp_name=$WANDB_PROJECT_NAME --epochs=90
    done
done