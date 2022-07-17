# =================== ImageNet LS experiments =======================

# Step 1: Install requirements. (Comment out if you run inside Docker container)
pip3 install -r requirements.txt

# Step 2: Check nvidia and nvcc versions
nvidia-smi
nvcc --version


# Step 3: Copy the above tar files to the following point
INPUT_F3LDER=/mnt/data/imagenet/


# Step 4: Create train and val folders
INPUT_FOLDER_TRAIN=$INPUT_FOLDER/train/ # Downloaded zip file should be 138GB
INPUT_FOLDER_VAL=$INPUT_FOLDER/val/ # # Downloaded zip file should be 6.7GB


# Step 5: Create output directory. I will store the output in this directory. 
OUTPUT_FOLDER=results/imagenet/


# Step 6 : Workspace directory (Place the code here)
WORKSPACE_DIR=/mnt/workspace/projects/ls-kd-compatibility/


# Step 7: Define hyper-parameters
NUM_GPUS=8
BATCH_SIZE=256 # per GPU
WORKERS=64
ALPHAS=(0.0 0.1) # mixture parameter $\alpha$ in LS
MODELS=(resnet50)
WANDB_PROJECT_NAME=test-icml

# >> From this step, python code will run. 
cd $WORKSPACE_DIR/
mkdir -p $OUTPUT_FOLDER

for MODEL in ${MODELS[*]}; do
    for ALPHA in ${ALPHAS[*]}; do
        echo $MODEL/$ALPHA
        python src/image_classification/imagenet/train_teacher.py $INPUT_FOLDER --output_dir=$OUTPUT_FOLDER --gpus=$NUM_GPUS \
        --alpha=$ALPHA -a=$MODEL -b=$BATCH_SIZE --exp_name=$WANDB_PROJECT_NAME\
        -j=$WORKERS --dist-url 'tcp://127.0.0.1:8080' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
    done
done