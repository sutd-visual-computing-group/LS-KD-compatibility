# Image Classification Experiments

## About the Code
This codebase is written in Pytorch. It is clearly documented with bash file execution points exposing all required arguments and hyper-parameters. We also provide Docker container details to run our code. 

:white_check_mark: Pytorch

:white_check_mark: ​NVIDIA DALI

:white_check_mark: ​Multi-GPU / Mixed-Precision training

:white_check_mark: ​DockerFile


## Hyper-parameters
### Standard Image Classification (ImageNet-1K) experiments

For all reported ImageNet-1K LS and KD experiments,
- batch size = 256
- adam optimizer (β1=0.9, β2​=0.999).
- initial learning rate = 0.1 decayed by a factor of 10 every 30 epochs
- epochs = 90 for LS experiments following standard ImageNet receipe and 
- epochs = 200 for KD experiments following [[1]](#1)
- Data augmentation for training : Random Resized Crop (224), Random Horizontal Flip
- Data augmentation for testing : Center Crop (224) after resizing the smallest side to 256
- For **Compact student distillation** task, replace the student with compact DNN architectures (i.e.: EfficientNet-B0)

### Fine-grained (CUB200) experiments

For all reported ImageNet-1K LS and KD experiments,
- Transfer Learning from ImageNet-1K supervised classification
- batch size = 256
- adam optimizer (β1=0.9, β2​=0.999).
- initial learning rate = 0.1 decayed by a factor of 10 every 30 epochs
- epochs = 90 for LS experiments following standard ImageNet receipe and 
- epochs = 200 for KD experiments following [[1]](#1)
- Data augmentation for training : Random Resized Crop (224), Random Horizontal Flip, Random Rotation, Random Grayscale, Random Color Jitter
- Data augmentation for testing : Center Crop (224) after resizing the smallest side to 256
- For **Compact student distillation** task, replace the student with compact DNN architectures (i.e.: MobileNet-V2)


## Running the Code

1. Create a new virtual environment and install all the dependencies

   ` pip install -r requirements.txt`

2. Download ImageNet-1K dataset from [here](https://www.image-net.org/download.php). For CUB experiments dataset will be downloaded automatically.

3. To train teacher models,
   - For ImageNet-1K, run `bash bash_scripts/imagenet/train_teacher.sh`
   - For CUB200, run `bash bash_scripts/cub/train_teacher.sh`

4. To train student models,
   - For ImageNet-1K, run `bash bash_scripts/imagenet/train_student.sh`
   - For CUB200, run `bash bash_scripts/cub/train_student.sh` 


## ImageNet-1K KD results

| ResNet-50 &rarr; ResNet-18 KD |               |                 |                 |
| ------------------------- | ------------- | --------------- | --------------- |
|                           | $\\alpha$ / T | $\\alpha=$ 0.0  | $\\alpha=$ 0.1  |
| Teacher : ResNet-50       | \-            | 76.132 / 92.862 | 76.200 / 93.082 |
| Student : ResNet-18       | T = 1         | 71.488 / 90.272 | 71.666 / 90.364 |
| Student : ResNet-18       | T = 2         | 71.360 / 90.362 | 68.860 / 89.352 |
| Student : ResNet-18       | T = 3         | 69.674 / 89.698 | 67.752 / 88.932 |
| Student : ResNet-18       | T = 64        | 66.194 / 88.706 | 64.362 / 87.698 |


| ResNet-50 &rarr; ResNet-50 KD |               |                 |                 |
| ------------------------- | ------------- | --------------- | --------------- |
|                           | $\\alpha$ / T | $\\alpha=$ 0.0  | $\\alpha=$ 0.1  |
| Teacher : ResNet-50       | \-            | 76.132 / 92.862 | 76.200 / 93.082 |
| Student : ResNet-50       | T = 1         | 76.328 / 92.996 | 76.896 / 93.236 |
| Student : ResNet-50       | T = 2         | 76.180 / 93.072 | 76.110 / 93.138 |
| Student : ResNet-50       | T = 3         | 75.488 / 92.670 | 75.790 / 93.006 |
| Student : ResNet-50       | T = 64        | 74.278 / 92.410 | 74.566 / 92.596 |


## CUB200 KD results

| ResNet-50 to ResNet-18 KD |               |                 |                 |
| ------------------------- | ------------- | --------------- | --------------- |
|                           | $\\alpha$ / T | $\\alpha=$ 0.0  | $\\alpha=$ 0.1  |
| Teacher : ResNet-50       | \-            | 81.584 / 95.927 | 82.068 / 96.168 |
| Student : ResNet-18       | T = 1         | 80.169 / 95.392 | 80.946 / 95.312 |
| Student : ResNet-18       | T = 2         | 80.808 / 95.593 | 80.428 / 95.518 |
| Student : ResNet-18       | T = 3         | 80.785 / 95.674 | 78.196 / 95.213 |
| Student : ResNet-18       | T = 64        | 73.611 / 94.529 | 67.161 / 93.062 |

| ResNet-50 to ResNet-50 KD |               |                 |                 |
| ------------------------- | ------------- | --------------- | --------------- |
|                           | $\\alpha$ / T | $\\alpha=$ 0.0  | $\\alpha=$ 0.1  |
| Teacher : ResNet-50       | \-            | 81.584 / 95.927 | 82.068 / 96.168 |
| Student : ResNet-50       | T = 1         | 82.902 / 96.358 | 83.742 / 96.778 |
| Student : ResNet-50       | T = 2         | 82.534 / 96.427 | 83.379 / 96.537 |
| Student : ResNet-50       | T = 3         | 82.091 / 96.243 | 82.142 / 96.427 |
| Student : ResNet-50       | T = 64        | 79.784 / 95.927 | 77.206 / 95.812 |


## Compact Student Distillation KD results

| ResNet-50 to ResNet-50 KD |               |                 |                 |
| ------------------------- | ------------- | --------------- | --------------- |
|                           | $\\alpha$ / T | $\\alpha=$ 0.0  | $\\alpha=$ 0.1  |
| Teacher : ResNet-50       | \-            | 81.584 / 95.927 | 82.068 / 96.168 |
| Student : MobileNetV2     | T = 1         | 81.144 / 95.677 | 81.731 / 95.754 |
| Student : MobileNetV2     | T = 2         | 81.895 / 95.858 | 80.609 / 95.47  |
| Student : MobileNetV2     | T = 3         | 81.257 / 95.677 | 78.961 / 95.306 |
| Student : MobileNetV2     | T = 64        | 75.441 / 94.702 | 70.435 / 93.494 |


## References

<a id="1">[1]</a> Shen, Zhiqiang, et al. "Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study." International Conference on Learning Representations. 2021.
