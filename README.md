<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Revisiting Label Smoothing & Knowledge Distillation</br>Compatibility: What was Missing?</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://keshik6.github.io/" target="_blank" style="text-decoration: none;">Keshigeyan&nbsp;Chandrasegaran</a>&nbsp;/&nbsp;
    <a href="https://scholar.google.com/citations?hl=en&user=9SE3GYMAAAAJ" target="_blank" style="text-decoration: none;">Ngoc&#8209;Trung&nbsp;Tran</a>&nbsp;/&nbsp;
    <a href="https://scholar.google.com/citations?user=CCutiMUAAAAJ&hl=en" target="_blank" style="text-decoration: none;">Yunqing&nbsp;Zhao</a>&nbsp;/&nbsp;
    <a href="https://sites.google.com/site/mancheung0407/" target="_blank" style="text-decoration: none;">Ngai&#8209;Man&nbsp;Cheung</a></br>
Singapore University of Technology and Design (SUTD)<br/>
<em>ICML&nbsp;2022&nbsp;</br></em>
<a href="https://keshik6.github.io/revisiting-ls-kd-compatibility/" title="Project" target="_blank" rel="nofollow" style="text-decoration: none;">Project</a> |
<a href="https://arxiv.org/abs/2206.14532" title="ICML Paper" target="_blank" rel="nofollow" style="text-decoration: none;">ICML Paper</a> |
<a href="https://drive.google.com/drive/folders/1GwqXRVYBpKGolNh2OLEzWUdOHx2XQ6G2?usp=sharing" title="Pre-trained Models" target="_blank" rel="nofollow" style="text-decoration: none;">Pre-trained Models</a>
</p>


## Abstract

This work investigates the compatibility between label smoothing (LS) and knowledge distillation (KD). Contemporary findings addressing this thesis statement take dichotomous standpoints: Muller et al. (2019); Shen et al. (2021). Critically, there is no effort to understand and resolve these contradictory findings, leaving the primal question — to smooth or not to smooth a teacher network? — unanswered. The main contributions of our work are the discovery, analysis and validation of systematic diffusion as the missing concept which is instrumental in understanding and resolving these contradictory findings. This systematic diffusion essentially curtails the benefits of distilling from an LS-trained teacher, thereby rendering KD at increased temperatures ineffective. Our discovery is comprehensively supported by large-scale experiments, analyses and case studies including image classification, neural machine translation and compact student distillation tasks spanning across multiple datasets and teacher-student architectures. Based on our analysis, <em>we suggest practitioners to use an LS-trained teacher with a low-temperature transfer to achieve high performance students.</em>

> **A rule of thumb for practitioners.** We suggest to use an LS-trained teacher with a low-temperature transfer (i.e. *T* = 1) to render high performance students.


## About the code
This codebase is written in Pytorch. It is clearly documented with bash file execution points exposing all required arguments and hyper-parameters. We also provide Docker container details to run our code. 

:white_check_mark: Pytorch

:white_check_mark: ​NVIDIA DALI

:white_check_mark: ​Multi-GPU / Mixed-Precision training

:white_check_mark: ​DockerFile



## Running the code

**ImageNet-1K LS / KD experiments  :** Clear steps on how to run and reproduce our results for ImageNet-1K LS and KD (Table 2, B.3) are provided in [src/image_classification/README.md](src/image_classification/README.md). We support Multi-GPU training and mixed-precision training. We use NVIDIA DALI library for training student networks.

**Machine Translation experiments  :** Clear steps on how to run and reproduce our results for machine translation LS and KD (Table 5, B.2) are provided in [src/neural_machine_translation/README.md](src/neural_machine_translation/README.md). We use [[1]](#1) following exact procedure as [[2]](#2)

**CUB200-2011 experiments  :** Clear steps on how to run and reproduce our results for fine-grained image classification (CUB200) LS and KD (Table 2, B.1) are provided in [src/image_classification/README.md](src/image_classification/README.md). We support Multi-GPU training and mixed-precision training.

**Compact Student Distillation :** Clear steps on how to run and reproduce our results for Compact Student distillation LS and KD (Table 4, B.3) are provided in [src/image_classification/README.md](src/image_classification/README.md). We support Multi-GPU training and mixed-precision training.

**Penultimate Layer Visualization :** Pseudocode for Penultimate visualization algorithm is provided in [src/visualization/visualization_algorithm.png](src/visualization/visualization_algorithm.png)  Refer [src/visualization/alpha-LS-KD_imagenet_centroids.py](src/visualization/alpha-LS-KD_imagenet_centroids.py) for Penultimate layer visualization code to reproduce all visualizations in the main paper and Supplementary (Figures 1, A.1, A.2). The code is clearly documented.


## ImageNet-1K KD Results

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

Results produced with 20.12-py3 ([Nvidia Pytorch Docker container](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12)) + Pytorch LTS 1.8.2 + CUDA11.1

## Pretrained Models

All pretrained image classification, fine-graind image clasification, neural machine translation and compact student distillation models are available [here](https://drive.google.com/drive/folders/1GwqXRVYBpKGolNh2OLEzWUdOHx2XQ6G2?usp=sharing)




## Citation

```markdown
@InProceedings{pmlr-v162-chandrasegaran22a,
    author    = {Chandrasegaran, Keshigeyan and Tran, Ngoc-Trung and Zhao, Yunqing and Cheung, Ngai-Man},
    title     = {Revisiting Label Smoothing and Knowledge Distillation Compatibility: What was Missing?},
    booktitle = {Proceedings of the 39th International Conference on Machine Learning},
    pages     = {2890-2916},
    year      = {2022},
    editor    = {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
    volume    = {162},
    series    = {Proceedings of Machine Learning Research},
    month     = {17-23 Jul},
    publisher = {PMLR},
}
```



## Acknowledgements

We gratefully acknowledge the following works and libraries:

- Pytorch official ImageNet Training : https://github.com/pytorch/examples/tree/main/imagenet
- DALI ImageNet Training : https://github.com/NVIDIA/DALI/blob/ce25d722bc47b8b4f3633ef008a85535db305789/docs/examples/use_cases/pytorch/resnet50/main.py
- Multilingual NMT with Knowledge Distillation on Fairseq (ICLR'19) : https://github.com/RayeRen/multilingual-kd-pytorch
- FairSeq Library : https://github.com/facebookresearch/fairseq
- Experiment Tracking with Weights and Biases : https://www.wandb.com/

Special thanks to [Lingeng Foo](https://github.com/Lingengfoo) and [Timothy Liu](https://github.com/tlkh) for valuable discussion.

## References

<a id="1">[1]</a> Tan, Xu, et al. "Multilingual Neural Machine Translation with Knowledge Distillation." International Conference on Learning Representations. 2019.

<a id="2">[2]</a> Shen, Zhiqiang, et al. "Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study." International Conference on Learning Representations. 2021.

