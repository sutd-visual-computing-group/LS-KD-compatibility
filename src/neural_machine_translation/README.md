# Machine Translation Experiments

## Running the Code
1. We follow the experiment setups of [[2]](#2) to perform English &rarr; German and English &rarr; Russian translation. 
Detailed instructions on reproducibility can be found at [[1]](#1)
2. If you encounter any errors during  intallation, please refer to [[1]](#1) for debugging.

## Hyper-parameters
For all reported machine translation experiments, we use the hyper-parameters from [[1]](#1)

## Results (BLEU scores)

| Transformer &rarr; Transformer KD | English &rarr; German |                |                |
| ----------------------------- | ----------------- | -------------- | -------------- |
|                               | $\\alpha$ / T     | $\\alpha=$ 0.0 | $\\alpha=$ 0.1 |
| Teacher : Transformer         | \-                | 26.461         | 26.75          |
| Student : Transfomer          | T = 1             | 24.914         | 25.085         |
| Student : Transfomer          | T = 2             | 23.103         | 23.421         |
| Student : Transfomer          | T = 3             | 21.999         | 22.076         |
| Student : Transfomer          | T = 64            | 6.564          | 6.461          |


| Transformer &rarr; Transformer KD | English &rarr; Russian |                |                |
| ----------------------------- | ------------------ | -------------- | -------------- |
|                               | $\\alpha$ / T      | $\\alpha=$ 0.0 | $\\alpha=$ 0.1 |
| Teacher : Transformer         | \-                 | 16.718         | 16.976         |
| Student : Transfomer          | T = 1              | 16.140         | 16.197         |
| Student : Transfomer          | T = 2              | 14.977         | 15.100         |
| Student : Transfomer          | T = 3              | 13.826         | 14.106         |
| Student : Transfomer          | T = 64             | 3.605          | 3.590          |

## References
<a id="1">[1]</a> https://github.com/RayeRen/multilingual-kd-pytorch

<a id="2">[2]</a> https://github.com/facebookresearch/fairseq 






