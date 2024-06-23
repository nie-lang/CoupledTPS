# <p align="center">Semi-Supervised Coupled Thin-Plate Spline Model for Rotation Correction and Beyond

## Introduction
This is the official implementation for [CoupledTPS](https://arxiv.org/abs/2401.13432) (TPAMI2024).

[Lang Nie](https://nie-lang.github.io/)<sup>1</sup>, [Chunyu Lin](https://faculty.bjtu.edu.cn/8549/)<sup>1</sup>, [Kang Liao](https://kangliao929.github.io/)<sup>2</sup>, [Shuaicheng Liu](http://www.liushuaicheng.org/)<sup>3</sup>, [Yao Zhao](https://faculty.bjtu.edu.cn/5900/)<sup>1</sup>

<sup>1</sup> Beijing Jiaotong University

<sup>2</sup> Nanyang Technological University

<sup>3</sup> University of Electronic Science and Technology of China

> ### Feature
> This paper tries to solve multiple single-image-based warping problems in a unified framework. 
![image](https://github.com/nie-lang/CoupledTPS/blob/main/fig.png)
The above figure shows three examples of our method. The proposed CoupledTPS corrects (a) the 2D in-plane tilt, (b) irregular boundaries, and (c) wide-angle portrait via a unified warping framework.

## Code
### Requirement
* numpy 1.19.5
* pytorch 1.7.1
* scikit-image 0.15.0
* tensorboard 2.9.0

We implement this work with Ubuntu, 3090Ti, and CUDA11. Refer to [environment.yml](https://github.com/nie-lang/CoupledTPS/blob/main/environment.yml) for more details.

### How to run it
* For **Rotation Correction**, please refer to [rotation/readme.md](https://github.com/nie-lang/CoupledTPS/blob/main/rotation/readme.md).
* For **Rectangling**, please refer to [rectangling/readme.md](https://github.com/nie-lang/CoupledTPS/blob/main/rectangling/readme.md).
* For **Portrait Correction**, please refer to [portrait/readme.md](https://github.com/nie-lang/CoupledTPS/blob/main/portrait/readme.md).

### Better Performance
> The latent condition is designed to speed up the inference, but we found it may depress the performance of CoupledTPS. To further unleash the potential of CoupledTPS, one can replace the latent condition with the image (refer to Fig. 3 of our paper), which could further break the performance bottleneck of TPS transformation.

We conduct an experiment on the rotation correction task and show the results below.

With Latent Condition
|  | PSNR| SSIM |
|:-------- |:-----|:-----|
|Inference iter 1  | 22.04|  0.668|
|Inference iter 2  | 22.20|  0.675|
|Inference iter 3  | 22.29|  0.679|
|Inference iter 4  | 22.28|  0.678|

Without Latent Condition
|  | PSNR| SSIM |
|:-------- |:-----|:-----|
|Inference iter 1  | 22.40| 0.680|
|Inference iter 2  | 23.21| 0.714|
|Inference iter 3  | 23.26| 0.716|
|Inference iter 4  | 23.25| 0.716|




## Meta
If you have any questions about this project, please feel free to drop me an email.

NIE Lang -- nielang@bjtu.edu.cn
```
@article{nie2024semi,
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Semi-Supervised Coupled Thin-Plate Spline Model for Rotation Correction and Beyond}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TPAMI.2024.3417024}
}

```


## References
[1] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Deep rectangling for image stitching: a learning baseline. CVPR (Oral), 2022.  
[2] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Deep Rotation Correction without Angle Prior. TIP, 2023.   
[3] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Parallax-Tolerant Unsupervised Deep Image Stitching. ICCV, 2023.   



