# InsightFace

![apm](https://img.shields.io/apm/l/vim-mode.svg)

PyTorch implementation of Additive Angular Margin Loss for Deep Face Recognition.
[paper](https://arxiv.org/pdf/1801.07698.pdf).
```
@article{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
journal={arXiv:1801.07698},
year={2018}
}
```
## Dataset

Function|Dataset|
|---|---|
|Train|MS-Celeb-1M|
|Test-1|LFW|
|Test-2|MegaFace|

### Introduction

MS-Celeb-1M dataset for training, 3,804,846 faces over 85,164 identities.


## Dependencies
- Python 3.6.8
- PyTorch 1.3.0

## Usage

### Data wrangling
Extract images, scan them, to get bounding boxes and landmarks:
```bash
$ python extract.py
$ python pre_process.py
```

Image alignment:
1. Face detection(MTCNN).
2. Face alignment(similar transformation).
3. Central face selection.
4. Resize -> 112x112. 

Original | Aligned & Resized | Original | Aligned & Resized |
|---|---|---|---|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_img.jpg)|

### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

## Performance evaluation

### LFW

#### Introduction
Use Labeled Faces in the Wild (LFW) dataset for performance evaluation:

- 13233 faces
- 5749 identities
- 1680 identities with >=2 photo

#### Download
Download LFW database put it under data folder:
```bash
$ wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
$ wget http://vis-www.cs.umass.edu/lfw/pairs.txt
$ wget http://vis-www.cs.umass.edu/lfw/people.txt
```

#### Start evaluation
```bash
$ python lfw_eval.py
```

#### Results
Backbones|LFW(%)|Inference speed(*)| 
|---|---|---|
|SE-LResNet101E-IR|99.83%|46.63 ms|
|SE-LResNet50E-IR|99.75%|27.30 ms|
|SE-LResNet18E-IR|99.65%|17.53 ms|

Note(*): with 1 Nvidia Tesla P100.

#### theta j Distribution

![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/theta_dist.png)

#### Error analysis
See also [LFW Face Database Errata](http://vis-www.cs.umass.edu/lfw/index.html#errata)

##### False Positive
2 false positives:

1|2|1|2|
|---|---|---|---|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fp_1.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fp_1.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fp_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fp_1_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fp_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fp_1_aligned.jpg)|


##### False Negative
8 false negative:

1|2|1|2|
|---|---|---|---|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/0_fn_1_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/1_fn_1_aligned.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/2_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/2_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/3_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/3_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/2_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/2_fn_1_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/3_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/3_fn_1_aligned.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/4_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/4_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/5_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/5_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/4_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/4_fn_1_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/5_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/5_fn_1_aligned.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/6_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/6_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/7_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/7_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/6_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/6_fn_1_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/7_fn_0_aligned.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/7_fn_1_aligned.jpg)|


### MegaFace
 
#### Introduction
 
MegaFace dataset includes 1,027,060 faces, 690,572 identities. [Link](http://megaface.cs.washington.edu/)
 
Challenge 1 is taken to test our model with 1 million distractors. 

![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/megaface_stats.png)
 
#### Download

1. Download MegaFace and FaceScrub Images
2. Download Linux DevKit from [MagaFace WebSite](http://megaface.cs.washington.edu/) then extract to megaface folder:

```bash
$ tar -vxf linux-devkit.tar.gz
```
 
#### Generate features

1. Crop MegaFace.
2. Generate features for FaceScrub and MegaFace.
3. Remove noises. 
Note: we used the noises list proposed by InsightFace, at https://github.com/deepinsight/insightface.

```bash
$ python3 megaface.py --action crop_megaface

$ find megaface/facescrub_images -name "*.bin" -type f -delete
$ find megaface/MegaFace_aligned/FlickrFinal2 -name "*.bin" -type f -delete

$ python3 megaface.py --action gen_features
```

#### Evaluation

Start MegaFace evaluation through devkit: 

```bash
$ cd megaface/devkit/experiments
$ python run_experiment.py -p /dev/code/mnt/InsightFace-v2/megaface/devkit/templatelists/facescrub_uncropped_features_list.json /dev/code/mnt/InsightFace-v2/megaface/MegaFace_aligned/FlickrFinal2 /dev/code/mnt/InsightFace-v2/megaface/facescrub_images _0.bin results -s 1000000
```

#### Results

##### Curves

Draw curves with matlab script @ megaface/draw_curve.m. 

CMC|ROC|
|---|---|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/megaface_cmc.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/megaface_roc.jpg)|
|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/megaface_cmc_2.jpg)|![image](https://github.com/foamliu/InsightFace-v2/raw/master/images/megaface_roc_2.jpg)|

##### Textual results
<pre>
Done matching! Score matrix size: 3379 972313
Saving to results/otherFiles/facescrub_megaface_0_1000000_1.bin
Computing test results with 1000000 images for set 1
Loaded 3379 probes spanning 80 classes
Loading from results/otherFiles/facescrub_facescrub_0.bin
Probe score matrix size: 3379 3379
distractor score matrix size: 3379 972313
Done loading. Time to compute some stats!
Finding top distractors!
Done sorting distractor scores
Making gallery!
Done Making Gallery!
Allocating ranks (972393)

Rank 1: 0.964733
</pre>

