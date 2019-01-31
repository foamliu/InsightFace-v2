# InsightFace

Reproduce ArcFace [论文](https://arxiv.org/pdf/1801.07698.pdf)

## DataSet

CASIA WebFace 数据集，10,575人物身份，494,414图片。

## Dependencies
- PyTorch 1.0.0

## Usage

### Data pre-processing
Extract images：
```bash
$ python pre_process.py
```

#### Image alignment：
Original | Aligned | Original | Aligned |
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

### Comparason
#|image size|network|use-se|loss func|gamma|batch size|weight decay|s|m|LFW accuracy|
|---|---|---|---|---|---|---|---|---|---|---|
|1|112x112|ResNet-152|True|focal|2.0|128|5e-4|50|0.5|99.38%|
|2|112x112|ResNet-101|True|focal|2.0|256|5e-4|50|0.5|99.27%|
|3|112x112|ResNet-101|False|focal|2.0|256|5e-4|50|0.5|99.23%|

## Performance evaluation

### LFW
Use Labeled Faces in the Wild (LFW) dataset for performance evaluation:

- 13233 faces
- 5749 identities
- 1680 identities with >=2 photo

#### Data preparation
Download LFW database put it under data folder:
```bash
$ wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
$ wget http://vis-www.cs.umass.edu/lfw/pairs.txt
$ wget http://vis-www.cs.umass.edu/lfw/people.txt
```

##### False Positive
1|2|1|2|
|---|---|---|---|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_fp_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_fp_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_fp_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_fp_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_fp_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_fp_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_fp_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_fp_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_fp_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_fp_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_fp_1.jpg)|

##### False Negative
1|2|1|2|
|---|---|---|---|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_fn_1.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_fn_1.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_fn_0.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_fn_1.jpg)|
 