## LiteDVDNet

LiteDVDNet, a fine-tuned version of FastDVDNet model that achieves the optimal balance between model size, processing speed, and denoising performance.

### Overview

This repository contains a PyTorch implementation of the LiteDVDNet video denoising algorithm. 
It includes various model modifications trained in accordance with the original article.

### Datasets 

Before running tests or training a model please download datasets <b>DAVIS_2017</b> (training) and 
<b>Set8</b> (validation and tests) by using following
[link](https://drive.google.com/file/d/1a809w-YIpt7ksO0eKuauqDFZVmkYbnVo/view?usp=drive_link). 
After downloading just unpack this archive to the repository root.

### Training

To train a model you can run following script:

```
run_training.py
```
Example of training options:
```
./train_options/lite_dvd_train.yml
```

### Testing

To compare models you acn run following script:

```
run_training.py
```
Example of testing options:
```
./test_options/7_1_cached_inference.yaml
```


### Acknowledgements

This repository is based on the original FastDVDNet research of Matias Tassano 
https://github.com/m-tassano/fastdvdnet


