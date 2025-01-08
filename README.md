<!-- ## A Two-stream Deep-learning Network for Heart Rate Estimation from Facial Image Sequence

## Paper

#### [Wen-Nung Lie], [Dao Q. Le], [Po-Han Huang], [Guan-Hao Fu], [Anh Nguyen Thi Quynh], [and Quynh Nguyen Quang Nhu] , “A Two-Stream Deep-Learning Network for Heart Rate Estimation From Facial Image Sequence”, NeurIPS 2020, Oral Presentation (105 out of 9454 submissions) 

#### Link: <https://ieeexplore.ieee.org/abstract/document/10735090>

## New Pre-Trained Model (Updated March 2023)

Please refer to [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) 

#### Abstract

This article presents a deep-learning-based two-stream network to estimate remote Photoplethysmogram (rPPG) signal and hence derive the heart rate (HR) from an RGB facial video. Our proposed network employs temporal modulation blocks (TMBs) to efficiently extract temporal dependencies and spatial attention blocks on a mean frame to learn spatial features. Our TMBs are composed of two subblocks that can simultaneously learn overall and channelwise spatiotemporal features, which are pivotal for the task. Data augmentation (DA) in training and multiple redundant estimations for noise removal in testing were also designed to make the training more effective and the inference more robust. Experimental results show that the proposed temporal shift-channelwise spatio-temporal network (TS-CST Net) has reached competitive and even superior performances among the state-of-the-art (SOTA) methods on four popular datasets, showcasing our network’s learning capability.

## Citation 

``` bash
@article{lie2024two,
  title={A Two-stream Deep-learning Network for Heart Rate Estimation from Facial Image Sequence},
  author={Lie, Wen-Nung and Le, Dao Q and Huang, Po-Han and Fu, Guan-Hao and Anh, Quynh Nguyen Thi and Nhu, Quynh Nguyen Quang},
  journal={IEEE Sensors Journal},
  year={2024},
  publisher={IEEE}
}
``` -->
## Installation
1. This project is developed using >= python 3.10 on Ubuntu 22.04.3! NVIDIA GPUs are needed. We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment.

```shell
  # 1. Create a conda virtual environment.
  conda create -n MS_rppg python=3.10 -y
  conda activate MS_rppg
  
  # 2. Install PyTorch >= v1.6.0 following [official instruction](https://pytorch.org/). Please adapt the cuda version to yours.
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
  
  # 3. Pull our code.
  git clone our git path
  cd MS_rppg
  
  # 4. Install other packages. This project doesn't have any special or difficult-to-install dependencies.
  pip install -r requirements.txt 
```

## Data preprocessing & build k-fold
2. Before training model, it is necessary to prepare the traing model which datatype it want. Executing [dataset_preprocess.py] transfer the video data to [h5py] data. After bulid h5py file, excuting [built_k_fold,py] generate k-fold trainging and testing dataset 
```shell
  # dataset_preprocess
  cd utils
  python dataset_preprocess.py --root_dir [original_DATASET_PATH] --save_dir [preprocessed_DATASET_PATH]

  # built_k_fold
  python built_k_fold.py --root_dir [preprocessed_DATASET_PATH] --save_dir [saved_ DATASET_PATH]
```
## Training 

`python train.py --exp_name test --exp_name [e.g., test] --data_dir [DATASET_PATH] --temporal [e.g., MMTS_CAN]`

## Inference 

`python code/predict_vitals.py --video_path [VIDEO_PATH]`

The default video sampling rate is 30Hz. 

#### Note

During the inference, the program will generate a sample pre-processed frame. Please ensure it is in portrait orientation. If not, you can comment out line 30 (rotation) in the `inference_preprocess.py`. 


## Contact

Please post your technical questions regarding this repo via Github Issues. 







