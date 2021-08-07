## Enriching Local and Global Contexts for Temporal Action Localization



This repo holds the codes and models for the ContextLoc framework presented on ICCV 2021

**Enriching Local and Global Contexts for Temporal Action Localization**
Zixin Zhu, Wei Tang, Le Wang*, Nanning Zheng, Gang Hua,  *ICCV 2021*, Montreal, Canada.

[[Paper]](https://arxiv.org/pdf/2107.12960.pdf)




# Contents
----

* [Usage Guide](#usage-guide)
   * [Prerequisites](#prerequisites)
   * [Code and Data Preparation](#code-and-data-preparation)
      * [Get the code](#get-the-code)
      * [Download Datasets](#download-datasets)
      * [Download Features](#download-features)
   * [Training](#training-ContextLoc)
   * [Testing](#testing-trained-models)

   * [Citation](#citation)
   * [Contact](#contact)


----
# Usage Guide

## Prerequisites


The training and testing in ContextLoc based on PGCN is reimplemented in PyTorch for the ease of use. 

- [PyTorch 1.0.1][pytorch]
                   
Other minor Python modules can be installed by running

```bash
pip install -r requirements.txt
```

 
 
## Code and Data Preparation


### Get the code

Clone this repo with git, **please remember to use --recursive**

```bash
git clone --recursive https://github.com/buxiangzhiren/ContextLoc.git
```

### Download Datasets

We support experimenting with THUMOS14 datasets for temporal action detection.

- THUMOS14: We need the validation videos for training and testing videos for testing. 
You can download them from the [THUMOS14 challenge website][thumos14].
 


### Download Features

Here, we use the I3D features (RGB+Flow) of PGCN for training and testing. 

THUMOS14: You can download it from [Google Cloud][features_google] or [Baidu Cloud][features_baidu].



## Training ContextLoc


Plesse first set the path of features in data/dataset_cfg.yaml

```bash
train_ft_path: $PATH_OF_TRAINING_FEATURES
test_ft_path: $PATH_OF_TESTING_FEATURES
```


Then, you can use the following commands to train ContextLoc

```bash
python ContextLoc_train.py thumos14 --snapshot_pre $PATH_TO_SAVE_MODEL
```

After training, there will be a checkpoint file whose name contains the information about dataset and the number of epoch.
This checkpoint file contains the trained model weights and can be used for testing.

## Testing Trained Models




You can obtain the detection scores by running 

```bash
sh test.sh TRAINING_CHECKPOINT
```

Here, `TRAINING_CHECKPOINT` denotes for the trained model.
This script will report the detection performance in terms of [mean average precision][map] at different IoU thresholds.

Please first change the "result" file of RGB to the "RGB_result" file and change the "result" file of Flow to the ''Flow_result" file.

Then put them in the "results" folder.

You can obtain the two-stream results on THUMOS14 by running
```bash
sh test_two_stream.sh
```


## Citation


Please cite the following paper if you feel ContextLoc useful to your research

```
@inproceedings{Zhu2021EnrichingLA,
  author={Zixin Zhu and Wei Tang and Le Wang and Nanning Zheng and G. Hua},
  title={Enriching Local and Global Contexts for Temporal Action Localization},
  booktitle   = {ICCV},
  year      = {2021},
}
```

## Contact
For any question, please file an issue or contact
```
Zixin Zhu: zhuzixin@stu.xjtu.edu.cn
```



[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[anaconda]:https://www.continuum.io/downloads
[tdd]:https://github.com/wanglimin/TDD
[anet]:https://github.com/yjxiong/anet2016-cuhk
[faq]:https://github.com/yjxiong/temporal-segment-networks/wiki/Frequently-Asked-Questions
[bs_line]:https://github.com/yjxiong/temporal-segment-networks/blob/master/models/ucf101/tsn_bn_inception_flow_train_val.prototxt#L8
[bug]:https://github.com/yjxiong/caffe/commit/c0d200ba0ed004edcfd387163395be7ea309dbc3
[tsn_site]:http://yjxiong.me/others/tsn/
[custom guide]:https://github.com/yjxiong/temporal-segment-networks/wiki/Working-on-custom-datasets.
[thumos14]:http://crcv.ucf.edu/THUMOS14/download.html
[tsn]:https://github.com/yjxiong/temporal-segment-networks
[anet_down]:https://github.com/activitynet/ActivityNet/tree/master/Crawler
[map]:http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf
[action_kinetics]:http://yjxiong.me/others/kinetics_action/
[pytorch]:https://github.com/pytorch/pytorch
[ssn]:http://yjxiong.me/others/ssn/
[untrimmednets]:https://github.com/wanglimin/UntrimmedNet
[emv]:https://github.com/zbwglory/MV-release
[features_google]: https://drive.google.com/open?id=1C6829qlU_vfuiPdJSqHz3qSqqc0SDCr_
[features_baidu]: https://pan.baidu.com/s/1Dqbcm5PKbK-8n0ZT9KzxGA
[features_baidu_anet_flow]: https://pan.baidu.com/s/1irWHfdF8RJCQcy1D10GlfA 
[features_google_anet_rgb]: https://drive.google.com/drive/folders/1UHT3S--vo8MCT8AX3ajHE6TcAThDxFlF?usp=sharing 
