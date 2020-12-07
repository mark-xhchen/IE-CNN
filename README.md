This repository contains the code for our COLING 2020 paper\[[pdf](https://www.aclweb.org/anthology/2020.coling-main.18.pdf)\]:

Xinhong Chen, Qing Li, Jianping Wang. A Unified Sequence Labeling Model for Emotion Cause Pair Extraction. In Proceedings of the 28th International Conference on Computational Linguistics, page 208-218.

Please cite our paper if you use this code.

## Dependencies

- **Python 2** (tested on python 2.7.15)
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.8.0

## Usage

Step1: Divide folds
- if you prefer our dataset devision, then please run the following commands, after which you should see 20 files with "fold#_train.txt" and "fold#_test.txt", where "#" should be 1 to 10.
- cd data_rand
- python divide_fold

Step2: Set up the parameters for the programs you want to run and run the programs
- Choose what program you want to run. For example, to run our models:
- cd ours-ECPE
- Change the dataset paths and the parameters as you want
- python main_cnn.py

Note that if you directly download the repo zip file from the github site, **the downloaded "w2v_200.txt" in directory "data_rand" may not be the correct file.** Please:
- open the "w2v_200.txt" file in github;
- right click on the website;
- choose "save as" to download the correct file, which should be around 80Mb.

If you are cloning the whole repo, the above issue should not be a problem.
