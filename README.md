This repository contains the code for our COLING 2020 paper:

Xinhong Chen, Qing Li, Jianping Wang. A Unified Sequence Labeling Model for Emotion Cause Pair Extraction. COLING 2020.

Please cite our paper if you use this code.

## Dependencies

- **Python 2** (tested on python 2.7.15)
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.8.0

## Usage

Step1: Divide folds
if you prefer our dataset devision, then please run the following commands, after which you should see 20 files with "fold#_train.txt" and "fold#_test.txt", where "#" should be 1 to 10.
- cd data_rand
- python divide_fold

Step2: Set up the parameters for the programs you want to run and run the programs
- Choose what program you want to run. For example, to run our models:
- cd ours-ECPE
- Change the dataset paths and the parameters as you want
- python main_cnn.py
