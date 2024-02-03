# 24HiDER

# Hi-DER
This is the official PyTorch implementation of “[Classification models for osteoarthritis grades of multiple joints based on continual learning]()”

We developed Hierarchical Dynamically Expandable Representation (Hi-DER), a hierarchical osteoarthritis (OA) classification model that can be continuously updated to OA classification of multiple joints, by using a continual learning strategy. Please refer to our paper for more details.

## Table of Contents
- [Introduction](#introduction)
- [Proposed Architecture](#proposed-architecture)
- [Pre-requisites](#pre-requisites)
- [Hyperparameters](#hyperparameters)
- [Run Experiment](#run-experiment)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Introduction
While osteoarthritis (OA) affects joints with different morphologies and diverse pathological changes, most previous studies are designed for single joint classification with limited generalizability. To date, there has been no study that developed a multiple joint hierarchical OA classification model with continuously expandable classification capabilities.![image](https://github.com/DigitalHealthcareLab/24HiDER/assets/61937818/511f2a84-63a7-43fb-b61f-56fa760b5872)


## Proposed Architecture
![Figure_2](https://github.com/DigitalHealthcareLab/24HiDER/assets/61937818/5226e695-c283-4010-a38f-9c7f9f4d83a4)

## Pre-requisites
Run the following commands to clone this repository and install the required packages.
```bash
git clone https://github.com/DigitalHealthcareLab/24HiDER.git
pip install -r requirements.txt
```

## Hyperparameters
- **memory_size**: The total number of preserved exemplar in the incremental learning process.
- **memory_per_class**: The number of preserved exemplar per class ($\frac{memory-size}{K-classes}$).
- **shuffle**: Whether to shuffle the class order or not.
- **init_cls**: The number of classes in the initial incremental step.
- **increment**: The number of classes in each incremental step $t$ ($t$ > 1).

## Run Experiment
- Edit the [hider.json](./exps/hider.json) file for global settings.
- Set hyperparameters for training in the [hider.py](./models/hider.py) file.
- Set data path in the [data.py](./utils/data.py) file.
- Edit the structure of hierarchical labels in the [base.py](./models/base.py), [hider.py](./models/hider.py), and [hierarchical_loss.py](./utils/hierarchical_loss.py) file.
- Run the following command to run the experiment.
```bash
python main.py --config=./exps/hider.json
```

## Acknowledgement
Our code is based on [PyCIL](https://github.com/G-U-N/PyCIL). We thank the authors for providing the great base code.

## Citation
If you find this code useful, please consider citing our paper.
