# Hi-DER
This is the official PyTorch implementation of “[Classification models for arthropathy grades of multiple joints based on hierarchical continual learning]()” (2024)

We propose Hierarchical Dynamically Expandable Representation (Hi-DER), a hierarchical continual arthropathy classification model that can be continuously updated to arthropathy classification of multiple joints, by using a continual learning strategy. Please refer to our paper for more details.

## Table of Contents
- [Introduction](#introduction)
- [Proposed Architecture](#proposed-architecture)
- [DeLong's Test](#delongs-test)
- [Model Robustness: Addressing Underrepresented Features of Osteoarthritis](#model-robustness-addressing-underrepresented-features-of-osteoarthritis)
- [Pre-requisites](#pre-requisites)
- [Hyperparameters](#hyperparameters)
- [Run Experiment](#run-experiment)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Introduction
Although arthropathy can affect diverse joints with different morphologies and pathological changes, most of the previous studies are designed for single joint classification with limited generalizability. To date, there has been no study that developed a multiple joint arthropathy classification model with continuously expandable classification capabilities. The proposed hierarchical continual classification model can be continuously updated for large-scale studies of multiple joints with various anatomical structures.

## Proposed Architecture
![Figure_2](https://github.com/DigitalHealthcareLab/24HiDER/assets/61937818/5226e695-c283-4010-a38f-9c7f9f4d83a4)

## DeLong's Test
The Hi-DER was developed using knee, elbow, ankle, and shoulder radiographs from Sinchon Severance hospital. To evaluate whether utilizing large public datasets affects the predictive performance of Hi-DER, we conducted DeLong's test comparing the AUCs of the model that employed the pre-trained feature extractor and the original model which was trained solely on internal datasets. The pre-training was done using the Osteoarthritis Initiative dataset. Below, we provide the p-values from DeLong’s test at a significance level of 0.05, showing no significant differences in most cases.

![delongs_for_github](https://github.com/DigitalHealthcareLab/24HiDER/assets/61937818/dd7556a5-b407-4e8f-baa0-8af168b28837)

## Model Robustness: Addressing Underrepresented Features of Osteoarthritis
The model's performance can be influenced by the underrepresentation of rare and mild cases. For example, ankle valgus osteoarthritis is significantly less common than varus osteoarthritis, leading to reduced diagnostic accuracy for such atypical cases. Additionally, early osteoarthritis with subtle minor osteophytes often poses challenges for osteoarthritis detection algorithms. These limitations highlight the need for more diverse and balanced datasets to enhance the model performance and generalizability. Our approach accounts for these challenges by focusing on generalized pathological findings across joint types such as joint space narrowing and osteophyte formation.

![Screenshot 2024-12-21 at 10 04 46 PM](https://github.com/user-attachments/assets/c002c96f-e2ed-49cd-a390-201952fcc786)

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
