# Advanced Breath Analysis through Hierarchical Deep Convolutional Neural Network for Multi-Cancer Screening
---

## Contents
- [Overview](#1-overview)
- [Contents](#2-contents)
- [System Requirements](#3-system-requirements)
- [Data avaliability](#4-data-avaliability)
- [Instructions for Use](#5-instructions-for-use)

---

## 1. Overview

This project provides a deep learning framework for multi-cancer screening using gas sensor array data from exhaled breath. A hierarchical deep CNN (HD-CNN) model is used to classify samples from healthy controls (HC), lung cancer (LC), and gastric cancer (GC) patients.

The model employs a two-stage structure:

Coarse classifier: HC vs. cancer

Fine classifier: LC vs. GC

Basic knowledge of Python, PyTorch, and CNNs is required.

---

## 2. Contents

## 2. Contents

- [`dataloader/dataloader.py`](dataloader/dataloader.py)  
  Defines the `CustomDataset` class used to load and preprocess breath sensor data from CSV files.  
  Provides input tensors along with coarse and fine labels for hierarchical classification.

- [`models/model_CNN.py`](models/model_CNN.py)  
  Implements a baseline 1D CNN model for flat 3-class classification (Healthy, Lung Cancer, Gastric Cancer) without hierarchical structure.

- [`models/model_HDCNN.py`](models/model_HDCNN.py)  
  Implements the Hierarchical Deep Convolutional Neural Network (HD-CNN).  
  Contains a two-stage classification pipeline with a coarse classifier (HC vs. CP) and a fine classifier (LC vs. GC),  
  combined through a probabilistic averaging layer.

- [`Train_CNN.py`](Train_CNN.py)  
  Training script for the CNN model.  
  Includes data loading, training loop, validation, and model checkpointing.

- [`Train_HDCNN.py`](Train_HDCNN.py)  
  Training script for the HD-CNN model.  
  Includes data loading, hierarchical training, validation, and checkpointing.

- [`data/`](data/)  
  Directory containing input sensor data and label CSV files.  
  Includes training and validation sets for both flat CNN and hierarchical HD-CNN.

- [`checkpoints/`](checkpoints/)  
  Folder for saving trained model weights, softmax outputs, and predictions at selected training epochs.

---

## 3. System Requirements

---

### Hardware Requirements
- **GPU**: NVIDIA RTX TITAN (24GB VRAM) — used for training and validation in this study  
  > Alternatively, any modern NVIDIA GPU with ≥11GB VRAM is recommended (e.g., RTX 3080, A6000)
- **RAM**: At least 16 GB
- **CPU**: Intel i7 or higher (8 threads or more recommended)
- **Storage**: ≥10 GB free space for datasets and model checkpoints
---

### Package Requirements

- torch: 1.12.1
- numpy: 1.23.5
- pandas: 1.5.3
- tqdm: 4.64.1
- scikit-learn: 1.2.1
- matplotlib: 3.7.0

---

## 4. Data avaliability

All main training results are presented in the main manuscript and Supplementary Information. Due to the inclusion of personally identifiable information, the raw and preprocessed input data are subject to access restrictions and are securely managed by Seoul National University Bundang Hospital. Researchers wishing to access the data must obtain approval from the hospital's Institutional Review Board (IRB) and receive authorization from the corresponding author.

---

## 5. Instructions for Use

## Instructions to Use

This repository provides the implementation of a Hierarchical Deep Convolutional Neural Network (HD-CNN) for breath-based multi-cancer classification using gas sensor time-series data.

### Step 1: Prepare the Dataset

- Organize your data in the following structure:
```Datasets
data/
├── data_training.csv
├── data_training_label_HDCNN.csv
├── data_valid.csv
└── data_valid_label_HDCNN.csv
```
- Each `*.csv` file should contain:
- Sensor data (`19 × 1800` time-series)
- Corresponding labels for coarse (Healthy vs. Cancer) and fine (Lung vs. Gastric) classification.

### Step 2: Set Up the Environment

Install the required packages using `pip` or `conda`:

```bash
pip install torch==1.12.1 numpy==1.23.5 pandas==1.5.3 tqdm==4.64.1 scikit-learn==1.2.1 matplotlib==3.7.0
```

### Step 3: Train the CNN/HD-CNN model

Run the training script:
```bash
python Train_HDCNN.py
```
This traininig script will:
- Load and preprocess the data
- Train the HD-CNN model
- Evaluate on validation set
- Save probability outputs and model checkpoints every 50 epochs (under checkpoints/)

### Step 4: Analyze Results
After training, you can find:
- Softmax probability outputs at checkpoints/val_probs_epochXX.csv
- Model weights saved periodically
- Validation accuracy printed for each epoch
---

