# COMP5318 Assignment 2: Image Classification

This repository contains the Group 131 submission for COMP5318 Assignment 2.
The project compares classical machine learning and neural network approaches for image classification.

## Group

- SID1: 540580169
- SID2: 530637868
- SID3: 540881594

## Project Overview

The notebook implements an image-classification workflow that includes:

- Loading and inspecting the provided image dataset
- Class-distribution analysis and sample visualization
- Image preprocessing, resizing, normalization, and flattening where required
- Random Forest classifier as the classical machine learning baseline
- Fully connected neural network
- Convolutional neural network
- Hyperparameter tuning and final model evaluation

## Final Results

Key final test results from the submitted notebook:

```text
Final Random Forest Accuracy: 0.6616
Final CNN Test Accuracy: 0.7768
```

The CNN achieved the strongest final performance among the implemented models.

## Repository Structure

```text
.
├── COMP5318_assignment2_Zheng_Tian_Ling.ipynb
├── data/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── reports/
│   ├── COMP5318_assignment2_Zheng_Tian_Ling.html
│   ├── COMP5318_assignment2_specification.pdf
│   └── RF_CNN_report.docx
├── requirements.txt
└── README.md
```

## How to Run

1. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Open and run:

   ```text
   COMP5318_assignment2_Zheng_Tian_Ling.ipynb
   ```

The notebook expects the dataset files to be available in the `data/` directory.

## Notes

The original working folder contained duplicate exports, temporary files, checkpoints, and large intermediate PDFs. This repository keeps the final notebook, final reports, assignment specification, and the dataset arrays needed to reproduce the notebook.
