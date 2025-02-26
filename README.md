# Extracting Activities of Daily Living from Ambient Sensor

## Overview

This project applies machine learning techniques to classify and segment daily activities using ambient sensor data. The repository includes supervised and unsupervised classification models, data preprocessing scripts, and segmentation methods such as IGTS and Logistic Regression Function.

## Features

- **Data Preprocessing**: Cleaning and preparing sensor data.
- **Segmentation Methods**: IGTS and Logistic Regression for activity segmentation.
- **Supervised Classification**: CNN-based model training and evaluation.
- **Unsupervised Classification**: Clustering-based activity recognition.
- **Model Evaluation**: Accuracy, loss curves, and confusion matrices.

## Installation

To set up the project, clone the repository and install dependencies:

```bash
git clone https://github.com/xinyi-dadada/Master-Project.git
cd Master-Project
pip install -r requirements.txt
```

## Usage

Run the following scripts to preprocess data, train models, and evaluate performance:

```bash
python data_preprocess.py  # Prepare data
python cnn_train.py        # Train CNN model
python cnn_eval.py         # Evaluate model
python classify_unsupervised.py  # Run unsupervised classification
```

## Repository Structure

```
📂 Master-Project
│── 📂 Data Preprocessing
│   │── 📜 data_preprocess.py            # Data preprocessing pipeline
│   │── 📜 part_tasks.py                 # Additional preprocessing tasks
│   │── 📜 folder.sh                     # Shell script for folder setup
│── 📂 Segmentation Methods
│   │── 📜 igts.py                       # IGTS segmentation method
│   │── 📜 result_igts.py                # IGTS segmentation results
│   │── 📜 seg_log.py                    # Logistic Regression segmentation
│── 📂 Supervised Classification
│   │── 📜 cnn.py                        # CNN-based classification
│   │── 📜 cnn_train.py                  # Training script
│   │── 📜 cnn_classification_usage.py   # CNN classification usage
│   │── 📜 cnn_data_prep.py              # Data preparation for CNN
│── 📂 Unsupervised Classification
│   │── 📜 classify_unsupervised.py      # Unsupervised learning model
│── 📂 Model Evaluation
│   │── 📜 cnn_eval.py                   # Model evaluation script
│   │── 📜 cnn_usage.py                  # CNN model usage

```



## Contact

For questions or collaboration, reach out to:

- **Cheng Xinyi**
- Email: [xinyi.cheng00@gmail.com](mailto\:xinyi.cheng00@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/xinyi-cheng-9b0aa0263/)

