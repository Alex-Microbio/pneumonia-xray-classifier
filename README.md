# pneumonia-xray-classifier

# Chest X-Ray Classifier with ResNet-18

This project uses a transfer learning approach with ResNet-18 to classify chest X-ray images as either **Normal** or **Pneumonia**. The model is trained on a public dataset and achieves high accuracy through a simple yet effective deep learning pipeline.

## Dataset

The dataset should be structured as follows:
data/ ├── train/ │ ├── NORMAL/ │ └── PNEUMONIA/ ├── val/ │ ├── NORMAL/ │ └── PNEUMONIA/ └── test/ ├── NORMAL/ └── PNEUMONIA/


Public source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model

- **Backbone:** ResNet-18 (pretrained on ImageNet)
- **Classifier Head:** Modified to 2 output classes (binary)
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Metric:** Accuracy, Confusion Matrix, Classification Report

## How to Run

1. Place your dataset under the `data/` folder (or update the path in the script).
2. Install dependencies:

```bash
pip install -r requirements.txt

