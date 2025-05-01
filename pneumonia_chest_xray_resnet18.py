# chest_xray_resnet18.py 

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
import matplotlib.pyplot as plt

# Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(101010)
np.random.seed(101010)
random.seed(101010)

# Data Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Paths
train_path = 'D:/chest_xray/chest_xray/chest_xray/train'
test_path = 'D:/chest_xray/chest_xray/chest_xray/test'

# Data Loading
train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4)

# Model
resnet18 = models.resnet18(pretrained=True)
for param in resnet18.parameters():
    param.requires_grad = False
resnet18.fc = nn.Linear(resnet18.fc.in_features, 1)
resnet18 = resnet18.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, resnet18.parameters()), lr=1e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scaler = torch.cuda.amp.GradScaler()

# Training
num_epochs = 3
for epoch in range(num_epochs):
    resnet18.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
resnet18.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnet18(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_labels, all_preds)
test_f1_score = f1_score(all_labels, all_preds, average='weighted')
print(f"Test Accuracy: {test_accuracy:.3f}, F1 Score: {test_f1_score:.3f}")

# Inference Utils

def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict_image(model, image_path, transform, device):
    original_image, image_tensor = load_and_preprocess_image(image_path, transform)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probability_pneumonia = torch.sigmoid(output).item()
        probability_normal = 1 - probability_pneumonia
    return original_image, {"NORMAL": probability_normal, "PNEUMONIA": probability_pneumonia}

def visualize_predictions(image, probabilities, class_names, actual_class, predicted_class, image_index):
    fig, axarr = plt.subplots(1, 2, figsize=(12, 6))
    axarr[0].imshow(image)
    axarr[0].axis("off")
    axarr[0].set_title(f"Image {image_index + 1}\nActual: {actual_class}\nPredicted: {predicted_class}", loc="left", bbox=dict(alpha=0.5))
    axarr[1].barh(list(probabilities.keys()), list(probabilities.values()))
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Probabilities")
    axarr[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

def get_random_image_from_folder(folder_path):
    classes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    random_class = random.choice(classes)
    class_folder = os.path.join(folder_path, random_class)
    image_files = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
    random_image_file = random.choice(image_files)
    return os.path.join(class_folder, random_image_file), random_class

# Demo on Random Images
test_folder_path = "D:/chest_xray/chest_xray/test"
class_names = ["NORMAL", "PNEUMONIA"]

for i in range(5):
    try:
        test_image_path, actual_class = get_random_image_from_folder(test_folder_path)
        original_image, probabilities = predict_image(resnet18, test_image_path, transform, device)
        predicted_class = max(probabilities, key=probabilities.get)
        visualize_predictions(original_image, probabilities, class_names, actual_class, predicted_class, i)
        print(f"Image {i + 1}:\n  Actual class: {actual_class}\n  Predicted class: {predicted_class}\n  Class Probabilities: {probabilities}\n{'-' * 30}")
    except Exception as e:
        print(f"Error processing image {i + 1}: {e}")
