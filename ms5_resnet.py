import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import resnet50, ResNet50_Weights

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
data_dir = "segmented_images"
classes = ['psa10_images', 'psa9_images', 'psa8_images', 'psa7_images']
num_classes = len(classes)
batch_size = 32
num_epochs = 20
learning_rate = 1e-3
val_split = 0.1
image_size = (523, 404)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

targets = np.array([y for _, y in dataset.samples])

sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

# Create subsets
train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

train_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=5, 
        translate=(0.05, 0.05),
        # scale=(0.9, 1.0), # slightly scaling down so entire original image is visible
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.ToTensor()
])

class CustomDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.classes = self.subset.dataset.classes
        self.class_to_idx = self.subset.dataset.class_to_idx

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = CustomDatasetWrapper(train_dataset, transform=train_transforms)
val_dataset = CustomDatasetWrapper(val_dataset, transform=val_transforms)


train_labels = [label for _, label in (train_dataset[i] for i in range(len(train_dataset)))]
class_sample_counts = np.bincount(train_labels)
print("Original training distribution:", class_sample_counts)

class_weights = 1. / class_sample_counts
sample_weights = [class_weights[label] for label in train_labels]


sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset)*2, replacement=True)

sim_samples = 1000
sim_indices = list(WeightedRandomSampler(sample_weights, num_samples=sim_samples, replacement=True))
sim_labels = [train_labels[i] for i in sim_indices]
sim_counts = np.bincount(sim_labels)
print("Simulated balanced distribution (approx.):", sim_counts)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Print class distributions in val set
val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
val_counts = np.bincount(val_labels)
print("Validation distribution:", val_counts)
print("Val class proportions:", val_counts / val_counts.sum())
print(len(train_loader))
print(len(val_loader))

model = resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False 

print("Got Resnet50")

class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

in_features = model.fc.in_features
model.fc = CustomHead(in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

best_val_acc = 0.0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Training {epoch}:")
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total_train += labels.size(0)

    epoch_train_loss = running_loss / total_train
    epoch_train_acc = running_corrects / total_train

    # Validation
    model.eval()
    running_val_loss = 0.0
    running_val_corrects = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_val_loss += loss.item() * images.size(0)
            running_val_corrects += torch.sum(preds == labels).item()
            total_val += labels.size(0)

    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = running_val_corrects / total_val

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_acc)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # Save best model
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), "best_model.pth")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=[c.replace('_images', '') for c in classes])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Best Model')
plt.show()
