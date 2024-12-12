import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from sklearn.model_selection import StratifiedShuffleSplit


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
patience = 10 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transforms.Compose([
        transforms.ToTensor() # Will apply augmentations later for train
    ])
)

targets = np.array([y for _, y in dataset.samples])

sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)

# Train transforms
train_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=5, 
        translate=(0.05, 0.05),
        scale=(0.9, 1.0)
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
        self.indices = self.subset.indices

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = CustomDatasetWrapper(train_subset, transform=train_transforms)
val_dataset = CustomDatasetWrapper(val_subset, transform=val_transforms)

train_labels = [label for _, label in (train_dataset[i] for i in range(len(train_dataset)))]
class_sample_counts = np.bincount(train_labels)
print("Original training distribution:", class_sample_counts)

class_weights = 1. / class_sample_counts
sample_weights = [class_weights[label] for label in train_labels]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset)*2, replacement=True)

# Check distribution after oversampling by simulation
sim_samples = 1000
sim_indices = list(WeightedRandomSampler(sample_weights, num_samples=sim_samples, replacement=True))
sim_labels = [train_labels[i] for i in sim_indices]
sim_counts = np.bincount(sim_labels)
print("Simulated balanced distribution (approx.):", sim_counts)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
val_counts = np.bincount(val_labels)
print("Validation distribution:", val_counts)
print("Val class proportions:", val_counts / val_counts.sum())
print(len(train_loader))
print(len(val_loader))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ResNetCustom(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(ResNetCustom, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # layers: list containing number of blocks in each layer
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)  
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) 
        return x

def get_resnet(depth, num_classes=4):
    total_blocks = (depth - 2) // 2

    # Distribute blocks evenly across 4 layers
    layers = [total_blocks//4]*4
    remainder = total_blocks % 4
    for i in range(remainder):
        layers[i] += 1

    model = ResNetCustom(BasicBlock, layers, num_classes=num_classes)
    return model

def train_model(depth, train_loader, val_loader, device, num_epochs=20, lr=1e-3, num_classes=4, patience=10):
    model = get_resnet(depth, num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_train_acc = 0.0
    epochs_no_improve = 0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}")
        model.train()
        running_corrects_train = 0
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
            running_corrects_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        epoch_train_acc = running_corrects_train / total_train

        # Validation
        model.eval()
        running_val_corrects = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                running_val_corrects += torch.sum(preds == labels).item()
                total_val += labels.size(0)

        epoch_val_acc = running_val_corrects / total_val

        print(f"Val: {epoch_val_acc}")

        train_acc_history.append(epoch_train_acc)
        val_acc_history.append(epoch_val_acc)

        improved = False
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            improved = True
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch_train_acc > best_train_acc:
            best_train_acc = epoch_train_acc

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} for depth {depth}, no improvement for {patience} epochs.")
            break

    plt.figure(figsize=(8,6))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Val Accuracy')
    plt.title(f'Train/Val Accuracy for Model Depth={depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"accuracy_plot_depth_{depth}.png")
    plt.close()

    return best_train_acc, best_val_acc

depths = [1, 2, 5, 10, 15, 20, 30]
best_train_accuracies = []
best_val_accuracies = []

for depth in depths:
    print(f"Training model with depth={depth}...")
    best_train_acc, best_val_acc = train_model(depth, train_loader, val_loader, device, 
                                               num_epochs=num_epochs, lr=learning_rate, 
                                               num_classes=num_classes, patience=patience)
    best_train_accuracies.append(best_train_acc)
    best_val_accuracies.append(best_val_acc)
    print(f"Depth={depth}: Best Train Acc={best_train_acc:.4f}, Best Val Acc={best_val_acc:.4f}")

plt.figure(figsize=(8,6))
plt.plot(depths, best_train_accuracies, marker='o', label='Best Train Accuracy')
plt.plot(depths, best_val_accuracies, marker='s', label='Best Validation Accuracy')
plt.title('Best Accuracy vs. Model Depth')
plt.xlabel('Model Depth')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig("fitting_graph.png") 
print("Plot saved as fitting_graph.png")
