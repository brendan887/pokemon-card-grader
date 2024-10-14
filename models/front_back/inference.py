# inference.py
import torch
from torchvision import transforms
from torchvision.io import read_image
from model import CustomResNet18
import cfg
import os
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """
    Custom dataset to load images from a list of image paths.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image

def load_model(model_path, num_classes):
    model = CustomResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

def predict_images(model, image_paths, class_names, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for images in data_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predicted_classes = [class_names[p] for p in preds]
            all_preds.extend(predicted_classes)

    return all_preds

def main():
    model_path = os.path.join('model_dir', 'model_weights.pth')
    model = load_model(model_path, num_classes=len(cfg.CLASS_NAMES))
    
    # Example usage with multiple images
    image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg', 'path_to_image3.jpg']
    predicted_classes = predict_images(model, image_paths, cfg.CLASS_NAMES, cfg.INFERENCE_BATCH_SIZE)
    for image_path, predicted_class in zip(image_paths, predicted_classes):
        print(f'Image: {image_path}, Predicted class: {predicted_class}')

if __name__ == '__main__':
    main()
