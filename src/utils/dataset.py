import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class BrazilianTrafficLightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.annotations = []
        
        # Carregar imagens e anotações
        for img_name in os.listdir(os.path.join(root_dir, 'images')):
            img_path = os.path.join(root_dir, 'images', img_name)
            ann_path = os.path.join(root_dir, 'annotations', 
                                  img_name.replace('.jpg', '.txt'))
            
            if os.path.exists(ann_path):
                self.images.append(img_path)
                self.annotations.append(ann_path)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carregar anotações
        boxes = []
        labels = []
        with open(self.annotations[idx], 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                boxes.append([x, y, w, h])
                labels.append(class_id)
                
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, {'boxes': boxes, 'labels': labels}
