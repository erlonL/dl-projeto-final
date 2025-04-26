import torch
from torch.utils.data import Dataset
import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image

class BrazilianTrafficLightDataset(Dataset):
    def __init__(self, images_dir, pascal_dir, transform=None):
        self.images_dir = images_dir
        self.pascal_dir = pascal_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.images)
    
    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        # Obter dimensões originais da imagem
        size = root.find('size')
        orig_width = float(size.find('width').text)
        orig_height = float(size.find('height').text)
        
        for obj in root.findall('./object'):
            labels.append(1)  # semáforo = 1, background = 0
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / orig_width
            ymin = float(bbox.find('ymin').text) / orig_height
            xmax = float(bbox.find('xmax').text) / orig_width
            ymax = float(bbox.find('ymax').text) / orig_height
            
            # Garantir que as coordenadas estejam no intervalo [0, 1]
            xmin = max(0, min(1, xmin))
            ymin = max(0, min(1, ymin))
            xmax = max(0, min(1, xmax))
            ymax = max(0, min(1, ymax))
            
            boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes, labels
    
    def __getitem__(self, idx):
        # Carregar imagem
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.pascal_dir, img_name.replace('.jpg', '.xml'))
        
        # Ler imagem usando PIL
        image = Image.open(img_path).convert('RGB')
        
        # Obter boxes e labels originais
        boxes, labels = self.parse_voc_xml(ann_path)
        
        # Converter para tensores
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Aplicar transformações se houver
        if self.transform is not None:
            # Converter imagem para tensor
            image = self.transform(image)
        
        return image, {'boxes': boxes, 'labels': labels}
    
    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        images = torch.stack(images, 0)
        return images, targets