import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

class TransferBlockMobile(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Primeiro ajustamos o número de canais com uma conv 1x1
        self.adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Depois aplicamos a depthwise separable convolution
        self.conv = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, 
                     groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            # Pointwise conv
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.adjust(x)
        return self.conv(x)

class RefineDetLite(nn.Module):
    def __init__(self, num_classes=2, input_size=320):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone - MobileNetV2
        backbone = mobilenet_v2(pretrained=True)
        self.features = backbone.features
        
        # Mapeamento dos índices e canais das features
        self.feature_indices = [6, 13, 17]  # Layers que queremos extrair
        self.out_channels = [32, 96, 320]   # Número de canais de saída dessas layers
        
        # Transfer Connection Blocks (TCB)
        self.tcb_layers = nn.ModuleList([
            TransferBlockMobile(320, 256),  # C5 -> 256
            TransferBlockMobile(96, 256),   # C4 -> 256
            TransferBlockMobile(32, 256)    # C3 -> 256
        ])
        
        # Detecção
        self.num_anchors = 6
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors * self.num_classes, 3, padding=1)
            for _ in range(3)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Conv2d(256, self.num_anchors * 4, 3, padding=1)
            for _ in range(3)
        ])
    
    def forward(self, x):
        # Feature extraction
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
                print(f"Feature shape {i}: {x.shape}")  # Debug print
        
        # Refino de features com TCB
        refined_features = []
        prev = None
        for i, feature in enumerate(reversed(features)):  # Processamos do fim para o início
            curr = self.tcb_layers[i](feature)
            print(f"After TCB {i}: {curr.shape}")  # Debug print
            
            if prev is not None:
                # Redimensionar prev para o tamanho atual
                prev = F.interpolate(prev, size=curr.shape[-2:], 
                                   mode='bilinear', align_corners=True)
                curr = curr + prev
            
            refined_features.insert(0, curr)  # Inserimos no início
            prev = curr
        
        # Predições
        cls_preds = []
        reg_preds = []
        
        for i, feature in enumerate(refined_features):
            cls_pred = self.cls_heads[i](feature)
            reg_pred = self.reg_heads[i](feature)
            
            # Reshape
            batch_size = cls_pred.size(0)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_pred = cls_pred.view(batch_size, -1, self.num_classes)
            
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous()
            reg_pred = reg_pred.view(batch_size, -1, 4)
            
            cls_preds.append(cls_pred)
            reg_preds.append(reg_pred)
        
        return torch.cat(cls_preds, dim=1), torch.cat(reg_preds, dim=1)