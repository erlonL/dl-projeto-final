import torch
import torch.nn as nn
import torchvision

class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone VGG16
        self.backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
        
        # Camadas auxiliares
        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        ])
        
        # Classificação e localização
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * 6, kernel_size=3, padding=1),
            nn.Conv2d(512, 4 * 6, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 6, kernel_size=3, padding=1)
        ])
        
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, num_classes * 6, kernel_size=3, padding=1),
            nn.Conv2d(512, num_classes * 6, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes * 6, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        features = []
        loc = []
        conf = []
        
        # Backbone
        for i in range(23):
            x = self.backbone[i](x)
        features.append(x)
        
        # Extras
        for i, layer in enumerate(self.extras):
            x = layer(x)
            features.append(x)
            
        # Detecção
        for i, feat in enumerate(features):
            loc.append(self.loc_layers[i](feat).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf_layers[i](feat).permute(0, 2, 3, 1).contiguous())
            
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        return loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
