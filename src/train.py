import torch
import torch.optim as optim
from models.ssd import SSD
from utils.dataset import BrazilianTrafficLightDataset
from torch.utils.data import DataLoader

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hiperparâmetros
    num_classes = 5  # Ajuste baseado nas classes de sinais brasileiros
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 100
    
    # Modelo
    model = SSD(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Dataset
    train_dataset = BrazilianTrafficLightDataset('data/train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward
            loc, conf = model(images)
            
            # Calcular perda
            loss = criterion(loc, conf, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
                
        # Salvar checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')

if __name__ == '__main__':
    train()
