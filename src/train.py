import torch
from models.refinedlite.model import RefineDetLite
from models.refinedlite.loss import RefineDetLiteLoss

# Inicializar modelo
model = RefineDetLite(num_classes=4, input_size=320)  # 3 classes + background
criterion = RefineDetLiteLoss(num_classes=4)

# Dados de exemplo
batch_size = 2
x = torch.randn(batch_size, 3, 320, 320)

# Forward pass
cls_preds, reg_preds = model(x)

# Treinar
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()

# Simular targets
cls_targets = torch.randint(0, 4, (batch_size, cls_preds.size(1)))
reg_targets = torch.randn_like(reg_preds)
pos_mask = torch.ones_like(cls_targets, dtype=torch.bool)

loss = criterion((cls_preds, reg_preds), 
                (cls_targets, reg_targets, pos_mask))
loss.backward()
optimizer.step()