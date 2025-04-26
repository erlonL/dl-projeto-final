import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class RefineDetLiteLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, predictions, targets):
        cls_preds, reg_preds = predictions # [B, N, 2], [B, N, 4]
        batch_size = cls_preds.size(0)
        total_loss = torch.tensor(0., device=cls_preds.device)
        
        for i in range(batch_size):
            # Obter alvos da imagem atual
            target_boxes = targets[i]['boxes']    # [M, 4]
            target_labels = targets[i]['labels']  # [M]
            
            if len(target_boxes) == 0:
                continue
            
            # Calcular IoU entre predições e alvos
            ious = self.box_iou(reg_preds[i], target_boxes)  # [N, M]
            max_ious, matched_idx = ious.max(dim=1)  # [N]
            
            # Criar máscaras para positivos e negativos
            pos_mask = max_ious >= 0.5
            neg_mask = max_ious < 0.4
            
            # Se houver matches positivos
            if pos_mask.sum() > 0:
                # Loss de classificação para positivos
                pos_pred_cls = cls_preds[i][pos_mask]  # [P, 2]
                pos_target_labels = target_labels[matched_idx[pos_mask]]  # [P]
                cls_loss = F.cross_entropy(pos_pred_cls, pos_target_labels)
                
                # Loss de regressão para positivos
                pos_pred_reg = reg_preds[i][pos_mask]  # [P, 4]
                pos_target_boxes = target_boxes[matched_idx[pos_mask]]  # [P, 4]
                reg_loss = F.smooth_l1_loss(pos_pred_reg, pos_target_boxes)
                
                # Loss de classificação para negativos (background)
                if neg_mask.sum() > 0:
                    neg_pred_cls = cls_preds[i][neg_mask]  # [N, 2]
                    neg_target = torch.zeros(neg_mask.sum(), 
                                          device=cls_preds.device,
                                          dtype=torch.long)  # [N]
                    cls_loss += 0.5 * F.cross_entropy(neg_pred_cls, neg_target)
                
                total_loss += cls_loss + reg_loss
            
        # Retornar perda média por batch
        return total_loss / batch_size if batch_size > 0 else total_loss
    
    def box_iou(self, boxes1, boxes2):
        """Calcula IoU entre dois conjuntos de boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
        
        # Calcular coordenadas da interseção
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
        
        union = area1[:, None] + area2 - inter  # [N, M]
        
        return inter / (union + 1e-6)  # [N, M]
    
    def focal_loss(self, preds, targets):
        """Implementação da Focal Loss"""
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()