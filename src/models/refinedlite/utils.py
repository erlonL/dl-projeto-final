import torch

def generate_anchors(feature_maps, input_size=320):
    """Gera anchors para cada mapa de características"""
    anchors = []
    scales = [0.1, 0.2, 0.4]
    
    for i, size in enumerate(feature_maps):
        grid_size = input_size / size
        
        for y in range(size):
            for x in range(size):
                cx = (x + 0.5) * grid_size / input_size
                cy = (y + 0.5) * grid_size / input_size
                
                for scale in scales:
                    w = h = scale
                    anchors.append([cx, cy, w, h])
                    
    return torch.tensor(anchors)

def decode_predictions(cls_preds, reg_preds, anchors, conf_thresh=0.5, nms_thresh=0.45):
    """Decodifica as predições da rede em boxes finais"""
    batch_size = cls_preds.size(0)
    num_classes = cls_preds.size(-1)
    
    # Aplicar softmax às predições de classe
    scores = F.softmax(cls_preds, dim=-1)
    
    # Decodificar as coordenadas das boxes
    boxes = decode_boxes(reg_preds, anchors)
    
    detections = []
    for i in range(batch_size):
        # NMS por classe
        det_per_img = []
        for c in range(1, num_classes):  # ignorar background
            mask = scores[i, :, c] > conf_thresh
            if mask.sum() == 0:
                continue
                
            boxes_c = boxes[i][mask]
            scores_c = scores[i, mask, c]
            
            # Aplicar NMS
            keep = nms(boxes_c, scores_c, nms_thresh)
            
            boxes_c = boxes_c[keep]
            scores_c = scores_c[keep]
            labels_c = torch.full_like(scores_c, c)
            
            det_per_img.append(torch.cat([boxes_c, scores_c.unsqueeze(-1), 
                                        labels_c.unsqueeze(-1)], dim=-1))
        
        if len(det_per_img) > 0:
            detections.append(torch.cat(det_per_img, dim=0))
        else:
            detections.append(torch.zeros((0, 6)))
            
    return detections