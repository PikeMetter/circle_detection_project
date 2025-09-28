import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, coord_weight=10.0):
        super(CombinedLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, pred_heatmap, pred_coords, true_heatmap, true_coords):
        # 热力图损失
        heatmap_loss = self.bce_loss(pred_heatmap, true_heatmap)
        
        # 坐标回归损失
        coord_loss = self.mse_loss(pred_coords, true_coords)
        
        # 组合损失
        total_loss = (self.heatmap_weight * heatmap_loss + 
                     self.coord_weight * coord_loss)
        
        return total_loss, heatmap_loss, coord_loss

class FocalLoss(nn.Module):
    """Focal Loss for heatmap"""
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * \
                   torch.pow(1 - target, self.beta) * neg_mask
        
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if pos_mask.sum() > 0:
            loss = -(pos_loss + neg_loss) / pos_mask.sum()
        else:
            loss = -neg_loss
            
        return loss
