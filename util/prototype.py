import torch
import torch.nn.functional as F

class PrototypeBank:
    def __init__(self, num_classes=2, momentum=0.999, feat_dim=256, device='cuda'):
        self.num_classes = num_classes
        self.momentum = momentum
        self.feat_dim = feat_dim
        self.device = device
        
        self.prototypes = torch.randn(num_classes, feat_dim).to(device)
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        self.initialized = torch.zeros(num_classes).bool().to(device)

    def update(self, features, masks):
        """
        features: [B, C, h, w] (Low Res Feature Map)
        masks: [B, H, W] (High Res Pseudo Labels)
        """
        mask_small = F.interpolate(masks.float().unsqueeze(1), size=features.shape[2:], mode='nearest').squeeze(1).long()
        
        features = features.permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        mask_small = mask_small.flatten()

        for c in range(self.num_classes):
            class_mask = (mask_small == c)
            if class_mask.sum() == 0: 
                continue 
            
            current_centroid = features[class_mask].mean(0)
            current_centroid = F.normalize(current_centroid, dim=0)
            
            if not self.initialized[c]:
                self.prototypes[c] = current_centroid
                self.initialized[c] = True
            else:
                self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * current_centroid
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)

    def get_consistency_weight(self, features, pseudo_labels):
        """
        Returns [B, H, W] weights. 1.0 = Perfect match to prototype, 0.0 = Bad match.
        """
        if not self.initialized.all():
            return torch.ones_like(pseudo_labels).float()

        H, W = pseudo_labels.shape[1], pseudo_labels.shape[2]

        features = F.normalize(features, dim=1)
        
        features_perm = features.permute(0, 2, 3, 1) 
        sim_map = torch.matmul(features_perm, self.prototypes.T) 
        
        sim_map = sim_map.permute(0, 3, 1, 2)

        sim_map_large = F.interpolate(sim_map, size=(H, W), mode='bilinear', align_corners=True)
        
        weight_map = torch.gather(sim_map_large, 1, pseudo_labels.unsqueeze(1)).squeeze(1)
        
        return torch.relu(weight_map).detach()
