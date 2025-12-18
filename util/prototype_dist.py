import torch
import torch.nn.functional as F

class PrototypeBank:
    def __init__(self, num_classes=2, momentum=0.999, feat_dim=2048):
        self.num_classes = num_classes
        self.momentum = momentum
        self.feat_dim = feat_dim
        self.prototypes = None
        self.initialized = False

    def update(self, features, masks):
        """
        Updates global prototypes.
        features: [B, C, h, w] (Low Res)
        masks: [B, H, W] (High Res)
        """
        if self.prototypes is None:
            self.feat_dim = features.shape[1]
            self.prototypes = torch.zeros(self.num_classes, self.feat_dim).to(features.device)

        # 1. Downsample mask to feature size (Nearest Neighbor)
        mask_small = F.interpolate(masks.float().unsqueeze(1), size=features.shape[2:], mode='nearest').squeeze(1).long()
        
        # 2. Flatten features
        features = features.permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        mask_small = mask_small.flatten()

        # 3. Update Prototypes (EMA)
        for c in range(self.num_classes):
            class_mask = (mask_small == c)
            if class_mask.sum() == 0: continue
            
            current_centroid = features[class_mask].mean(0)
            current_centroid = F.normalize(current_centroid, dim=0)
            
            if not self.initialized:
                self.prototypes[c] = current_centroid
            else:
                self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * current_centroid
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)
        
        self.initialized = True

    def get_consistency_weight(self, features, pseudo_labels):
        """
        Calculates similarity efficiently.
        features: [B, 2048, h, w] -> High dimensional, Low spatial
        pseudo_labels: [B, H, W] -> Low dimensional, High spatial
        """
        if not self.initialized:
            return torch.ones_like(pseudo_labels).float()

        B, C, h, w = features.shape
        H, W = pseudo_labels.shape[1], pseudo_labels.shape[2]

        # 1. Normalize Low-Res Features
        features = F.normalize(features, dim=1) # [B, 2048, h, w]
        
        # 2. Calculate Similarity to ALL classes at Low-Res
        # We use einsum for memory efficiency: 
        # B=Batch, C=Channels, X=Height, Y=Width, K=Class
        # features(b,c,x,y) * protos(k,c) -> sim_map(b,k,x,y)
        sim_map = torch.einsum('bcxy, kc -> bkxy', features, self.prototypes)
        
        # 3. Clamp Negative Correlations
        sim_map = torch.clamp(sim_map, min=0.0) # [B, 2, h, w]

        # 4. Upsample the Similarity Map (2 channels is cheap to resize!)
        # [B, 2, h, w] -> [B, 2, H, W]
        sim_map_large = F.interpolate(sim_map, size=(H, W), mode='bilinear', align_corners=True)
        
        # 5. Select the weight corresponding to the predicted class
        # pseudo_labels: [B, H, W] -> [B, 1, H, W] for gather
        weight_map = torch.gather(sim_map_large, 1, pseudo_labels.unsqueeze(1))
        
        return weight_map.squeeze(1).detach()