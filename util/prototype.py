import torch
import torch.nn.functional as F

class PrototypeBank:
    def __init__(self, num_classes=2, feat_dim=256, queue_size=1000, device='cuda'):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.queue_size = queue_size
        self.device = device
        
        self.queue = torch.randn(num_classes, queue_size, feat_dim).to(device)
        self.queue = F.normalize(self.queue, dim=2)
        
        self.ptr = torch.zeros(num_classes, dtype=torch.long).to(device)
        
        self.full = torch.zeros(num_classes, dtype=torch.bool).to(device)

    def update(self, features, masks):
        """
        features: [B, C, h, w] -> Reshaped/Selected features
        masks: [B, H, W] -> High Res Pseudo Labels
        """
        mask_small = F.interpolate(masks.float().unsqueeze(1), size=features.shape[2:], mode='nearest').squeeze(1).long()
        
        features = features.permute(0, 2, 3, 1).reshape(-1, self.feat_dim)
        mask_small = mask_small.flatten()

        for c in range(self.num_classes):
            class_mask = (mask_small == c)
            if class_mask.sum() == 0: 
                continue 
            
            valid_feats = features[class_mask]
            
            if valid_feats.shape[0] > 100:
                idx = torch.randperm(valid_feats.shape[0])[:100]
                valid_feats = valid_feats[idx]
            
            valid_feats = F.normalize(valid_feats, dim=1)
            
            batch_size = valid_feats.shape[0]
            ptr = self.ptr[c]
            
            if ptr + batch_size >= self.queue_size:
                space_left = self.queue_size - ptr
                self.queue[c, ptr:] = valid_feats[:space_left]
                
                remaining = batch_size - space_left
                self.queue[c, :remaining] = valid_feats[space_left:]
                self.ptr[c] = remaining
                self.full[c] = True
            else:
                self.queue[c, ptr:ptr+batch_size] = valid_feats
                self.ptr[c] += batch_size

    def get_consistency_weight(self, features, pseudo_labels):
        if not self.full.all():
            return torch.ones_like(pseudo_labels).float()

        B, C, h, w = features.shape
        H, W = pseudo_labels.shape[1], pseudo_labels.shape[2]
        
        features = F.normalize(features, dim=1)
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, self.feat_dim) # [N, 256]

        flat_queue = self.queue.view(-1, self.feat_dim).t() # [256, 2*K]
        
        sim_matrix = torch.matmul(features_flat, flat_queue)
        
        sim_matrix = sim_matrix.view(-1, self.num_classes, self.queue_size)
        
        max_sim, _ = sim_matrix.max(dim=2) # [N, 2]
        
        max_sim_map = max_sim.view(B, h, w, self.num_classes)
        
        max_sim_map = max_sim_map.permute(0, 3, 1, 2)
        max_sim_large = F.interpolate(max_sim_map, size=(H, W), mode='bilinear', align_corners=True)
        
        weight_map = torch.gather(max_sim_large, 1, pseudo_labels.unsqueeze(1)).squeeze(1)
        
        return torch.relu(weight_map).detach()