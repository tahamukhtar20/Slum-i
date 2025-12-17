import numpy as np
import logging
import os


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def __len__(self):
        return len(self.dataloader)


def entropy_meaning_loss(logits, pseudo_labels, mask_ratio):
    """
    logits: (N, C) - Flattened logits from strong augmentation
    pseudo_labels: (N) - Hard pseudo labels
    mask_ratio: Boolean mask of pixels that passed the threshold
    """
    logits_masked = logits[mask_ratio]
    if logits_masked.shape[0] == 0:
        return torch.tensor(0.0).to(logits.device)
    
    pseudo_labels_masked = pseudo_labels[mask_ratio]
    num_classes = logits.shape[1]

    probs = torch.softmax(logits_masked, dim=1)
    
    targets = torch.zeros_like(probs).scatter_(1, pseudo_labels_masked.unsqueeze(1), 1)
    
    non_target_probs = probs * (1 - targets)
    
    loss = -torch.log(1 - non_target_probs + 1e-7)
    
    return loss.sum(dim=1).mean()


def adaptive_negative_learning_loss(logits_strong, logits_weak, threshold_k=0.75):
    """
    logits_strong: (N, C)
    logits_weak: (N, C)
    threshold_k: Float (0 to 1). Percentage of classes to treat as negative.
                 e.g. 0.75 means the bottom 75% of classes are negative.
    """
    probs_weak = torch.softmax(logits_weak.detach(), dim=1)
    probs_strong = torch.softmax(logits_strong, dim=1)
    
    num_classes = logits_weak.shape[1]
    k = int(num_classes * threshold_k) 
    
    _, negative_indices = torch.topk(probs_weak, k, dim=1, largest=False)
    
    negative_probs_strong = torch.gather(probs_strong, 1, negative_indices)
    
    loss = -torch.log(1 - negative_probs_strong + 1e-7)
    
    return loss.mean()


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
