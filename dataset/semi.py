from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_geo_index(root, pool_ids):
    """(row, col) -> filename for tiles in the training pool, from mapcv's patches.geojson."""
    import json
    feats = json.load(open(os.path.join(root, 'patches.geojson')))['features']
    pool = set(pool_ids)
    index, pos = {}, {}
    for f in feats:
        p = f['properties']
        if p['filename'] in pool:
            index[(p['row'], p['col'])] = p['filename']
            pos[p['filename']] = (p['row'], p['col'])
    return {'index': index, 'pos': pos}


def rot90(img, mask, k):
    ops = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
    if k % 4:
        img, mask = img.transpose(ops[k % 4]), mask.transpose(ops[k % 4])
    return img, mask


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None,
                 d4=False, geo=None, geo_margin=256, canvas_size=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        # d4: random 90-degree rotation on top of the horizontal flip (all 8 dihedral poses).
        self.d4 = d4
        # geo: index of training-pool tiles; train_u then also returns a context canvas
        # stitched from neighbouring tiles (never val/test tiles) for the teacher.
        self.geo = geo
        self.geo_margin = geo_margin
        self.canvas_size = canvas_size or (2 * size if size else None)
        # Multispectral (Sentinel-2) dataset: Images/<stem>.npy (bands x 64 x 64 uint16, 0 = no data),
        # per-band train mean/std in s2_meta.json; windows are resized to MS_INPUT (or size).
        meta = os.path.join(root, 's2_meta.json')
        self.ms = os.path.exists(meta)
        if self.ms:
            import json
            m = json.load(open(meta))
            self.ms_mean = torch.tensor(m['mean'], dtype=torch.float32).view(-1, 1, 1)
            self.ms_std = torch.tensor(m['std'], dtype=torch.float32).view(-1, 1, 1)

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'val':
            if id_path is not None:
                val_path = id_path
            else:
                # Prefer root-relative splits (mapcv layout); fall back to repo-level
                val_path = os.path.join(root, 'splits', 'val.txt')
                if not os.path.exists(val_path):
                    val_path = 'splits/%s/val.txt' % name
            with open(val_path, 'r') as f:
                self.ids = f.read().splitlines()
        else:
            if id_path is not None:
                test_path = id_path
            else:
                test_path = os.path.join(root, 'splits', 'test.txt')
                if not os.path.exists(test_path):
                    test_path = 'splits/%s/test.txt' % name
            with open(test_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        if self.ms:
            return self._ms_item(id)
        img = Image.open(os.path.join(self.root, 'Images', id)).convert('RGB')
        if self.mode == 'train_u':
            # Unlabeled ground truth is never read; only crop padding is ignored (as in UniMatch).
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = np.array(Image.open(os.path.join(self.root, 'Masks', id)))
            # mapcv masks: 0 background, 1 slum, 255 ignore. Legacy 0/255 masks only on request.
            if os.environ.get('SLUMI_LEGACY_MASKS') == '1':
                mask = mask // 255
            mask = Image.fromarray(mask.astype(np.uint8))

        if self.mode == 'val' or self.mode == 'test':
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train_u' and self.geo is not None:
            return self._geo_item(id)

        img, mask = resize(img, mask, (0.8, 1.2))
        img, mask = crop(img, mask, self.size, 254 if self.mode == 'train_u' else 255)
        img, mask = hflip(img, mask, p=0.5)
        if self.d4:
            img, mask = rot90(img, mask, random.randint(0, 3))

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = torch.zeros((mask.size[1], mask.size[0]), dtype=torch.long)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), normalize(img_s1), normalize(img_s2), ignore_mask, cutmix_box1, cutmix_box2

    MS_INPUT = 224

    def _ms_load(self, id):
        x = np.load(os.path.join(self.root, 'Images', os.path.splitext(id)[0] + '.npy')).astype(np.float32)
        x = torch.from_numpy(x)
        nodata = (x == 0).all(0)
        x = (x - self.ms_mean) / self.ms_std
        x[:, nodata] = 0
        return x, nodata

    @staticmethod
    def _ms_strong(x):
        """Band-agnostic strong augmentation (UniMatch's ColorJitter / RandomGrayscale / blur analogues)."""
        import torchvision.transforms.functional as TF
        if random.random() < 0.8:  # per-band contrast and brightness, in normalised units
            c = x.shape[0]
            mean = x.mean((1, 2), keepdim=True)
            x = (x - mean) * torch.empty(c, 1, 1).uniform_(0.5, 1.5) + mean + torch.empty(c, 1, 1).uniform_(-0.5, 0.5)
        if random.random() < 0.2:  # spectral averaging
            x = x.mean(0, keepdim=True).expand_as(x).clone()
        if random.random() < 0.5:  # Gaussian blur
            sigma = random.uniform(0.1, 2.0)
            k = 2 * math.ceil(3 * sigma) + 1
            x = TF.gaussian_blur(x, [k, k], [sigma, sigma])
        return x

    def _ms_item(self, id):
        import torch.nn.functional as F
        img, nodata = self._ms_load(id)
        if self.mode == 'train_u':
            # no ground truth; pixels without imagery (cloud, other-split context) are ignored like padding
            mask = nodata.float() * 254
        else:
            mask = torch.from_numpy(np.array(Image.open(
                os.path.join(self.root, 'Masks', os.path.splitext(id)[0] + '.png')))).float()
        S = self.size or self.MS_INPUT

        def rs(i, m, n):
            return (F.interpolate(i[None], size=(n, n), mode='bilinear', align_corners=False)[0],
                    F.interpolate(m[None, None], size=(n, n), mode='nearest')[0, 0])

        if self.mode == 'val' or self.mode == 'test':
            img, mask = rs(img, mask, S)
            return img, mask.long(), id

        n = int(S * random.uniform(0.8, 1.2))
        img, mask = rs(img, mask, n)
        if n < S:
            p = S - n
            img = F.pad(img, (0, p, 0, p), value=0.0)
            mask = F.pad(mask, (0, p, 0, p), value=254.0 if self.mode == 'train_u' else 255.0)
        y, x = random.randint(0, img.shape[1] - S), random.randint(0, img.shape[2] - S)
        img, mask = img[:, y:y + S, x:x + S], mask[y:y + S, x:x + S]
        if random.random() < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)
        mask = mask.long()

        if self.mode == 'train_l':
            return img, mask

        img_s1, img_s2 = self._ms_strong(img.clone()), self._ms_strong(img.clone())
        cutmix_box1, cutmix_box2 = obtain_cutmix_box(S, p=0.5), obtain_cutmix_box(S, p=0.5)
        ignore_mask = torch.zeros((S, S), dtype=torch.long)
        ignore_mask[mask == 254] = 255
        return img, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def _stitch(self, id):
        """Tile plus a margin of real neighbouring imagery; missing neighbours are marked 254."""
        tile = Image.open(os.path.join(self.root, 'Images', id)).convert('RGB')
        t = tile.size[0]
        m = self.geo_margin
        r0, c0 = self.geo['pos'][id]
        big = Image.new('RGB', (3 * t, 3 * t))
        valid = Image.new('L', (3 * t, 3 * t), 254)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                name = id if (dr, dc) == (0, 0) else self.geo['index'].get((r0 + dr * t, c0 + dc * t))
                if name is None:
                    continue
                im = tile if name == id else Image.open(os.path.join(self.root, 'Images', name)).convert('RGB')
                big.paste(im, ((dc + 1) * t, (dr + 1) * t))
                valid.paste(0, ((dc + 1) * t, (dr + 1) * t, (dc + 2) * t, (dr + 2) * t))
        box = (t - m, t - m, 2 * t + m, 2 * t + m)
        return big.crop(box), valid.crop(box)

    def _geo_item(self, id):
        canvas, valid = self._stitch(id)
        canvas, valid = resize(canvas, valid, (0.8, 1.2))
        canvas, valid = crop(canvas, valid, self.canvas_size, 254)
        canvas, valid = hflip(canvas, valid, p=0.5)
        if self.d4:
            canvas, valid = rot90(canvas, valid, random.randint(0, 3))
        # Student window: a random crop of the canvas, so the same ground lands near its edge
        # as often as near its centre, while the teacher sees the whole canvas.
        y0 = random.randint(0, self.canvas_size - self.size)
        x0 = random.randint(0, self.canvas_size - self.size)
        img = canvas.crop((x0, y0, x0 + self.size, y0 + self.size))
        mask = valid.crop((x0, y0, x0 + self.size, y0 + self.size))

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = torch.zeros((self.size, self.size), dtype=torch.long)
        ignore_mask[torch.from_numpy(np.array(mask)).long() == 254] = 255
        canvas_ignore = torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.long)
        canvas_ignore[torch.from_numpy(np.array(valid)).long() == 254] = 255
        return (normalize(img_w), normalize(img_s1), normalize(img_s2), ignore_mask, cutmix_box1, cutmix_box2,
                normalize(canvas), canvas_ignore, torch.tensor([y0, x0]))

    def __len__(self):
        return len(self.ids)
