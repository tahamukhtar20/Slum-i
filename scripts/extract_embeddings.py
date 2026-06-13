"""
Extract DINOv2-Small CLS embeddings for all images in each city.
Saves per-city NPZ: {'filenames': [...], 'embeddings': (N, 384)}
Output: exp_dev/filter/embeddings/<city>_embeddings.npz
"""
import os
from pathlib import Path
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.backbone.dinov2 import vit_small

ROOT      = Path(os.environ.get('SLUM_I_ROOT', Path(__file__).resolve().parents[1]))
DATA_ROOT = ROOT / 'data_pipeline' / 'datasets'
OUT_DIR   = f'{ROOT}/exp_dev/filter/embeddings'
os.makedirs(OUT_DIR, exist_ok=True)

CITIES = ['eldaein', 'elgeneina', 'karachi', 'lahore', 'lower', 'makoko', 'medellin', 'mumbai']
PATCH_SIZE = 14
IMG_SIZE   = 518  # must be divisible by patch_size (518 = 37*14)
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageFolder(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return transform(img), idx

# Load DINOv2-Small
print('Loading DINOv2-Small...')
model = vit_small(patch_size=PATCH_SIZE, img_size=IMG_SIZE, num_register_tokens=0)
state = torch.load(f'{ROOT}/pretrained/dinov2_small.pth', map_location='cpu')
# handle wrapped state dicts
if 'model' in state:
    state = state['model']
elif 'teacher' in state:
    state = {k.replace('backbone.', ''): v for k, v in state['teacher'].items() if 'backbone.' in k}
model.load_state_dict(state, strict=False)
model.eval().cuda()
print('Model loaded.')

for city in CITIES:
    out_path = f'{OUT_DIR}/{city}_embeddings.npz'
    if os.path.exists(out_path):
        print(f'[{city}] already exists, skipping.')
        continue

    img_paths = sorted(glob.glob(f'{DATA_ROOT}/{city}/Images/*.png'))
    print(f'[{city}] {len(img_paths)} images...')

    loader = DataLoader(ImageFolder(img_paths), batch_size=BATCH_SIZE,
                        num_workers=4, pin_memory=True, shuffle=False)

    all_embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.cuda()
            feats = model.forward_features(imgs)
            cls = feats['x_norm_clstoken']          # (B, 384)
            cls = F.normalize(cls, dim=-1)
            all_embs.append(cls.cpu().float().numpy())

    embeddings = np.concatenate(all_embs, axis=0)
    filenames  = [os.path.basename(p) for p in img_paths]

    np.savez(out_path, filenames=filenames, embeddings=embeddings)
    print(f'[{city}] saved {embeddings.shape} -> {out_path}')

print('Done.')
