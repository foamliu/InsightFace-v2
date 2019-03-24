import os
import pickle
import random
from io import BytesIO

import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR
from config import pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train-enhanced': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, args, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.args = args
        self.samples = data

        if split == 'train':
            if not self.args.enhanced_aug:
                self.transformer = data_transforms['train']
            else:
                self.transformer = data_transforms['train-enhanced']

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        label = sample['label']

        filename = os.path.join(IMG_DIR, filename)
        img = cv.imread(filename)  # BGR
        img = img[..., ::-1]  # RGB
        img = Image.fromarray(img, 'RGB')  # RGB
        if self.args.enhanced_aug:
            img = self.compress_aug(img)  # RGB
        img = self.transformer(img)  # RGB

        return img, label

    def __len__(self):
        return len(self.samples)

    def compress_aug(self, img):
        buf = BytesIO()
        q = random.randint(2, 20)
        img.save(buf, format='JPEG', quality=q)
        buf = buf.getvalue()
        img = Image.open(BytesIO(buf))
        return img
