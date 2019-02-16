import os
import pickle

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR
from config import num_workers, pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.samples = data

        if split == 'train':
            self.transformer = data_transforms['train']

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        filename = os.path.join(IMG_DIR, filename)
        img = cv.imread(filename)
        label = sample['label']

        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        np.random.shuffle(self.samples)


if __name__ == "__main__":
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)

    print(len(train_dataset))
    print(len(train_loader))
