import pickle
import random

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import pickle_file, batch_size, num_workers
from utils import align_face

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

        samples = data['samples']

        num_samples = len(samples)
        num_train = num_samples

        if split == 'train':
            self.samples = samples[:num_train]
            self.transformer = data_transforms['train']

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['full_path']
        landmarks = sample['landmarks']

        try:
            img = align_face(full_path, landmarks)
        except Exception:
            print('full_path: ' + full_path)
            raise

        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        class_id = sample['class_id']
        return img, class_id

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        np.random.shuffle(self.samples)


def show_align():
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = random.sample(data['samples'], 10)

    for i, sample in enumerate(samples):
        full_path = sample['full_path']
        landmarks = sample['landmarks']
        raw = cv.imread(full_path)
        raw = cv.resize(raw, (224, 224))
        img = align_face(full_path, landmarks)
        filename = 'images/{}_raw.jpg'.format(i)
        cv.imwrite(filename, raw)
        filename = 'images/{}_img.jpg'.format(i)
        cv.imwrite(filename, img)


if __name__ == "__main__":
    train_dataset = ArcFaceDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)

    print(batch_size)
    print(len(train_dataset))
    print(len(train_loader))
