import argparse
import json
import multiprocessing as mp
import os
import struct
from multiprocessing import Pool

import cv2 as cv
import numpy as np
import torch
import tqdm
from torchvision import transforms
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import align_face, get_central_face_attributes


def walkdir(folder, ext, orgkey, newkey):
    """Walk through each files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for filename in [f for f in files if f.lower().endswith(ext)]:
            yield (os.path.abspath(os.path.join(dirpath, filename)), orgkey, newkey)


def crop_one_image(item):
    filepath, orgkey, newkey = item
    new_fn = filepath.replace(orgkey, newkey)
    tardir = os.path.dirname(new_fn)
    if not os.path.isdir(tardir):
        os.makedirs(tardir)

    if not os.path.exists(new_fn):
        is_valid, bounding_boxes, landmarks = get_central_face_attributes(filepath)
        if is_valid:
            img = align_face(filepath, landmarks)
            cv.imwrite(new_fn, img)


def crop(path, orgkey, newkey):
    print('cropping {}...'.format(path))

    # Preprocess the total files count
    filecounter = 0
    for filepath in walkdir(path, '.jpg', orgkey, newkey):
        filecounter += 1

    cpu_count = mp.cpu_count()
    with Pool(cpu_count) as p:
        r = list(tqdm(p.imap(crop_one_image, walkdir(path, '.jpg', orgkey, newkey)), total=filecounter, unit="files"))

    print('{} images were cropped successfully.'.format(len(r)))


def gen_feature(path):
    print('gen features {}...'.format(path))
    # Preprocess the total files count
    filecounter = 0
    for filepath in walkdir(path, '.jpg'):
        filecounter += 1

    checkpoint = torch.load('BEST_checkpoint.tar')
    model = checkpoint['model'].to(device)
    model.eval()
    transformer = data_transforms['val']

    with torch.no_grad():
        for filepath in tqdm(walkdir(path, '.jpg'), total=filecounter, unit="files"):
            tarfile = filepath + '_0.bin'
            if not os.path.exists(tarfile):
                tmp = torch.zeros([1, 3, 112, 112], dtype=torch.float)
                tmp[0] = get_image(cv.imread(filepath, True), transformer)
                feature = model(tmp.to(device))[0].cpu().numpy()
                write_feature(tarfile, feature / np.linalg.norm(feature))


def get_image(img, transformer):
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    return img.to(device)


def read_feature(filename):
    f = open(filename, 'rb')
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4 * 4))
    mat = np.fromstring(f.read(rows * 4), dtype=np.dtype('float32'))
    return mat.reshape(rows, 1)


def write_feature(filename, m):
    header = struct.pack('iiii', m.shape[0], 1, 4, 5)
    f = open(filename, 'wb')
    f.write(header)
    f.write(m.data)


def remove_noise():
    for line in open('megaface/megaface_noises.txt', 'r'):
        filename = 'megaface/MegaFace/FlickrFinal2/' + line.strip() + '_0.bin'
        if os.path.exists(filename):
            print(filename)
            os.remove(filename)

    noise = set()
    for line in open('megaface/facescrub_noises.txt', 'r'):
        noise.add((line.strip().replace('png', 'jpg') + '0.bin').replace('_', '').replace(' ', ''))
    for root, dirs, files in os.walk('megaface/facescrub_images'):
        for f in files:
            if f.replace('_', '').replace(' ', '') in noise:
                filename = os.path.join(root, f)
                if os.path.exists(filename):
                    print(filename)
                    os.remove(filename)


def test():
    root1 = '/root/lin/data/FaceScrub_aligned/Benicio Del Toro'
    root2 = '/root/lin/data/FaceScrub_aligned/Ben Kingsley'
    for f1 in os.listdir(root1):
        for f2 in os.listdir(root2):
            if f1.lower().endswith('.bin') and f2.lower().endswith('.bin'):
                filename1 = os.path.join(root1, f1)
                filename2 = os.path.join(root2, f2)
                fea1 = read_feature(filename1)
                fea2 = read_feature(filename2)
                print(((fea1 - fea2) ** 2).sum() ** 0.5)


def match_result():
    with open('matches_facescrub_megaface_0_1000000_1.json', 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
        for i in range(len(load_dict)):
            print(load_dict[i]['probes'])


def pngtojpg(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1] == '.png':
                img = cv.imread(os.path.join(root, f))
                newfilename = f.replace(".png", ".jpg")
                cv.imwrite(os.path.join(root, newfilename), img)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--action', default='crop_megaface', help='action')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.action == 'crop_megaface':
        crop('megaface/MegaFace/FlickrFinal2', 'MegaFace', 'MegaFace_aligned')
    elif args.action == 'crop_facescrub':
        crop('/newdisk/facescrub_images', 'facescrub', 'facescrub_aligned')
    elif args.action == 'gen_features':
        gen_feature('megaface/facescrub_images')
        gen_feature('megaface/MegaFace/FlickrFinal2')
        remove_noise()

    # match_result()
    #
    # pngtojpg('/newdisk/facescrub_images')
    # crop('/root/lin/data/FaceScrub', 'FaceScrub', 'FaceScrub_aligned')
    # gen_feature('/root/lin/data/FaceScrub_aligned')
    #
    #
