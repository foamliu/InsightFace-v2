import math
import os
import pickle
import random
import tarfile

import cv2 as cv
import numpy as np
import scipy.stats
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from config import device
from data_gen import data_transforms
from utils import align_face, get_face_all_attributes, draw_bboxes

angles_file = 'data/angles.txt'
lfw_pickle = 'data/lfw_funneled.pkl'


def extract(filename):
    with tarfile.open(filename, 'r') as tar:
        tar.extractall('data')


def process():
    subjects = [d for d in os.listdir('data/lfw_funneled') if os.path.isdir(os.path.join('data/lfw_funneled', d))]
    assert (len(subjects) == 5749), "Number of subjects is: {}!".format(len(subjects))

    file_names = []
    for i in range(len(subjects)):
        sub = subjects[i]
        folder = os.path.join('data/lfw_funneled', sub)
        files = [f for f in os.listdir(folder) if
                 os.path.isfile(os.path.join(folder, f)) and f.lower().endswith('.jpg')]
        for file in files:
            filename = os.path.join(folder, file)
            file_names.append({'filename': filename, 'class_id': i, 'subject': sub})

    assert (len(file_names) == 13233), "Number of files is: {}!".format(len(file_names))

    samples = []
    for item in file_names:
        filename = item['filename']
        class_id = item['class_id']
        sub = item['subject']
        is_valid, bounding_boxes, landmarks = get_face_all_attributes(filename)

        if is_valid:
            samples.append(
                {'class_id': class_id, 'subject': sub, 'full_path': filename, 'bounding_boxes': bounding_boxes,
                 'landmarks': landmarks})

    with open(lfw_pickle, 'wb') as file:
        save = {
            'samples': samples
        }
        pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)


def get_image(samples, transformer, file):
    filtered = [sample for sample in samples if file in sample['full_path'].replace('\\', '/')]
    assert (len(filtered) == 1), 'len(filtered): {} file:{}'.format(len(filtered), file)
    sample = filtered[0]
    full_path = sample['full_path']
    landmarks = sample['landmarks']
    img = align_face(full_path, landmarks)
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.to(device)
    return img


def evaluate(model):
    model.eval()

    with open(lfw_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    filename = 'data/lfw_test_pair.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()

    transformer = data_transforms['val']

    angles = []

    with torch.no_grad():
        for line in lines:
            tokens = line.split()
            file0 = tokens[0]
            img0 = get_image(samples, transformer, file0)
            file1 = tokens[1]
            img1 = get_image(samples, transformer, file1)
            imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
            imgs[0] = img0
            imgs[1] = img1
            output = model(imgs)
            # print('output.size(): ' + str(output.size()))
            feature0 = output[0].cpu().numpy()
            feature1 = output[1].cpu().numpy()
            x0 = feature0 / np.linalg.norm(feature0)
            x1 = feature1 / np.linalg.norm(feature1)
            cosine = np.dot(x0, x1)
            # print('cosine: ' + str(cosine))
            theta = math.acos(cosine)
            theta = theta * 180 / math.pi
            is_same = tokens[2]
            angles.append('{} {}\n'.format(theta, is_same))

    with open('data/angles.txt', 'w') as file:
        file.writelines(angles)


def visualize(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    ones = []
    zeros = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            ones.append(angle)
        else:
            zeros.append(angle)

    bins = np.linspace(0, 180, 181)

    plt.hist(zeros, bins, density=True, alpha=0.5, label='0', facecolor='red')
    plt.hist(ones, bins, density=True, alpha=0.5, label='1', facecolor='blue')

    mu_0 = np.mean(zeros)
    sigma_0 = np.std(zeros)
    y_0 = scipy.stats.norm.pdf(bins, mu_0, sigma_0)
    plt.plot(bins, y_0, 'r--')
    mu_1 = np.mean(ones)
    sigma_1 = np.std(ones)
    y_1 = scipy.stats.norm.pdf(bins, mu_1, sigma_1)
    plt.plot(bins, y_1, 'b--')
    plt.xlabel('theta')
    plt.ylabel('theta j Distribution')
    plt.title(
        r'Histogram : mu_0={:.4f},sigma_0={:.4f}, mu_1={:.4f},sigma_1={:.4f}'.format(mu_0, sigma_0, mu_1, sigma_1))

    plt.legend(loc='upper right')
    plt.plot([threshold, threshold], [0, 0.05], 'k-', lw=2)
    plt.savefig('images/theta_dist.png')
    plt.show()


def accuracy(threshold):
    with open(angles_file) as file:
        lines = file.readlines()

    wrong = 0
    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if type == 1:
            if angle > threshold:
                wrong += 1
        else:
            if angle <= threshold:
                wrong += 1

    accuracy = 1 - wrong / 6000
    return accuracy


def show_bboxes(folder):
    with open(lfw_pickle, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']
    for sample in tqdm(samples):
        full_path = sample['full_path']
        bounding_boxes = sample['bounding_boxes']
        landmarks = sample['landmarks']
        img = cv.imread(full_path)
        img = draw_bboxes(img, bounding_boxes, landmarks)
        filename = os.path.basename(full_path)
        filename = os.path.join(folder, filename)
        cv.imwrite(filename, img)


def error_analysis(threshold):
    with open(angles_file) as file:
        angle_lines = file.readlines()

    fp = []
    fn = []
    for i, line in enumerate(angle_lines):
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        if angle <= threshold and type == 0:
            fp.append(i)
        if angle > threshold and type == 1:
            fn.append(i)

    print('len(fp): ' + str(len(fp)))
    print('len(fn): ' + str(len(fn)))

    num_fp = min(10, len(fp))
    fp_list = random.sample(fp, num_fp)
    num_fn = min(10, len(fn))
    fn_list = random.sample(fn, num_fn)

    filename = 'data/lfw_test_pair.txt'
    with open(filename, 'r') as file:
        pair_lines = file.readlines()

    for i in range(num_fp):
        fp_id = fp_list[i]
        fp_line = pair_lines[fp_id]
        tokens = fp_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fp_0.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fp_1.jpg'.format(i))

    for i in range(num_fn):
        fn_id = fn_list[i]
        fn_line = pair_lines[fn_id]
        tokens = fn_line.split()
        file0 = tokens[0]
        copy_file(file0, '{}_fn_0.jpg'.format(i))
        file1 = tokens[1]
        copy_file(file1, '{}_fn_1.jpg'.format(i))


def copy_file(old, new):
    filename = os.path.join('data/lfw_funneled', old)
    img = cv.imread(filename)
    cv.resize(img, (224, 224))
    filename = os.path.join('images', new)
    cv.imwrite(filename, img)


def get_threshold():
    with open(angles_file, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        tokens = line.split()
        angle = float(tokens[0])
        type = int(tokens[1])
        data.append({'angle': angle, 'type': type})

    min_error = 6000
    min_threshold = 0

    for d in data:
        threshold = d['angle']
        type1 = len([s for s in data if s['angle'] <= threshold and s['type'] == 0])
        type2 = len([s for s in data if s['angle'] > threshold and s['type'] == 1])
        num_errors = type1 + type2
        if num_errors < min_error:
            min_error = num_errors
            min_threshold = threshold

    # print(min_error, min_threshold)
    return min_threshold


def lfw_test(model):
    filename = 'data/lfw-funneled.tgz'
    if not os.path.isdir('data/lfw_funneled'):
        print('Extracting {}...'.format(filename))
        extract(filename)

    # if not os.path.isfile(lfw_pickle):
    print('Processing {}...'.format(lfw_pickle))
    process()

    # if not os.path.isfile(angles_file):
    print('Evaluating {}...'.format(angles_file))
    evaluate(model)

    print('Calculating threshold...')
    # threshold = 70.36
    thres = get_threshold()
    print('Calculating accuracy...')
    acc = accuracy(thres)
    print('Accuracy: {}%, threshold: {}'.format(acc * 100, thres))
    return acc, thres


if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    acc, threshold = lfw_test(model)

    print('Visualizing {}...'.format(angles_file))
    visualize(threshold)

    print('error analysis...')
    error_analysis(threshold)
