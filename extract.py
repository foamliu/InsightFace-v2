import pickle
import os
import cv2 as cv
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm

from config import path_imgidx, path_imgrec
from utils import ensure_folder

if __name__ == "__main__":
    folder = 'data/images'
    ensure_folder(folder)
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    samples = []

    # %% 1 ~ 3804847
    for i in tqdm(range(3804846)):
        try:
            header, s = recordio.unpack(imgrec.read_idx(i + 1))
            img = mx.image.imdecode(s).asnumpy()
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            label = int(header.label)
            filename = '{}.png'.format(i)
            samples.append({'img': filename, label: label})
            filename = os.path.join(folder, filename)
            cv.imwrite(filename, img)
        except:
            pass

    with open('data/faces_ms1m_112x112.pickle', 'wb') as file:
        pickle.dump(samples, file)

    print('num_samples: ' + str(len(samples)))
