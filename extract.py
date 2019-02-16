import pickle

import mxnet as mx
from mxnet import recordio
from tqdm import tqdm

from config import path_imgidx, path_imgrec

if __name__ == "__main__":
    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    samples = []

    # %% 1 ~ 3804847
    for i in tqdm(range(3804846)):
        try:
            header, s = recordio.unpack(imgrec.read_idx(i + 1))
            img = mx.image.imdecode(s).asnumpy()
            label = int(header.label)
            samples.append({'img': img, label: label})
        except:
            pass

    with open('data/faces_ms1m_112x112.pickle', 'wb') as file:
        pickle.dump(samples, file)

    print('num_samples: ' + str(len(samples)))
