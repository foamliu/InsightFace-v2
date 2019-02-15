import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import recordio

from config import path_imgidx, path_imgrec

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

# %% 1 ~ 3804847
for i in range(3804846):
    header, s = recordio.unpack(imgrec.read_idx(i + 1))
    img = mx.image.imdecode(s).asnumpy()
    plt.imshow(img)
    plt.title('id=' + str(i) + 'label=' + str(header.label))
    plt.pause(0.1)
