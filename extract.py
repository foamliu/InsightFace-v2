import os

import zipfile


def extract(filename):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall('data')
    zip_ref.close()


if __name__ == "__main__":
    if not os.path.isdir('data/faces_ms1m_112x112'):
        extract('data/faces_ms1m_112x112.zip')
