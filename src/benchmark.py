import numpy as np
import tensorflow as tf
import os
import wget

from zipfile import ZipFile

from inference import Model
from models import paths


class Benchmark():
    _data = []
    _data_base_path = '../data/'

    def __init__(self, dataset='coco'):
        tf.enable_eager_execution()

        self.download_dataset(dataset)

    def run(self):
        ''' Execute each model in a given selection of models '''

        for name, url in paths.items():
            model = Model(name, url)

            for image in self._data:
                image_path = self._data_base_path + image
                image_raw = tf.read_file(image_path)
                image_tensor = tf.image.decode_image(image_raw)

                print('[DEBUG] Tensor shape:', image_tensor.shape)

                model.run(np.array(image_tensor.numpy()))

    def download_dataset(self, name):
        ''' Downloads the given dataset for benchmark '''

        print('[INFO] Checking dataset existence...')

        if (name == 'coco'):
            if (not os.path.exists('../data/coco')):
                print('[INFO] Downloading Microsoft COCO dataset:')

                url = 'http://images.cocodataset.org/zips/test2017.zip'
                compressed_data = wget.download(url)

                with ZipFile(compressed_data, 'r') as file:
                    filename = 'test2017'

                    file.extractall('../data/')
                    os.rename('../data/{}'.format(filename),
                              '../data/{}'.format(name))
                    os.remove('{}.zip'.format(filename))

            self._data_base_path = '../data/coco/'
            self._data = os.listdir(self._data_base_path)
        print('[INFO] Images dataset loaded')


if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.run()
