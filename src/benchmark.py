import tensorflow as tf
import os
import wget

from zipfile import ZipFile

from inference import Model
from models import paths


class Benchmark():
    _data = []

    def __init__(self):
        self.download_dataset('coco')

    def run(self):
        ''' Execute each model in a given selection of models '''
        for name, url in paths.items():
            print('##### {} #####\n'.format(name))

            model = Model(name, url)

            for image in self._data:
                model.run(image)
                pass

    def download_dataset(self, name):
        ''' Downloads the given dataset for benchmark '''
        print('[INFO] Checking dataset existence...')

        if (name == 'coco' and not os.path.exists('../data/coco')):
            print('[INFO] Downloading Microsoft COCO dataset:')

            url = 'http://images.cocodataset.org/zips/test2017.zip'
            compressed_data = wget.download(url)

            with ZipFile(compressed_data, 'r') as file:
                file.extractall('../data/')

                os.rename('../data/{}'.format('test2017'),
                          '../data/{}'.format(name))

                self._data = file.namelist()
        print('\n[INFO] Data:', self._data)


if __name__ == '__main__':
    benchmark = Benchmark()
    benchmark.run()
