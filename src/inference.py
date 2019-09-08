import tensorflow as tf
import os
import wget
import tarfile

from models import graph_path


class Model():
    def __init__(self, name, url):
        self._name = name
        self._url = url

        self.setup()

    def run(self, image):
        print('[INFO] Will execute inference for {}'.format(self._name))

    def setup(self):
        '''
            Setup Tensorflow session to execute a given model. The steps
            for setup are:
                - Download and extract pretrained models;
        '''
        print('[INFO] Creating session with model {}'.format(self._name))

        if (not os.path.exists(graph_path(self._name))):
            print('[INFO] Downloading {} weights & graph:'.format(self._name))
            weights_file = wget.download(self._url,
                                         out="{}.tar.gz".format(self._name))

            tar = tarfile.open(weights_file)
            tar.extractall(path='../model/')
            folder = tar.getmembers()[0]
            folder_name = folder.name.partition('/')[0]
            os.rename('../model/{}'.format(folder_name),
                      '../model/{}'.format(self._name))
            os.remove('{}.tar.gz'.format(self._name))

        with tf.gfile.FastGFile(graph_path(self._name), 'rb') as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())
            final_graph = tf.import_graph_def(graph_def, name="")

        self._session = tf.Session(graph=final_graph)
        print('\n[INFO] Graph for {} loaded'.format(self._name))
