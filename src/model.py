import numpy as np
import tensorflow as tf
import os
import wget
import tarfile
import matplotlib.pyplot as plt

from networks import graph_path, tensors
from timer import Timer
from utils import resize


class Model():
    def __init__(self, name, url):
        self._name = name
        self._url = url
        self._session = None
        self._timer = Timer(name)

        self.setup()

    def __del__(self):
        if (self._session):
            self._session.close()

    def run(self, image):
        print('[INFO] Executing inference for {}'.format(self._name))

        input_tensor_name = tensors[self._name]['input']

        expanded_image = np.expand_dims(resize(image), 0)

        print('[INFO] Input Image shape:', expanded_image.shape)

        self._timer.start()
        output = self._session.run(tensors[self._name]['output'],
                                   feed_dict={
                                       input_tensor_name: expanded_image
                                   })
        self._timer.end()

        print('[DEBUG] Forward pass finished! Predictions count:' +
              ' {}, elapsed time: {}'.format(output[1],
                                             self._timer.checkpoint['elapsed'])
              )

    def setup(self):
        '''
            Setup Tensorflow session to execute a given model. The steps
            for setup are:
                - Download and extract pretrained models;
                - Setup the Session used during inference;
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
        print('[INFO] Graph for {} loaded'.format(self._name))

    def show(self, image_data):
        plt.imshow(image_data)
        plt.show()

    @property
    def graph(self):
        return self._session.graph

    @property
    def timer(self):
        return self._timer.checkpoints
