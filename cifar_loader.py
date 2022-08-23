import pickle
import os
import numpy as np

class CIFAR10:
    def __init__(self, data_path) -> None:

        self.data_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = os.path.join(self.data_path, data_path)

        train_filenames = ["data_batch_%d" % i for i in range(1,6)]
        test_filename = "test_batch"
        
        test_dict = self.unpickle(os.path.join(self.data_path, test_filename))        
        self.test_data = test_dict[b'data']
        self.test_labels = test_dict[b'labels']

        self.train_data = []
        self.train_labels = []
        for train_filename in train_filenames:
            train_dict = self.unpickle(os.path.join(self.data_path, train_filename))
            self.train_data += [train_dict[b'data']]
            self.train_labels += [train_dict[b'labels']]
        self.train_data = np.concatenate(self.train_data, axis=0)
        self.train_labels = np.concatenate(self.train_labels, axis=0)

        label_names = self.unpickle(os.path.join(self.data_path, "batches.meta"))
        self.label_names = [x.decode("utf-8") for x in label_names[b'label_names']]


    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict