import os
import definitions

import numpy as np


class Loader(object):
    def __init__(self, data_dir=definitions.data_dir):
        self.data_dir = data_dir
        self.images_dir = definitions.train_images_dir
        self.target_dir = definitions.train_target_dir

        self.batch_prefix = 'batch_'
        self.file_format = '.npz'
        self.get_batch_name = lambda number: self.batch_prefix + str(number) + self.file_format

        self.current_batch = 0
        self.batches_count = len(os.listdir(self.images_dir))

    def load_batch(self):
        self.current_batch += 1
        if self.current_batch > self.batches_count:
            self.current_batch = 1

        batch_name = self.get_batch_name(self.current_batch)
        target_file = definitions.path_join(self.target_dir, batch_name)
        images_file = definitions.path_join(self.images_dir, batch_name)

        data = np.load(target_file)
        targets, ids = data['targets'], data['ids']

        data = np.load(images_file)
        images = data['images']

        return images, targets, ids

    def get_batches(self, batches_count=None, start_batch=None):
        if batches_count == None:
            batches_count = self.batches_count

        if start_batch == None:
            start_batch = 1
            self.current_batch = 0
        elif batches_count > self.batches_count:
            batches_count = self.batches_count
            print('Max batches count = {}'.format(batches_count))

        for i in range(start_batch-1, batches_count):
            try:
                answer = self.load_batch() #TODO: load_batch()
                print('{} was loaded'.format(self.get_batch_name(self.current_batch)))
            except:
                print('{} was not loaded'.format(self.get_batch_name(self.current_batch)))
                continue
            yield answer