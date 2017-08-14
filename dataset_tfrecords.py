import logging
import os.path as path

import tensorflow as tf
from dataset import Dataset


class DatasetTFrecords(Dataset):
    def __init__(self, name, data_dir, min_queue_examples, num_preprocess_threads=None,
                 num_readers=None):
        """
        :param name: name of the dataset. The tf records would have this name as prefix
        :param data_dir: directory with the tf records with the format name-*
        :param min_queue_examples: see Dataset
        :param num_preprocess_threads: see Dataset
        :param num_readers: see Dataset
        """
        self._size = None
        self.data_dir = data_dir
        super(DatasetTFrecords, self).__init__(name=name,
                                               min_queue_examples=min_queue_examples,
                                               num_preprocess_threads=num_preprocess_threads,
                                               num_readers=num_readers)

    def _count_num_records(self):
        """
        Goes throw all te examples and counts them. This function is called from get_size the first
        time.
        :return int: the number of examples
        """
        size = 0
        g = tf.Graph()
        with g.as_default():
            queue = tf.train.string_input_producer(self.data_files(), num_epochs=1, shuffle=False)
            _, example_serialized = self.get_reader().read(queue)
            inputs, outputs = self.decode_tfrecord(example_serialized, only_counting=True)

            sess = tf.Session(graph=g)
            with sess.as_default():
                logging.info('Counting TensorFlow records in %s...', self.name)
                sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                try:
                    while True:
                        sess.run([inputs, outputs])
                        size += 1
                except tf.errors.OutOfRangeError:
                    pass
                finally:
                    coord.request_stop()
            coord.join(threads)
        logging.info('%d records in %s', size, self.name)
        return size

    def get_size(self):
        if self._size is None:
            self._size = self._count_num_records()
        return self._size

    def parse_example(self, example_serialized):
        return self.decode_tfrecord(example_serialized)

    def decode_tfrecord(self, tfrecord, only_counting=False):
        """
        Decode a tf record and return the inputs and outputs
        :param tfrecord: tf record to decode
        :param boolean only_counting: Whether this function was called from count_num_records or
        not. If this was called form count_num_records the function can skip the operations to
        process the data.
        :return: a tuple (inputs,outputs) with the input data for the model and the expected output.
        The input and output can be any a list of tensors if we have several inputs or outputs for
        the same graph.
        """
        raise NotImplementedError('Should have implemented this')

    def get_reader(self):
        return tf.TFRecordReader()

    def data_files(self):
        """
        :return: python list of all (sharded) data set files.
        :raises: ValueError: if there are not data_files matching the subset.
        """
        tf_record_pattern = path.join(self.data_dir, self.name + '-*')
        data_files = tf.gfile.Glob(tf_record_pattern)

        if not data_files:
            raise ValueError('No files found for dataset {} in {}'.format(self.name, self.data_dir))
        return data_files
