from trainer import Trainer, get_task_spec
import numpy as np
import multiprocessing
import time
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.contrib.data import TextLineDataset, Dataset


class MyTrainer(Trainer):
    def __init__(self, dataset):
        self.dataset = dataset
        self.print_timestamp = 0
        super(MyTrainer, self).__init__('/tmp/logdir')

    def create_graph(self):
        next_tensor = self.dataset.make_one_shot_iterator().get_next()
        inputs, outputs = next_tensor[0], next_tensor[1]
        self.global_step = training_util.get_or_create_global_step()
        global_step_increase = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_increase]):
            self.inputs = tf.identity(inputs)
            self.outputs = tf.identity(outputs)

    def begin(self):
        self.print_timestamp = time.time()

    def train_step(self, session, graph_data):
        step, value_inputs, value_outputs = session.run([self.global_step,
                                                         self.inputs, self.outputs])
        # print information every 5 minutes
        if self.is_chief and time.time() > self.print_timestamp + 5 * 60:
            print('{}: {}, {}'.format(step, value_inputs.tolist(), value_outputs.tolist()))
            self.print_timestamp = time.time()


def create_dataset(num_epochs, num_workers, worker_index):
    # create the dataset of files with the data
    dataset = Dataset.list_files('dataset_filelines_test_*.txt')

    # split the dataset in shards, be sure the number of files is proportional
    # to the number of workers
    dataset = dataset.shard(num_workers, worker_index)

    # set the number of epochs
    dataset = dataset.repeat(num_epochs)

    # shuffle the data, use a big buffer size to shuffle all filenames
    dataset = dataset.shuffle(buffer_size=100)

    # read on sample per data file of each shard
    dataset = dataset.interleave(TextLineDataset,
                                 # number of readers the same as number of CPUs
                                 cycle_length=multiprocessing.cpu_count(),
                                 # block size is 1 to get directly a flat map
                                 block_length=1)

    # function to parse each line in the file lines
    def _parse_line(line):
        return np.int32(line), np.int32(line)

    # process each example
    dataset = dataset.map(
        lambda line: tf.py_func(_parse_line, [line], [tf.int32, tf.int32]),
        # use as many threads as CPUs + 1
        num_parallel_calls=multiprocessing.cpu_count() + 1,
        # buffer the data as CPUs * batch_size + minimum_size
        output_buffer_size=batch_size * multiprocessing.cpu_count() + 3
    )
    return dataset


if __name__ == '__main__':
    # TODO this is for the future TensorFlow 1.4

    # get the task spec of distributed training
    task_spec = get_task_spec()

    # join if this is a parameter server and quit when it finish, otherwise continue
    if task_spec.join_if_ps():
        quit()

    batch_size = 2
    epochs = 1

    # create the dataset and count the number of samples
    dataset = create_dataset(epochs, task_spec.num_workers, task_spec.index)

    # shuffle the dataset for training
    dataset = dataset.shuffle(buffer_size=10)
    # set the batch size
    dataset = dataset.batch(batch_size)

    # run the training for the whole dataset
    MyTrainer(dataset=dataset).train()
