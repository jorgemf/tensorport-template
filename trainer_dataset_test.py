from trainer import Trainer, get_task_spec
import numpy as np
import multiprocessing
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.contrib.data import TextLineDataset, Dataset


class MyTrainer(Trainer):
    def __init__(self, dataset, num_steps):
        self.dataset = dataset
        super(MyTrainer, self).__init__('/tmp/logdir', num_steps=num_steps)

    def create_graph(self):
        next_tensor = self.dataset.make_one_shot_iterator().get_next()
        inputs, outputs = next_tensor[0], next_tensor[1]
        self.global_step = training_util.get_or_create_global_step()
        global_step_increase = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_increase]):
            self.inputs = tf.identity(inputs)
            self.outputs = tf.identity(outputs)

    def train_step(self, session, graph_data):
        step, value_inputs, value_outputs = session.run([self.global_step,
                                                         self.inputs, self.outputs])
        print('{}: {}, {}'.format(step, value_inputs.tolist(), value_outputs.tolist()))


def create_dataset():
    # create a dataset that read data from txt files
    dataset = TextLineDataset(['dataset_filelines_test_1.txt', 'dataset_filelines_test_2.txt'])

    # count the number of examples in the dataset with an iterator, we suppose one sample per line
    count_iterator = dataset.repeat(1).make_one_shot_iterator()
    samples = 0
    try:
        next_element = count_iterator.get_next()
        with tf.Session() as sess:
            while True:
                sess.run(next_element)
                samples += 1
    except:
        pass
    print('{} samples in the dataset'.format(samples))

    # function to parse each line in the file lines
    def _parse_line(line):
        return np.int32(line), np.int32(line)

    # map the read line into tensors
    dataset = dataset.map(
        lambda line: tf.py_func(_parse_line, [line], [tf.int32, tf.int32]),
        # use as many threads as CPUs + 1
        num_threads=multiprocessing.cpu_count() + 1,
        # buffer the data as CPUs * batch_size + minimum_size
        output_buffer_size=batch_size * multiprocessing.cpu_count() + 3
    )

    return dataset, samples

if __name__ == '__main__':
    batch_size = 2
    epochs = 1

    # create the dataset and count the number of samples
    dataset, samples = create_dataset()

    # shuffle the dataset for training
    dataset = dataset.shuffle(buffer_size=10)
    # set the batch size
    dataset = dataset.batch(batch_size)

    # number of steps in an epoch is the dataset size divided by the batch size
    one_epoch_steps = samples / batch_size

    # run the training
    MyTrainer(dataset=dataset, num_steps=one_epoch_steps * epochs).train()
