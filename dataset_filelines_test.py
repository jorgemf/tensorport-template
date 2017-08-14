from dataset_filelines import DatasetFilelines
from trainer import Trainer
import tensorflow as tf
from tensorflow.python.training import training_util
import numpy as np


class MyDatasetFilelines(DatasetFilelines):
    def __init__(self):
        super(MyDatasetFilelines, self).__init__('test', ['dataset_filelines_test_1.txt',
                                                          'dataset_filelines_test_2.txt'], 10)

    def py_func_parse_example(self, example_serialized):
        return [
            np.asarray(int(example_serialized), dtype=np.int32),
            np.asarray(-int(example_serialized), dtype=np.int32)
        ]

    def py_func_parse_example_types(self):
        return [tf.int32, tf.int32]

    def py_func_parse_example_inputs_outputs(self):
        return 1, 1

    def py_fun_parse_example_reshape(self, inputs, outputs):
        inputs[0] = tf.reshape(inputs[0], [1])
        outputs[0] = tf.reshape(outputs[0], [1])
        return inputs, outputs


class MyTrainer(Trainer):
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = 2
        num_steps = dataset.get_size() / self.batch_size
        super(MyTrainer, self).__init__('/tmp/logdir', num_steps=num_steps)

    def create_graph(self):
        inputs, outputs = self.dataset.read(self.batch_size)
        self.global_step = training_util.get_or_create_global_step()
        global_step_increase = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_increase]):
            self.inputs = tf.identity(inputs)
            self.outputs = tf.identity(outputs)

    def train_step(self, session, graph_data):
        step, value_inputs, value_outputs = session.run([self.global_step,
                                                         self.inputs, self.outputs])
        print('{}: {}, {}'.format(step, value_inputs.tolist(), value_outputs.tolist()))


if __name__ == '__main__':
    dataset = MyDatasetFilelines()
    print('{} records in the dataset'.format(dataset.get_size()))
    MyTrainer(dataset).train()
