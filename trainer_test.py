import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_util
from trainer import Trainer
from tf_dataset import TFDataSet


class MyDataSet(TFDataSet):
    def __init__(self):
        super(MyDataSet, self).__init__('my_dataset', 'dataset_filelines_test_*.txt',
                                        min_queue_examples=2, shuffle_size=5)

    def _map(self, example_serialized):
        def _parse(line):
            return np.int32(line), np.int32(line)

        a, b = tf.py_func(_parse, [example_serialized], [tf.int32, tf.int32], stateful=True)
        return a, b


class MyTrainer(Trainer):
    def __init__(self, dataset):
        super(MyTrainer, self).__init__('/tmp/logdir', dataset=dataset)

    def create_graph(self, dataset_tensor, batch_size):
        inputs, outputs = dataset_tensor
        self.global_step = training_util.get_or_create_global_step()
        global_step_increase = tf.assign_add(self.global_step, 1)
        with tf.control_dependencies([global_step_increase]):
            self.inputs = tf.identity(inputs)
            self.outputs = tf.identity(outputs)

    def step(self, session, graph_data):
        step, value_inputs, value_outputs = session.run([self.global_step,
                                                         self.inputs, self.outputs])
        print('{}: {}, {}'.format(step, value_inputs.tolist(), value_outputs.tolist()))


if __name__ == '__main__':
    # run the training
    MyTrainer(dataset=MyDataSet()).run(epochs=1, batch_size=2)
