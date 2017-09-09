from distributed_training import *
from tf_dataset import TFDataSet
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.python.training import training_util
import numpy as np


class MyDummyDataSet(TFDataSet):
    def __init__(self):
        super(MyDummyDataSet, self).__init__('my_dataset', 'dataset_filelines_test_*.txt',
                                             min_queue_examples=2, shuffle_size=5)

    def _map(self, example_serialized):
        def _parse(line):
            input = np.float32(line)
            # a simple equation of the input to generate the output
            output = input + 10 + input * 2
            # generate 2 inputs and 1 output
            return input, np.float32(input * 3), np.float32(output)

        input_1, input_2, output = tf.py_func(func=_parse,
                                              inp=[example_serialized],
                                              Tout=[tf.float32, tf.float32, tf.float32],
                                              stateful=True)
        # set shapes for data
        input_1 = tf.reshape(input_1, [1])
        input_2 = tf.reshape(input_2, [1])
        output = tf.reshape(output, [1])
        # we could perform this operation here or in the graph
        input = tf.concat([input_1, input_2], axis=0)
        return input, output


def model_fn_example(dataset_tensor, evaluation):
    input, output = dataset_tensor
    net_output = layers.fully_connected(input, 1, activation_fn=None)
    batch_error = losses.mean_squared_error(output, net_output)
    graph_data = {}
    global_step = training_util.get_or_create_global_step()

    # use different metrics depending of evaluation
    if evaluation:
        # accumulate the error for the result
        error_sum = tf.Variable(0.0, dtype=tf.float32, name='accumulated_error', trainable=False)
        error_sum = tf.assign_add(error_sum, batch_error)
        count = tf.Variable(0.0, dtype=tf.float32, name='data_samples', trainable=False)
        count = tf.assign_add(count, 1)
        error = error_sum / count
        graph_data['error'] = error
    else:
        # use moving averages for the error
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        update_op = ema.apply([batch_error])
        error = ema.average(batch_error)
        # add train operator
        sgd = tf.train.GradientDescentOptimizer(0.00001)
        train_op = sgd.minimize(batch_error, global_step)
        graph_data['error'] = error
        graph_data['update_op'] = update_op
        graph_data['train_op'] = train_op

    # add error to summary
    tf.summary.scalar('mse_error', error)
    return graph_data


if __name__ == '__main__':
    logdir = '/tmp/tensorport_template_test_logdir'
    trainer = DistributedTrainer(log_dir=logdir,
                                 dataset=MyDummyDataSet(),
                                 model_fn=model_fn_example,
                                 batch_size=8,
                                 epochs=20,
                                 task_spec=get_task_spec(),
                                 save_checkpoint_secs=10,
                                 save_summaries_steps=10,
                                 log_step_count_steps=10)
    trainer.run()
    evaluator = DistributedEvaluator(log_dir=logdir,
                                     # using same dataset as training here, only for testing
                                     dataset=MyDummyDataSet(),
                                     model_fn=model_fn_example,
                                     infinite_loop=False)
    evaluator.run()
