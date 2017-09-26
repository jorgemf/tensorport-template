# TensorPort Template

This is a template to train models of [TensorFlow](https://www.tensorflow.org/) in [TensorPort](https://tensorport.com/).


## Adding this repository as submodule

```sh
git submodule add -b master https://github.com/tensorport/tensorport-template tensorport_template/
```

**NOTE**: Currently TensorPort doesn't support submodules from GitHub

## Basic setup of TensorPort

Set the environment variables:

```sh
export PROJECT_DIR="~/myprojects/project/"
export DATA_DIR="~/mydata/data/"
```

Create the projects in TensorPort (see the [get started guide](https://tensorport.com/get-started/)), the `PROJECT_DIR` and `DATA_DIR` need to be git repositories (skip the git commands if they already are git repositories):

```sh
pip install --upgrade git-lfs 
pip install --upgrade tensorport
tport login
cd $PROJECT_DIR
git init
git add *
git commit -m "update project"
tport create project
cd $DATA_DIR
git init
git add *
git-lfs track *
git commit -m "update data"
tport create dataset
cd $PROJECT_DIR
```

## Example

```python
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
```


### Distributed training with continuous evaluation

The recommended way to perform an evaluation at the same time the training is running is by using a new process that loads the checkpoints and runs the model with the evaluation dataset. This functionallity is under [distributed_training.py](https://github.com/tensorport/tensorport-template/blob/master/distributed_training.py). You can use calling to the function `launch_train_evaluation`. The last worker server will be use only for the evaluation. 

For example:

```python
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


def model_fn_example(dataset_tensor, evaluation, batch_size):
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
                                 task_spec=get_task_spec(),
                                 save_checkpoint_secs=10,
                                 save_summaries_steps=10,
                                 log_step_count_steps=10)
    trainer.run(epochs=20, batch_size=8)
    evaluator = DistributedEvaluator(log_dir=logdir,
                                     # using same dataset as training here, only for testing
                                     dataset=MyDummyDataSet(),
                                     model_fn=model_fn_example,
                                     infinite_loop=False)
    evaluator.run()
```

## Running a job

You can use the scrip `train.sh` to update the data in TensorPort and create a new job.

