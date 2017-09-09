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

## Examples

You have several examples to create and train a model:

| [trainer_test.py](https://github.com/tensorport/tensorport-template/blob/master/trainer_test.py) | basic example that feeds the data directly in `session.run()` |
| [trainer_dataset_test.py](https://github.com/tensorport/tensorport-template/blob/master/trainer_dataset_test.py) | example that uses the `TFDataSet` class  load the data from txt files. `TFDataSet` is basically a wrapper of the TensorFlow class `Dataset` with support for distributed training |


### Distributed training with continuous evaluation

The recommended way to perform an evaluation at the same time the training is running is by using a new process that loads the checkpoints and runs the model with the evaluation dataset. This functionallity is under [distributed_training.py](https://github.com/tensorport/tensorport-template/blob/master/distributed_training.py). You can use calling to the function `launch_train_evaluation`. The last worker server will be use only for the evaluation. 

For example:

```python
from distributed_training import launch_train_evaluation
from tf_dataset import TFDataSet
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.python.training import training_util
import numpy as np

# dummy dataset
class MyDummyDataSet(TFDataSet):
    def __init__(self):
        super(MyDummyDataSet, self).__init__(name='my_dataset', 
        									 data_files_pattern='dataset_filelines_test_*.txt',
                                             min_queue_examples=2, 
                                             shuffle_size=5)

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
        # concat all the input into one tensor, we could also return 3 values in the 
        # tuple and make the concat in the graph
        input = tf.concat([input_1, input_2], axis=0)
        return input, output

# function that creates the model
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
    else:
        # use moving averages for the error
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        update_op = ema.apply([batch_error])
        error = ema.average(batch_error)
        sgd = tf.train.GradientDescentOptimizer(0.00001)
        train_op = sgd.minimize(batch_error, global_step)
        # graph to execute in session.run for the training
        graph_data['error'] = error
        graph_data['update_op'] = update_op
        graph_data['train_op'] = train_op

    # add error to summary, this will show in tensorboard for training and test.
    # the summary operator will be executed in session.run during evaluation
    tf.summary.scalar('mse_error', error)
    return graph_data

if __name__ == '__main__':
	# use same dataset for training and testing, usually you will have 2 different dataset
	dataset = MyDummyDataSet()
	log_dir = '/tmp/logdir'  # /tmp/logdir/eval will contain the evaluation summary 
	launch_train_evaluation(model_fn=model_fn_example, 
							log_dir=log_dir, 
							epochs=10, 
							train_batch_size=16, 
							train_datasest=dataset,
                            test_dataset=dataset)
```

## Running a job

You can use the scrip `train.sh` to update the data in TensorPort and create a new job.

