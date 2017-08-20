# TensorPort Template

This is a template to train models of [TensorFlow](https://www.tensorflow.org/) in [TensorPort](https://tensorport.com/).


## Adding this repository as submodule

```sh
git submodule add -b master https://github.com/jorgemf/tensorport-template tensorport_template/
```

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

## Training a model

Create your model and the functions to run a train step:

```python
from tensorport_template.trainer import Trainer, get_task_spec
import tensorflow as tf

class MyTrainer(Trainer):
    def __init__(self):
        # call the super constructor with at least the dataset_spec and the log dir
        super(MyTrainer, self).__init__('/tmp/logdir', max_time=10,
                                        hooks=[tf.train.StopAtStepHook(last_step=10)])

    def create_graph(self):
    	# create your graph here and return information about it if you want
    	graph_data = {'input': input_tensor, 
    	              'output': output_tensor,
    	              'loss': loss_tensor,
    	              'saver': tf.train.Saver()}
    	return graph_data

    def train_step(self, session, graph_data):
        # run a train step
    	result = session.run([graph_data['output']],
    	                     feed_dict={ graph_data['input']: ... })

    def create_hooks(self, graph_data):
        hooks = [
            # stops the raining when the result is nan
            tf.train.NanTensorHook(graph_data['result']),
        ]
        return hooks, None


if __name__ == '__main__':
    trainer = MyTrainer()
    trainer.train()
```

Test it works in local:

```sh
python mytrainer.py
```

Set the environment variables:

```sh
export PROJECT_DIR="~/myprojects/project/"
export DATA_DIR="~/mydata/data/"
```

Run the `train.sh` script that update the repositories and runs a job in TensorPort:


```sh
./tensorport_template/train.sh
```

## Datasets

Aditionally you can set up datasets extending the classes `DatasetFilelines` if you read the data from a text file or `DatasetTFrecords` if you read the data from TensorFlow records (protocol buffers with the data samples).

Here is an example to read records from a text file:

```python
from dataset_filelines import DatasetFilelines
from trainer import Trainer, get_task_spec
import tensorflow as tf
from tensorflow.python.training import training_util
import numpy as np


class MyDatasetFilelines(DatasetFilelines):
    def __init__(self):
        super(MyDatasetFilelines, self).__init__('test', ['dataset_filelines_test_1.txt',
                                                          'dataset_filelines_test_2.txt'], 10)

    def py_func_parse_example(self, example_serialized):
        # perform any logic here with the example_serialized, this is python code. Use
        # py_fun_parse_example_reshape if you want to do transformation with TensorFlow
        return [
            np.asarray(int(example_serialized), dtype=np.int32),
            np.asarray(-int(example_serialized), dtype=np.int32)
        ]
        # if you want to return serveral inputs or outputs:
        # return [ input_1, input_2, ..., output_1, output_2, ... ]
        # also set up py_func_parse_example_inputs_outputs() according to it

    def py_func_parse_example_types(self):
        return [tf.int32, tf.int32]

    def py_func_parse_example_inputs_outputs(self):
        return 1, 1

    def py_fun_parse_example_reshape(self, inputs, outputs):
        # reshape the inputs and outputs to the real shapes (no batching here), you can
        # also perform any transformation with TensorFlow of the inputs/outputs
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

```

