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
from tensorport_template.trainer import Trainer
import tensorflow as tf

class MyTrainer(Trainer):
    def __init__(self):
        # set where your dataset is
        dataset_spec = DatasetSpec('test', '', 'local_repo')
        # call the super constructor with at least the dataset_spec and the log dir
        super(MyTrainer, self).__init__(dataset_spec, '/tmp/logdir', max_time=10,
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

    def pre_train(self, session, graph_data):
    	# load the checkpoints here
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt:
            graph_data['saver'].restore(session, ckpt.model_checkpoint_path)

    def post_train(self, session, graph_data):
        # trained finished but session still live
        print('Training finished! Loss: {}'.format(graph_data['loss'].eval()))


if __name__ == '__main__':
    trainer = MyTrainer()
    trainer.train(create_graph_fn=trainer.create_graph,
                  train_step_fn=trainer.train_step,
                  pre_train_fn=trainer.pre_train,
                  post_train_fn=trainer.post_train)
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