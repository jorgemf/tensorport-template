# TensorPort Template

This is a template to train models of [TensorFlow](https://www.tensorflow.org/) in [TensorPort](https://tensorport.com/).


## Adding this repository as submodule

```sh
git submodule add -b master https://github.com/jorgemf/tensorport-template tensorport_template/
```

**NOTE**: Currently TensorPort doesn't support submodules in GitHub

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

| | | |
|-|-|-|
| trainer_test.py | basic example that feeds the data directly in `session.run()` |
| trainer_dataset_test.py | example that uses the `Dataset` class from TensorFlow to load the data from txt files |
| trainer_dataset_distributed_test.py | like trainer_dataset_test but adapted for distributed training. _Not supported yet as it uses APIs in development_ |

## Running a job

You can use the scrip `train.sh` to update the data in TensorPort and create a new job.