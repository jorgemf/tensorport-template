from trainer import *
from tensorflow.python.training import training_util


class MyTrainer(Trainer):
    def __init__(self):
        dataset_spec = DatasetSpec('test', '', 'local_repo')
        super(MyTrainer, self).__init__(dataset_spec, '/tmp/logdir', max_time=10,
                                        hooks=[tf.train.StopAtStepHook(last_step=10)])

    def sample_create_graph_fn(self):
        """
        Example of function to create a graph
        :return: Information related with the graph. It will be passed to other functions
        """
        var_x = tf.placeholder(dtype=tf.float32, shape=[], name='x')
        var_y = tf.placeholder(dtype=tf.float32, shape=[], name='y')
        result = tf.multiply(var_x, var_y, name='result')

        global_step = training_util.get_or_create_global_step()
        # as we don't have an optimizer we need to update the global step ourselves
        global_step_increase = tf.assign_add(global_step, 1)
        with tf.control_dependencies([global_step_increase]):
            result = tf.identity(result)

        saver = tf.train.Saver()
        tf.summary.scalar('result', result)

        # add a hook to stop the training in nan value of result (useful for the loss)
        self.hooks.append(tf.train.NanTensorHook(result))

        return {'x': var_x, 'y': var_y, 'result': result, 'saver': saver,
                'step_counter': global_step}

    def sample_train_step_fn(self, session, graph_data):
        """
        Example of function to run a train step
        :param tf.train.MonitoredSession session: session to run the graph
        :param graph_data: the graph data returned in create_graph_fn
        """
        result, step_counter = session.run([graph_data['result'], graph_data['step_counter']],
                                           feed_dict={
                                               graph_data['x']: 2.1,
                                               graph_data['y']: 3.0
                                           })
        print('{} : {}'.format(step_counter, result))

    def sample_pre_train_fn(self, session, graph_data):
        """
        This function is called just before the firs time train_step_fn is called.
        It is useful, for example, to load models from checkpoints.
        :param tf.train.MonitoredSession session: session to run the graph
        :param graph_data: the graph data returned in create_graph_fn
        """
        # example to restore a graph from a checkpoint and creating the global_step
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt:
            graph_data['saver'].restore(session, ckpt.model_checkpoint_path)

    def sample_post_train_fn(self, session, graph_data):
        """
        This function is called just after the train finished.
        :param tf.train.MonitoredSession session: session to run the graph
        :param graph_data: the graph data returned in create_graph_fn
        """
        pass


if __name__ == '__main__':
    trainer = MyTrainer()
    trainer.train(create_graph_fn=trainer.sample_create_graph_fn,
                  train_step_fn=trainer.sample_train_step_fn)
