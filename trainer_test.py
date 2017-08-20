from trainer import *
from tensorflow.python.training import training_util


class MyTrainer(Trainer):
    def __init__(self):
        super(MyTrainer, self).__init__('/tmp/logdir', max_time=10)

    def create_graph(self):
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

        return {'x': var_x, 'y': var_y, 'result': result, 'saver': saver,
                'step_counter': global_step}

    def train_step(self, session, graph_data):
        """
        Example of function to run a train step
        :param tf.train.MonitoredSession session: session to run the graph
        :param graph_data: the graph data returned in create_graph
        """
        result, step_counter = session.run([graph_data['result'], graph_data['step_counter']],
                                           feed_dict={
                                               graph_data['x']: 2.1,
                                               graph_data['y']: 3.0
                                           })
        print('{} : {}'.format(step_counter, result))

    def create_hooks(self, graph_data):
        """
        Example of function to create hooks.
        :param graph_data: the graph data returned in create_graph
        :return: A tuple with two lists of hooks or none. First list if the hooks for all nodes and
        the second list are the hooks only for the master node.
        """

        hooks = [
            # stops the training after 10 stetps
            tf.train.StopAtStepHook(last_step=10),
            # stops the raining when the result is nan
            tf.train.NanTensorHook(graph_data['result']),
        ]
        return hooks, None


if __name__ == '__main__':
    trainer = MyTrainer()
    trainer.train()
