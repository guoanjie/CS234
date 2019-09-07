import tensorflow as tf
import tensorflow.contrib.layers as layers

from linear import Linear


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        with tf.variable_scope(scope, reuse=reuse):
            conv_1 = layers.conv2d(state, 32, 8, stride=4, activation_fn=tf.nn.relu)
            conv_2 = layers.conv2d(conv_1, 64, 4, stride=2, activation_fn=tf.nn.relu)
            conv_3 = layers.conv2d(conv_2, 64, 3, stride=1, activation_fn=tf.nn.relu)
            fc = layers.fully_connected(layers.flatten(conv_3), 512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(fc, num_actions, activation_fn=None)

        return out
