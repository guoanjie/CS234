import tensorflow as tf
import tensorflow.contrib.layers as layers

from core.deep_q_learning import DQN
import config


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        state_shape = list(self.env.observation_space.shape)

        self.s = tf.placeholder(tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2] * config.state_history), name="state")
        self.a = tf.placeholder(tf.int32, shape=(None,), name="action")
        self.r = tf.placeholder(tf.float32, shape=(None,), name="reward")
        self.sp = tf.placeholder(tf.uint8, shape=(None, state_shape[0], state_shape[1], state_shape[2] * config.state_history), name="next_state")
        self.done_mask = tf.placeholder(tf.bool, shape=(None,), name="done_mask")
        self.lr = tf.placeholder(tf.float32, shape=(), name="lr")


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        flattened = layers.flatten(state, scope=scope)
        out = layers.fully_connected(flattened, num_actions, activation_fn=None, reuse=reuse, scope=scope)

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        
        collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        ops = [tf.assign(target_var, var) for var, target_var in zip(collection, target_collection)]
        self.update_target_op = tf.group(*ops)


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        not_done = 1 - tf.cast(self.done_mask, dtype=tf.float32)
        q_samp = self.r + self.config.gamma * not_done * tf.reduce_max(target_q, axis=1)
        q_sa = tf.reduce_sum(q * tf.one_hot(self.a, num_actions), axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q_sa))


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """

        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        grads_and_vars = opt.compute_gradients(self.loss, variables)
        if self.config.grad_clip:
            grads_and_vars = [(tf.clip_by_norm(gv[0], self.config.clip_val), gv[1]) for gv in grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.grad_norm = tf.global_norm([gv[0] for gv in grads_and_vars])
