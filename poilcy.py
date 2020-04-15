import numpy as np
import tensorflow as tf

from tf_utils import create_checkpoint


class BasePolicy(object):
    def __init__(self, num_action):
        self._num_action = num_action

    def update(self, x, y, epochs=300, batch_size=32, verbose=False):
        """ Learn to be able to select an action given a context """
        del x, y, epochs, batch_size, verbose

    def select_action(self, context):
        """ Take an action based on the computed scores given the context

        :param context: contexts
        :return action: one hot vector representing an action (num_data x num_action)
        :return score: uniform dist prob over each action (num_data x num_action)
        """
        raise NotImplementedError


class UniformPolicy(BasePolicy):
    def __init__(self, num_action):
        super(UniformPolicy, self).__init__(num_action=num_action)

    def select_action(self, context):
        """ returns the uniformly sampled actions
            p(a | x) = 1 / K
            where K is the number of arms(actions)

            :param context: contexts
            :return action: one hot vector representing an action (num_data x num_action)
            :return score: uniform dist prob over each action (num_data x num_action)
        """
        # get the number of samples to sample
        num_data = context.shape[0]

        # sample from Uniform dist
        action_id = np.random.uniform(low=0, high=self._num_action, size=num_data).astype(np.int8)

        # one-hot vectorise the action indices
        action = np.eye(self._num_action)[action_id, :]

        # work out the scores on each action
        score = np.tile(np.ones(self._num_action) / self._num_action, (num_data, 1))
        return action, score


class DeterministicPolicy(BasePolicy):
    def __init__(self, num_action, weight_path="./model"):
        """ Deterministic Policy """
        super(DeterministicPolicy, self).__init__(num_action=num_action)

        class Model(tf.keras.Model):
            def __init__(self, num_action):
                super(Model, self).__init__()
                self.dense1 = tf.keras.layers.Dense(16, activation='relu')
                self.dense2 = tf.keras.layers.Dense(16, activation='relu')
                self.pred = tf.keras.layers.Dense(num_action, activation='softmax')

            def call(self, inputs):
                x = self.dense1(inputs)
                x = self.dense2(x)
                return self.pred(x)

        self._model = Model(num_action=num_action)
        self._optimizer = tf.compat.v1.train.AdamOptimizer()
        self._loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self._global_ts = tf.compat.v1.train.get_or_create_global_step().assign(0)
        self._manager = create_checkpoint(model=self._model,
                                          optimizer=self._optimizer,
                                          model_dir=weight_path)

    def update(self, x, y, epochs=300, batch_size=32, verbose=False):
        """ Learn to be able to select an action given a context """
        # Only if we haven't trained the model yet, we move on to the training.
        # This is because the task is less difficult so that one time of training should be enough!
        if self._global_ts == 0:
            for epoch in range(epochs):
                _idx = np.random.randint(low=0, high=x.shape[0] - 1, size=batch_size)
                _x, _y = np.array(x[_idx, :], dtype=np.float32), np.array(y[_idx, :], dtype=np.float32)
                loss = self._update(x=_x, y=_y)
                if verbose: print("Epoch: [{}/{}] | Loss: {}".format(epoch, epochs, loss.numpy()))
            self._global_ts.assign_add(1)
            self._manager.save()

    @tf.function
    def _update(self, x, y):
        with tf.GradientTape() as tape:
            pred = self._model(x)
            loss = self._loss_fn(y_pred=pred, y_true=y)

            # get gradients
            grads = tape.gradient(loss, self._model.trainable_variables)

            # apply processed gradients to the network
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        return loss

    def select_action(self, context):
        """ Get scores for each arm(action) given a context and compute an action """
        # get scores of each action given a context(state in RL literature)
        score = np.asarray(self._model(context))

        # get an action vector: num_action dim vector with 1 representing the predicted label
        # TODO: think if we really need to one-hot vectorise the action vector
        action = np.eye(score.shape[1])[np.argmax(score, axis=-1) - 1]
        return action, score


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from data.data_manager import load_ecoli
    from utils import eager_setup

    eager_setup()

    data = load_ecoli()  # [ecoli] x: (336, 7) y: (336, 8) num_label: 8
    train_x, test_x, train_y, test_y = train_test_split(data.x, data.y, test_size=0.3)
    label = np.argmax(test_y, axis=1)

    print("=== Test UniformPolicy ===")
    policy = UniformPolicy(num_action=data.num_label)
    action, score = policy.select_action(context=test_x)
    pred = np.argmax(score, axis=1)
    print(np.mean(pred == label))
    print("Taken action: {} Score: {}".format(action.shape, score.shape))

    print("=== Test DeterministicPolicy ===")
    policy = DeterministicPolicy(num_action=data.num_label, weight_path="./model/{}".format("ecoli"))
    policy.update(x=train_x, y=train_y, epochs=1000, batch_size=64, verbose=False)
    action, score = policy.select_action(context=test_x)
    pred = np.argmax(score, axis=1)
    print("Accuracy: {}".format(np.mean(pred == label)))
    print("Taken action: {} Score: {}".format(action.shape, score.shape))
