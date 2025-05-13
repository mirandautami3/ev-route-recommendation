import numpy as np
import random
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size    = state_size
        self.action_size   = action_size
        self.memory        = deque(maxlen=5000)
        self.gamma         = 0.99    # discount rate
        self.epsilon       = 1.0     # exploration rate
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995   # decay per replay
        # build main & target networks
        self.model        = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        opt = tf.keras.optimizers.Adam(learning_rate=2.5e-4)
        model.compile(optimizer=opt, loss='mse')
        return model

    def update_target_model(self):
        # copy weights from primary model to target model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # store experience tuple
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # epsilon-greedy on full action space
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # exploitation: pick action with highest Q-value
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        # prepare training batches
        states      = np.vstack([m[0] for m in minibatch])
        actions     = [m[1] for m in minibatch]
        rewards     = [m[2] for m in minibatch]
        next_states = np.vstack([m[3] for m in minibatch])
        dones       = [m[4] for m in minibatch]

        # predict Q-values for current states and next states
        q_current = self.model.predict(states, verbose=0)
        q_next    = self.target_model.predict(next_states, verbose=0)

        # build target Q-value matrix
        for i in range(batch_size):
            if dones[i]:
                q_current[i, actions[i]] = rewards[i]
            else:
                q_current[i, actions[i]] = rewards[i] + self.gamma * np.amax(q_next[i])

        # train on batch
        self.model.fit(states, q_current, epochs=1, verbose=0)

        # decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay