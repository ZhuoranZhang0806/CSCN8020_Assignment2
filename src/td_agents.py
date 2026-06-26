import numpy as np
import time


class TDAgent:
    """Base TD agent. It stores Q-table and common action selection methods."""

    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration factor
        self.rng = np.random.default_rng(seed)

        self.action_desc = [
            "Move south (down)",
            "Move north (up)",
            "Move east (right)",
            "Move west (left)",
            "Pickup passenger",
            "Drop off passenger"
        ]

        # Initialize Q-values for all state-action pairs
        self.q_values = np.zeros((num_states, num_actions))

    def select_action(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.num_actions))
        else:
            return self.select_greedy_action(state)

    def select_greedy_action(self, state):
        """Choose the action with the highest Q-value."""
        q_values_state = self.q_values[int(state)]
        return int(np.argmax(q_values_state))

    def update_q_values(self, state, action, reward, next_state, next_action=None):
        """Update Q-values. Child classes implement the real update rule."""
        pass

    def get_q_values(self):
        """Return the Q-table."""
        return self.q_values

    def test(self, env, num_episodes=1, verbose=False):
        """Run the trained agent with greedy policy."""
        for _ in range(num_episodes):
            done = False
            state, _ = env.reset()
            env.render()

            while not done:
                action = self.select_greedy_action(state)

                env.render()
                time.sleep(0.1)

                if verbose:
                    print("Moving: %s" % self.action_desc[action])

                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state

            time.sleep(1.0)


class SarsaAgent(TDAgent):
    """SARSA agent. Kept as course reference."""

    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon, seed)

    def update_q_values(self, state, action, reward, next_state, next_action=None):
        """Apply SARSA update rule."""
        current_q_value = self.q_values[int(state), int(action)]
        next_q_value = self.q_values[int(next_state), int(next_action)]
        target = reward + self.gamma * next_q_value
        self.q_values[int(state), int(action)] += self.alpha * (target - current_q_value)


class QLAgent(TDAgent):
    """Q-Learning agent used for this assignment."""

    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None):
        super().__init__(num_states, num_actions, alpha, gamma, epsilon, seed)

    def update_q_values(self, state, action, reward, next_state, next_action=None):
        """Apply Q-Learning update rule."""
        current_q_value = self.q_values[int(state), int(action)]
        max_next_q_value = np.max(self.q_values[int(next_state)])
        target = reward + self.gamma * max_next_q_value
        self.q_values[int(state), int(action)] += self.alpha * (target - current_q_value)