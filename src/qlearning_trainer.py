import numpy as np


class QLearningTrainer:
    """Run Q-Learning training loop and collect metrics."""

    def __init__(self, max_steps_per_episode=200):
        self.max_steps_per_episode = max_steps_per_episode

    def train(self, env, agent, num_episodes, experiment_name="experiment", logger=None):
        """Train agent and return steps/returns for each episode."""
        steps_per_episode = []
        returns_per_episode = []

        if logger is not None:
            logger.info("=" * 60)
            logger.info(f"Start training experiment: {experiment_name}")
            logger.info(
                f"Hyperparameters: alpha={agent.alpha}, "
                f"gamma={agent.gamma}, epsilon={agent.epsilon}"
            )
            logger.info(f"Number of episodes: {num_episodes}")

        for episode in range(num_episodes):
            state, _ = env.reset()

            total_reward = 0
            steps = 0
            done = False

            while not done and steps < self.max_steps_per_episode:
                action = agent.select_action(state, training=True)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.update_q_values(
                    state,
                    action,
                    reward,
                    next_state
                )

                total_reward += reward
                steps += 1
                state = next_state

            steps_per_episode.append(steps)
            returns_per_episode.append(total_reward)

            # Log progress every 500 episodes
            if episode % 500 == 0:
                print("Episode: %d" % episode)

                if logger is not None:
                    logger.info(
                        f"Episode {episode}: "
                        f"steps={steps}, return={total_reward}"
                    )

        average_steps = np.mean(steps_per_episode)
        average_return = np.mean(returns_per_episode)

        if logger is not None:
            logger.info(f"Finished experiment: {experiment_name}")
            logger.info(f"Average steps: {average_steps}")
            logger.info(f"Average return: {average_return}")
            logger.info("=" * 60)

        result = {
            "experiment_name": experiment_name,
            "episodes": list(range(1, num_episodes + 1)),
            "steps_per_episode": steps_per_episode,
            "returns_per_episode": returns_per_episode,
            "average_steps": float(average_steps),
            "average_return": float(average_return)
        }

        return result