from pathlib import Path

import numpy as np
import pandas as pd

from td_agents import QLAgent
from taxi_environment import TaxiEnvironmentManager
from qlearning_trainer import QLearningTrainer


class ExperimentRunner:
    """Run baseline, parameter experiments, evaluation, and save results."""

    def __init__(
        self,
        env_name="Taxi-v4",
        num_episodes=5000,
        max_steps_per_episode=10000,
        results_dir="results",
        seed=42,
        logger=None
    ):
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.results_dir = Path(results_dir)
        self.seed = seed
        self.logger = logger

        self.results_dir.mkdir(exist_ok=True)

        self.env_manager = TaxiEnvironmentManager(env_name)
        self.trainer = QLearningTrainer(max_steps_per_episode)

    def calculate_average_reward_per_step(self, returns, steps):
        """Calculate average reward per step for each episode."""
        avg_reward_per_step = []

        for i in range(len(returns)):
            if steps[i] != 0:
                avg_reward_per_step.append(returns[i] / steps[i])
            else:
                avg_reward_per_step.append(0)

        return avg_reward_per_step

    def run_one_experiment(self, experiment_name, alpha, epsilon, gamma=0.9):
        """Run one Q-Learning experiment."""
        env = self.env_manager.create_env()
        info = self.env_manager.get_env_info(env)

        agent = QLAgent(
            num_states=info["num_states"],
            num_actions=info["num_actions"],
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            seed=self.seed
        )

        if self.logger is not None:
            self.logger.info(f"Running experiment: {experiment_name}")

        result = self.trainer.train(
            env=env,
            agent=agent,
            num_episodes=self.num_episodes,
            experiment_name=experiment_name,
            logger=self.logger
        )

        # Calculate average reward per step for training episodes
        avg_reward_per_step = self.calculate_average_reward_per_step(
            result["returns_per_episode"],
            result["steps_per_episode"]
        )

        result["average_reward_per_step_per_episode"] = avg_reward_per_step
        result["average_reward_per_step"] = float(np.mean(avg_reward_per_step))

        evaluation = self.evaluate(agent)

        result["alpha"] = alpha
        result["epsilon"] = epsilon
        result["gamma"] = gamma
        result["evaluation_average_steps"] = evaluation["evaluation_average_steps"]
        result["evaluation_average_return"] = evaluation["evaluation_average_return"]
        result["evaluation_success_rate"] = evaluation["evaluation_success_rate"]
        result["evaluation_average_reward_per_step"] = evaluation["evaluation_average_reward_per_step"]

        self.save_episode_results(result)
        env.close()

        return result

    def run_required_experiments(self):
        """Run all required baseline and hyperparameter experiments."""
        experiments = [
            ("baseline", 0.1, 0.1, 0.9),
            ("alpha_0.01", 0.01, 0.1, 0.9),
            ("alpha_0.001", 0.001, 0.1, 0.9),
            ("alpha_0.2", 0.2, 0.1, 0.9),
            ("epsilon_0.2", 0.1, 0.2, 0.9),
            ("epsilon_0.3", 0.1, 0.3, 0.9),
        ]

        all_results = {}
        summary_rows = []

        for experiment_name, alpha, epsilon, gamma in experiments:
            result = self.run_one_experiment(
                experiment_name=experiment_name,
                alpha=alpha,
                epsilon=epsilon,
                gamma=gamma
            )

            all_results[experiment_name] = result

            summary_rows.append({
                "experiment_name": experiment_name,
                "alpha": alpha,
                "epsilon": epsilon,
                "gamma": gamma,
                "training_average_steps": result["average_steps"],
                "training_average_return": result["average_return"],
                "training_average_reward_per_step": result["average_reward_per_step"],
                "evaluation_average_steps": result["evaluation_average_steps"],
                "evaluation_average_return": result["evaluation_average_return"],
                "evaluation_average_reward_per_step": result["evaluation_average_reward_per_step"],
                "evaluation_success_rate": result["evaluation_success_rate"]
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.results_dir / "summary.csv", index=False)

        if self.logger is not None:
            self.logger.info("All required experiments finished")
            self.logger.info("\n%s" % summary_df.to_string())

        return all_results, summary_df

    def evaluate(self, agent, num_eval_episodes=100):
        """Evaluate trained agent without exploration."""
        env = self.env_manager.create_env()

        eval_steps = []
        eval_returns = []

        for _ in range(num_eval_episodes):
            state, _ = env.reset()

            total_reward = 0
            steps = 0
            done = False

            while not done and steps < self.max_steps_per_episode:
                action = agent.select_action(state, training=False)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                steps += 1
                state = next_state

            eval_steps.append(steps)
            eval_returns.append(total_reward)

        env.close()

        eval_avg_reward_per_step = self.calculate_average_reward_per_step(
            eval_returns,
            eval_steps
        )

        return {
            "evaluation_average_steps": float(np.mean(eval_steps)),
            "evaluation_average_return": float(np.mean(eval_returns)),
            "evaluation_average_reward_per_step": float(np.mean(eval_avg_reward_per_step)),
            "evaluation_success_rate": float(np.mean([r > 0 for r in eval_returns]))
        }

    def save_episode_results(self, result):
        """Save episode results to CSV file."""
        df = pd.DataFrame({
            "episode": result["episodes"],
            "steps": result["steps_per_episode"],
            "return": result["returns_per_episode"],
            "average_reward_per_step": result["average_reward_per_step_per_episode"]
        })

        file_path = self.results_dir / f"{result['experiment_name']}.csv"
        df.to_csv(file_path, index=False)

    def run_best_combination_experiment(self, summary_df):
        """Select and re-run the best hyperparameter combination."""
        best_row = summary_df.sort_values(
            by=["evaluation_average_return", "evaluation_average_steps"],
            ascending=[False, True]
        ).iloc[0]

        best_alpha = float(best_row["alpha"])
        best_epsilon = float(best_row["epsilon"])
        best_gamma = float(best_row["gamma"])

        if self.logger is not None:
            self.logger.info("Selected best combination")
            self.logger.info(
                f"alpha={best_alpha}, epsilon={best_epsilon}, gamma={best_gamma}"
            )

        best_result = self.run_one_experiment(
            experiment_name="best_combination",
            alpha=best_alpha,
            epsilon=best_epsilon,
            gamma=best_gamma
        )

        return best_result