from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


class PlotManager:
    """Create and save plots for training and comparison."""

    def __init__(self, plots_dir="plots"):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)

    def moving_average(self, values, window=100):
        """Calculate moving average."""
        return pd.Series(values).rolling(window=window).mean()

    def plot_training_curves(self, all_results):
        """Plot training curves for all experiments."""
        for experiment_name, result in all_results.items():
            print(f"Creating plots for {experiment_name}...")
            self.plot_steps(result)
            self.plot_returns(result)
            self.plot_average_reward_per_step(result)

    def plot_steps(self, result):
        """Plot steps per episode."""
        experiment_name = result["experiment_name"]
        episodes = result["episodes"]
        steps = result["steps_per_episode"]

        save_path = self.plots_dir / f"{experiment_name}_steps.png"

        plt.figure()

        # Raw steps per episode
        plt.plot(
            episodes,
            steps,
            label="Steps per episode",
            alpha=0.35
        )

        # Smoothed steps
        plt.plot(
            episodes,
            self.moving_average(steps),
            label="Moving average steps",
            color="orange",
            linewidth=2
        )

        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title(f"Steps per Episode - {experiment_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_returns(self, result):
        """Plot return per episode and average return over time."""
        experiment_name = result["experiment_name"]
        episodes = result["episodes"]
        returns = result["returns_per_episode"]

        save_path = self.plots_dir / f"{experiment_name}_return.png"

        plt.figure()

        # Raw return per episode
        plt.plot(
            episodes,
            returns,
            label="Return per episode",
            alpha=0.35
        )

        # Average return over time
        plt.plot(
            episodes,
            self.moving_average(returns),
            label="Average return over time",
            color="gold",
            linewidth=2
        )

        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title(f"Return and Average Return Over Time - {experiment_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_average_reward_per_step(self, result):
        """Plot average reward per step for each episode."""
        experiment_name = result["experiment_name"]
        episodes = result["episodes"]
        returns = result["returns_per_episode"]
        steps = result["steps_per_episode"]

        avg_reward_per_step = [
            returns[i] / steps[i] if steps[i] != 0 else 0
            for i in range(len(returns))
        ]

        save_path = self.plots_dir / f"{experiment_name}_average_reward_per_step.png"

        plt.figure()

        # Raw average reward per step
        plt.plot(
            episodes,
            avg_reward_per_step,
            label="Average reward per step",
            alpha=0.35
        )

        # Smoothed average reward per step
        plt.plot(
            episodes,
            self.moving_average(avg_reward_per_step),
            label="Moving average reward per step",
            color="gold",
            linewidth=2
        )

        plt.xlabel("Episode")
        plt.ylabel("Average Reward per Step")
        plt.title(f"Average Reward per Step - {experiment_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_comparison(self, summary_df):
        """Create comparison plots."""
        print("Creating comparison plots...")
        self.plot_average_steps_comparison(summary_df)
        self.plot_average_return_comparison(summary_df)
        self.plot_average_reward_per_step_comparison(summary_df)
        self.plot_success_rate_comparison(summary_df)

    def plot_average_steps_comparison(self, summary_df):
        """Compare evaluation average steps."""
        save_path = self.plots_dir / "comparison_average_steps.png"

        plt.figure()
        plt.bar(summary_df["experiment_name"], summary_df["evaluation_average_steps"])
        plt.xlabel("Experiment")
        plt.ylabel("Evaluation Average Steps")
        plt.title("Evaluation Average Steps Comparison")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_average_return_comparison(self, summary_df):
        """Compare evaluation average return."""
        save_path = self.plots_dir / "comparison_average_return.png"

        plt.figure()
        plt.bar(summary_df["experiment_name"], summary_df["evaluation_average_return"])
        plt.xlabel("Experiment")
        plt.ylabel("Evaluation Average Return")
        plt.title("Evaluation Average Return Comparison")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_average_reward_per_step_comparison(self, summary_df):
        """Compare evaluation average reward per step."""
        save_path = self.plots_dir / "comparison_average_reward_per_step.png"

        plt.figure()
        plt.bar(
            summary_df["experiment_name"],
            summary_df["evaluation_average_reward_per_step"]
        )
        plt.xlabel("Experiment")
        plt.ylabel("Evaluation Average Reward per Step")
        plt.title("Evaluation Average Reward per Step Comparison")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_success_rate_comparison(self, summary_df):
        """Compare evaluation success rate."""
        save_path = self.plots_dir / "comparison_success_rate.png"

        plt.figure()
        plt.bar(summary_df["experiment_name"], summary_df["evaluation_success_rate"])
        plt.xlabel("Experiment")
        plt.ylabel("Success Rate")
        plt.title("Evaluation Success Rate Comparison")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_combined_curves(self, all_results, window=100):
        """Plot all experiments together in combined smoothed charts."""
        print("Creating combined plots...")
        self.plot_combined_smoothed_return(all_results, window)
        self.plot_combined_smoothed_steps(all_results, window)
        self.plot_combined_average_reward_per_step(all_results, window)

    def plot_combined_smoothed_return(self, all_results, window=100):
        """Plot smoothed return of all experiments in one chart."""
        save_path = self.plots_dir / "combined_smoothed_return.png"

        plt.figure()

        for experiment_name, result in all_results.items():
            episodes = result["episodes"]
            returns = result["returns_per_episode"]
            smoothed_returns = self.moving_average(returns, window=window)

            plt.plot(episodes, smoothed_returns, label=experiment_name)

        plt.xlabel("Episode")
        plt.ylabel("Smoothed Return")
        plt.title("Smoothed Return per Episode Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_combined_smoothed_steps(self, all_results, window=100):
        """Plot smoothed steps of all experiments in one chart."""
        save_path = self.plots_dir / "combined_smoothed_steps.png"

        plt.figure()

        for experiment_name, result in all_results.items():
            episodes = result["episodes"]
            steps = result["steps_per_episode"]
            smoothed_steps = self.moving_average(steps, window=window)

            plt.plot(episodes, smoothed_steps, label=experiment_name)

        plt.xlabel("Episode")
        plt.ylabel("Smoothed Steps")
        plt.title("Smoothed Steps per Episode Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    def plot_combined_average_reward_per_step(self, all_results, window=100):
        """Plot smoothed average reward per step of all experiments in one chart."""
        save_path = self.plots_dir / "combined_average_reward_per_step.png"

        plt.figure()

        for experiment_name, result in all_results.items():
            episodes = result["episodes"]
            returns = result["returns_per_episode"]
            steps = result["steps_per_episode"]

            avg_reward_per_step = [
                returns[i] / steps[i] if steps[i] != 0 else 0
                for i in range(len(returns))
            ]

            smoothed_values = self.moving_average(avg_reward_per_step, window=window)

            plt.plot(episodes, smoothed_values, label=experiment_name)

        plt.xlabel("Episode")
        plt.ylabel("Average Reward per Step")
        plt.title("Average Reward per Step Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")