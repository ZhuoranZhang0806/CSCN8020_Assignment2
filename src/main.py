from pathlib import Path
from datetime import datetime

from metrics_logger import setup_logger
from taxi_environment import TaxiEnvironmentManager
from experiment_runner import ExperimentRunner
from plot_manager import PlotManager


def main():
    """Main program for Assignment 2."""

    # Get project root folder
    ROOT_DIR = Path(__file__).resolve().parents[1]

    # Output folders in project root
    LOGS_DIR = ROOT_DIR / "logs"
    RESULTS_DIR = ROOT_DIR / "results"
    PLOTS_DIR = ROOT_DIR / "plots"

    # Create output folders
    LOGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = LOGS_DIR / f"assignment2_execution_{timestamp}.log"

    # Setup logger
    logger = setup_logger(LOG_PATH)

    try:
        # Basic settings
        env_name = "Taxi-v4"
        num_episodes = 5000
        max_steps_per_episode = 10000
        seed = 42

        logger.info("Assignment 2 program started")
        logger.info(f"Project root: {ROOT_DIR}")
        logger.info(f"Environment: {env_name}")
        logger.info(f"Number of episodes: {num_episodes}")
        logger.info(f"Max steps per episode: {max_steps_per_episode}")
        logger.info(f"Log file: {LOG_PATH}")
        logger.info(f"Results folder: {RESULTS_DIR}")
        logger.info(f"Plots folder: {PLOTS_DIR}")

        # Create and check Taxi environment
        env_manager = TaxiEnvironmentManager(env_name=env_name)
        env = env_manager.create_env()

        print("Taxi Environment Information:")
        env_manager.print_env_info(env)

        state, _ = env.reset()
        print("\nExample initial state:")
        print("State number:", state)
        print(env_manager.describe_observation(state))

        env.close()

        # Run all required experiments
        runner = ExperimentRunner(
            env_name=env_name,
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            results_dir=RESULTS_DIR,
            seed=seed,
            logger=logger
        )

        print("\nRunning required experiments...")
        logger.info("Running required experiments")

        all_results, summary_df = runner.run_required_experiments()

        print("\nExperiment Summary:")
        print(summary_df)

        logger.info("Experiment summary:")
        logger.info("\n%s" % summary_df.to_string())

        # Create plots
        plot_manager = PlotManager(plots_dir=PLOTS_DIR)

        print("\nCreating individual plots...")
        logger.info("Creating individual plots")
        plot_manager.plot_training_curves(all_results)

        print("\nCreating comparison bar plots...")
        logger.info("Creating comparison bar plots")
        plot_manager.plot_comparison(summary_df)

        print("\nCreating combined curve plots...")
        logger.info("Creating combined curve plots")
        plot_manager.plot_combined_curves(all_results, window=100)

        # Run best combination experiment
        print("\nRunning best combination experiment...")
        logger.info("Running best combination experiment")

        best_result = runner.run_best_combination_experiment(summary_df)

        print("\nBest Combination Result:")
        print("Experiment:", best_result["experiment_name"])
        print("Alpha:", best_result["alpha"])
        print("Epsilon:", best_result["epsilon"])
        print("Gamma:", best_result["gamma"])
        print("Average steps:", best_result["average_steps"])
        print("Average return:", best_result["average_return"])
        print("Evaluation average steps:", best_result["evaluation_average_steps"])
        print("Evaluation average return:", best_result["evaluation_average_return"])
        print("Evaluation success rate:", best_result["evaluation_success_rate"])

        logger.info("Best Combination Result:")
        logger.info(f"Experiment: {best_result['experiment_name']}")
        logger.info(f"Alpha: {best_result['alpha']}")
        logger.info(f"Epsilon: {best_result['epsilon']}")
        logger.info(f"Gamma: {best_result['gamma']}")
        logger.info(f"Average steps: {best_result['average_steps']}")
        logger.info(f"Average return: {best_result['average_return']}")
        logger.info(f"Evaluation average steps: {best_result['evaluation_average_steps']}")
        logger.info(f"Evaluation average return: {best_result['evaluation_average_return']}")
        logger.info(f"Evaluation success rate: {best_result['evaluation_success_rate']}")

        # Plot best combination result
        print("\nCreating best combination plots...")
        logger.info("Creating best combination plots")

        plot_manager.plot_steps(best_result)
        plot_manager.plot_returns(best_result)
        plot_manager.plot_average_reward_per_step(best_result)

        logger.info("Assignment 2 program finished")

        print("\nProgram finished.")
        print("Log saved to:", LOG_PATH)
        print("Results saved to:", RESULTS_DIR)
        print("Plots saved to:", PLOTS_DIR)

    except Exception:
        logger.error("Program stopped because of an error", exc_info=True)
        raise


if __name__ == "__main__":
    main()