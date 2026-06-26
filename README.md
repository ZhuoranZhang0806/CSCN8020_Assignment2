# CSCN8020 Assignment 2 - Q-Learning Taxi Project

**Full Name:** Zhuoran Zhang
**Student ID:** 9048508

## Assignment Title

CSCN8020 Assignment 2 - Reinforcement Learning Programming

## Repository Link

https://github.com/ZhuoranZhang0806/CSCN8020_Assignment2

## Project Summary

This project implements the Q-Learning algorithm using the Gymnasium Taxi-v4 environment. The goal is to train an agent to pick up a passenger from the correct location and drop the passenger off at the correct destination. The project includes environment setup, Q-table initialization, epsilon-greedy action selection, Q-value updates, training loops, logging, metric collection, hyperparameter experiments, plot generation, and final evaluation. The experiments compare different learning rates and exploration factors to observe how they affect training performance.

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/ZhuoranZhang0806/CSCN8020_Assignment2.git
```

2. Open the project folder:

```bash
cd CSCN8020_Assignment2
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Open the Jupyter Notebook:

```bash
jupyter notebook CSCN8020_Assignment2.ipynb
```

5. Run the full Python application from the terminal:

```bash
python src/main.py
```

The main Python code is stored in the `src/` folder. The notebook explains the project, shows the experiment results, and includes Markdown discussion for the Taxi environment, Q-Learning implementation, metrics, plots, and conclusions.

## Major Files in the Repository

| File / Folder                | Description                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `CSCN8020_Assignment2.ipynb` | Main Jupyter Notebook report with explanations, plots, observations, and experiment discussion |
| `src/main.py`                | Main program entry point that runs the full experiment workflow                                |
| `src/taxi_environment.py`    | Creates and describes the Taxi Gymnasium environment                                           |
| `src/td_agents.py`           | Contains the Q-Learning agent and Q-table logic                                                |
| `src/qlearning_trainer.py`   | Contains the Q-Learning training loop and metric collection                                    |
| `src/experiment_runner.py`   | Runs baseline, hyperparameter experiments, best combination, and evaluation                    |
| `src/plot_manager.py`        | Generates training plots and comparison plots                                                  |
| `src/metrics_logger.py`      | Creates the log file and records experiment progress                                           |
| `plots/`                     | Stores generated plots                                                                         |
| `results/`                   | Stores experiment result CSV files                                                             |
| `logs/`                      | Stores execution log files                                                                     |
| `requirements.txt`           | Lists required Python packages                                                                 |
| `.gitignore`                 | Excludes unnecessary cache, virtual environment, and temporary files                           |

## Q-Learning Implementation

The Q-Learning agent uses a Q-table to store the value of each action in each state. The Taxi environment has 500 states and 6 actions, so the Q-table stores values for all state-action pairs.

During training, the agent uses epsilon-greedy action selection. This means the agent sometimes explores by choosing a random action and sometimes exploits by choosing the action with the highest Q-value.

After each step, the agent updates the Q-table using the Q-Learning update equation:

```text
Q(s, a) = Q(s, a) + alpha * [reward + gamma * max(Q(next_state, next_action)) - Q(s, a)]
```

The program records return per episode, steps per episode, average reward per step, evaluation return, evaluation steps, and success rate. Different learning rates and exploration factors are tested and compared with the baseline.

## Cloneable .git URL

```bash
https://github.com/ZhuoranZhang0806/CSCN8020_Assignment2.git
```
