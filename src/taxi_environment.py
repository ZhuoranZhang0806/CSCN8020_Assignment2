import gymnasium as gym


class TaxiEnvironmentManager:
    """Create and describe the Taxi Gymnasium environment."""

    def __init__(self, env_name="Taxi-v4"):
        self.env_name = env_name

    def create_env(self, render_mode=None, remove_time_limit=True):
        """Create Taxi environment."""
        if render_mode is None:
            env = gym.make(self.env_name)
        else:
            env = gym.make(self.env_name, render_mode=render_mode)

        # Remove Gymnasium default TimeLimit wrapper
        if remove_time_limit:
            env = env.unwrapped

        return env

    def get_env_info(self, env):
        """Return state/action information from environment."""

        reward_range = getattr(env, "reward_range", None)

        if reward_range is None and hasattr(env, "unwrapped"):
            reward_range = getattr(env.unwrapped, "reward_range", None)

        if reward_range is None:
            reward_range = "Not directly available"

        return {
            "num_states": env.observation_space.n,
            "num_actions": env.action_space.n,
            "observation_space": env.observation_space,
            "action_space": env.action_space,
            "reward_range": reward_range
        }

    def print_env_info(self, env):
        """Print basic Taxi environment information."""
        info = self.get_env_info(env)

        action_desc = {
            0: "Move south (down)",
            1: "Move north (up)",
            2: "Move east (right)",
            3: "Move west (left)",
            4: "Pickup passenger",
            5: "Drop off passenger"
        }

        print("Observation space:", info["observation_space"])
        print("Observation space size:", info["num_states"])
        print("Reward range:", info["reward_range"])
        print("Number of actions:", info["num_actions"])
        print("Action description:", action_desc)

    def decode_observation(self, obs):
        """Decode Taxi observation number into readable parts."""
        obs = int(obs)

        destination = obs % 4
        obs = obs // 4

        passenger_location = obs % 5
        obs = obs // 5

        taxi_col = obs % 5
        taxi_row = obs // 5

        return {
            "taxi_row": int(taxi_row),
            "taxi_col": int(taxi_col),
            "passenger_location": int(passenger_location),
            "destination": int(destination)
        }

    def describe_observation(self, obs):
        """Return a readable sentence for the current observation."""
        location_desc = {
            0: "Red",
            1: "Green",
            2: "Yellow",
            3: "Blue",
            4: "In taxi"
        }

        obs_dict = self.decode_observation(obs)

        description = (
            "Passenger is at: {0}, wants to go to {1}. "
            "Taxi currently at ({2}, {3})"
        ).format(
            location_desc[obs_dict["passenger_location"]],
            location_desc[obs_dict["destination"]],
            obs_dict["taxi_row"],
            obs_dict["taxi_col"]
        )

        return description