from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environments.maze import ACTION_NAMES, MazeEnv


State = tuple[int, int, int, int]


class QLearningAgent:
    """Tabular off-policy Q-Learning agent."""

    def __init__(
        self,
        environment: MazeEnv,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 800,
        epsilon_schedule: str = "linear",
        seed: int = 8,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(
                "alpha must be in the interval (0, 1]."
            )

        if not 0 <= gamma < 1:
            raise ValueError(
                "gamma must be in the interval [0, 1)."
            )

        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError(
                "epsilon values must satisfy "
                "0 <= end <= start <= 1."
            )

        if epsilon_decay_episodes <= 0:
            raise ValueError(
                "epsilon_decay_episodes must be positive."
            )

        if epsilon_schedule not in {
            "linear",
            "exponential",
        }:
            raise ValueError(
                "epsilon_schedule must be "
                "'linear' or 'exponential'."
            )

        self.environment = environment

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = (
            epsilon_decay_episodes
        )
        self.epsilon_schedule = epsilon_schedule

        self.seed = seed
        self.random_generator = random.Random(seed)

        self.states = environment.get_all_states()

        self.q_table: dict[State, list[float]] = {
            state: [0.0 for _ in ACTION_NAMES]
            for state in self.states
        }

        self.training_history: list[
            dict[str, Any]
        ] = []

        self.first_update_record: (
            dict[str, Any] | None
        ) = None

    def epsilon_for_episode(
        self,
        episode: int,
    ) -> float:
        """Calculate epsilon using the selected schedule."""

        if self.epsilon_schedule == "linear":
            fraction = min(
                episode / self.epsilon_decay_episodes,
                1.0,
            )

            epsilon = (
                self.epsilon_start
                + fraction
                * (
                    self.epsilon_end
                    - self.epsilon_start
                )
            )

            return max(
                self.epsilon_end,
                epsilon,
            )

        decay_rate = (
            self.epsilon_end
            / self.epsilon_start
        ) ** (
            1.0 / self.epsilon_decay_episodes
        )

        epsilon = (
            self.epsilon_start
            * decay_rate**episode
        )

        return max(
            self.epsilon_end,
            epsilon,
        )

    def greedy_actions(
        self,
        state: State,
    ) -> list[int]:
        """Return all actions tied for maximum Q."""

        action_values = self.q_table[state]
        maximum_value = max(action_values)

        return [
            action
            for action, value in enumerate(
                action_values
            )
            if math.isclose(
                value,
                maximum_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

    def select_action(
        self,
        state: State,
        epsilon: float,
    ) -> int:
        """Select an action using epsilon-greedy."""

        if self.random_generator.random() < epsilon:
            return self.random_generator.choice(
                list(ACTION_NAMES)
            )

        return self.random_generator.choice(
            self.greedy_actions(state)
        )

    def update_q_value(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        """Apply one real Q-Learning update."""

        old_q_value = self.q_table[state][action]

        if terminated or truncated:
            next_maximum_q = 0.0
        else:
            next_maximum_q = max(
                self.q_table[next_state]
            )

        target = (
            reward
            + self.gamma * next_maximum_q
        )

        td_error = target - old_q_value

        new_q_value = (
            old_q_value
            + self.alpha * td_error
        )

        self.q_table[state][action] = new_q_value

        update_record = {
            "state": list(state),
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": reward,
            "next_state": list(next_state),
            "terminated": terminated,
            "truncated": truncated,
            "old_q": old_q_value,
            "next_max_q": next_maximum_q,
            "target": target,
            "td_error": td_error,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "new_q": new_q_value,
        }

        if self.first_update_record is None:
            self.first_update_record = (
                update_record.copy()
            )

        return update_record

    def train(
        self,
        episodes: int,
    ) -> list[dict[str, Any]]:
        """Train the agent for a number of episodes."""

        if episodes <= 0:
            raise ValueError(
                "episodes must be positive."
            )

        self.training_history = []
        self.first_update_record = None
        self.training_state_visits = {}
        
        for episode in range(episodes):
            epsilon = self.epsilon_for_episode(
                episode
            )

            state, _ = self.environment.reset(
                seed=self.seed + episode
            )
            self.training_state_visits[state] = (
                self.training_state_visits.get(state, 0) + 1
            )
            episode_reward = 0.0
            wall_collisions = 0
            penalty_visits = 0
            closed_door_attempts = 0
            normal_moves = 0
            door_passes = 0
            periodic_gate_closed = 0
            periodic_gate_passes = 0
            goal_reached_count = 0
            step_limit_reached_count = 0
            key_collected = False
            success = False

            terminated = False
            truncated = False

            while not terminated and not truncated:
                action = self.select_action(
                    state,
                    epsilon,
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.environment.step(action)

                self.update_q_value(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=terminated,
                    truncated=truncated,
                )

                events = info["events"]

                wall_collisions += events.count(
                    "wall_collision"
                )

                penalty_visits += events.count(
                    "penalty_cell"
                )

                closed_door_attempts += events.count(
                    "closed_door_attempt"
                )

                normal_moves += events.count(
                    "normal_move"
                )

                door_passes += events.count(
                    "door_passed"
                )

                periodic_gate_closed += events.count(
                    "periodic_gate_closed"
                )

                periodic_gate_passes += events.count(
                    "periodic_gate_passed"
                )

                goal_reached_count += events.count(
                    "goal_reached"
                )

                step_limit_reached_count += events.count(
                    "step_limit_reached"
                )

                if "key_collected" in events:
                    key_collected = True

                if "goal_reached" in events:
                    success = True

                episode_reward += reward

                self.training_state_visits[next_state] = (
                    self.training_state_visits.get(next_state, 0) + 1
                )

                state = next_state

            episode_record = {
                "episode": episode + 1,
                "epsilon": epsilon,
                "reward": episode_reward,
                "steps": self.environment.step_count,
                "success": int(success),
                "normal_moves": normal_moves,
                "wall_collisions": wall_collisions,
                "penalty_visits": penalty_visits,
                "closed_door_attempts":
                    closed_door_attempts,
                "door_passes": door_passes,
                "periodic_gate_closed":
                    periodic_gate_closed,
                "periodic_gate_passes":
                    periodic_gate_passes,
                "key_collected": int(key_collected),
                "goal_reached_count":
                    goal_reached_count,
                "step_limit_reached_count":
                    step_limit_reached_count,
                "terminated": int(terminated),
                "truncated": int(truncated),
            }

            self.training_history.append(
                episode_record
            )

            if (
                episode == 0
                or (episode + 1) % 100 == 0
                or episode + 1 == episodes
            ):
                recent_records = (
                    self.training_history[-100:]
                )

                recent_success_rate = (
                    100.0
                    * sum(
                        record["success"]
                        for record in recent_records
                    )
                    / len(recent_records)
                )

                print(
                    f"Episode {episode + 1:4d} | "
                    f"epsilon={epsilon:.4f} | "
                    f"reward={episode_reward:.2f} | "
                    f"steps={self.environment.step_count} | "
                    f"recent success="
                    f"{recent_success_rate:.1f}%"
                )

        return self.training_history

    def greedy_policy(
        self,
    ) -> dict[State, int | None]:
        """Extract the greedy policy from Q."""

        policy: dict[State, int | None] = {}

        for state in self.states:
            if self.environment.is_terminal_state(state):
                policy[state] = None
            else:
                policy[state] = min(
                    self.greedy_actions(state)
                )

        return policy

    def evaluate(
        self,
        episodes: int = 100,
        seed_offset: int = 10_000,
    ) -> dict[str, float]:
        """Evaluate the greedy policy without exploration."""

        successes = 0
        total_reward = 0.0
        total_steps = 0

        for episode in range(episodes):
            state, _ = self.environment.reset(
                seed=(
                    self.seed
                    + seed_offset
                    + episode
                )
            )

            terminated = False
            truncated = False
            episode_success = False

            while not terminated and not truncated:
                action = self.random_generator.choice(
                    self.greedy_actions(state)
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.environment.step(action)

                total_reward += reward
                state = next_state

                if "goal_reached" in info["events"]:
                    episode_success = True

            successes += int(episode_success)
            total_steps += self.environment.step_count

        return {
            "episodes": float(episodes),
            "success_rate": (
                100.0 * successes / episodes
            ),
            "average_reward": (
                total_reward / episodes
            ),
            "average_steps": (
                total_steps / episodes
            ),
        }

    def save_training_history(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save episode metrics as CSV."""

        if not self.training_history:
            raise RuntimeError(
                "Train the agent before saving history."
            )

        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "results"
                / "raw_data"
                / (
                    "q_learning_"
                    f"{self.epsilon_schedule}_training.csv"
                )
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            self.training_history[0].keys()
        )

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                self.training_history
            )

        return output_path

    def save_model(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save the Q table and configuration."""

        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "results"
                / "models"
                / (
                    "q_learning_"
                    f"{self.epsilon_schedule}.json"
                )
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        q_records = [
            {
                "state": list(state),
                "q_values": values,
                "best_action": (
                    None
                    if self.environment.is_terminal_state(
                        state
                    )
                    else min(
                        self.greedy_actions(state)
                    )
                ),
            }
            for state, values in sorted(
                self.q_table.items()
            )
        ]

        model_data = {
            "algorithm": "Q-Learning",
            "student_id": "40403384",
            "seed": self.seed,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_episodes":
                self.epsilon_decay_episodes,
            "epsilon_schedule":
                self.epsilon_schedule,
            "reward_mode":
                self.environment.reward_mode,
            "number_of_states": len(self.states),
            "q_table": q_records,
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                model_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path

    def save_update_example(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save one real Q update for the report."""

        if self.first_update_record is None:
            raise RuntimeError(
                "No Q update has been recorded."
            )

        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "results"
                / "raw_data"
                / "q_learning_update_example.json"
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.first_update_record,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path


if __name__ == "__main__":
    environment = MazeEnv(
        transition_seed=8,
        reward_mode="shaped",
        gamma=0.95,
    )

    agent = QLearningAgent(
        environment=environment,
        alpha=0.10,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=500,
        epsilon_schedule="linear",
        seed=8,
    )

    agent.train(episodes=600)

    evaluation = agent.evaluate(
        episodes=100
    )

    history_path = (
        agent.save_training_history()
    )

    model_path = agent.save_model()

    update_path = (
        agent.save_update_example()
    )

    print()
    print("Q-Learning training finished.")

    print(
        "Evaluation success rate:",
        f'{evaluation["success_rate"]:.2f}%',
    )

    print(
        "Evaluation average reward:",
        f'{evaluation["average_reward"]:.2f}',
    )

    print(
        "Evaluation average steps:",
        f'{evaluation["average_steps"]:.2f}',
    )

    print("Training CSV saved to:", history_path)
    print("Model saved to:", model_path)
    print("Q update example saved to:", update_path)