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
TraceKey = tuple[State, int]


class SARSALambdaAgent:
    """Tabular on-policy SARSA(lambda) with replacing traces."""

    def __init__(
        self,
        environment: MazeEnv,
        alpha: float = 0.10,
        gamma: float = 0.95,
        lambda_value: float = 0.70,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
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

        if not 0 <= lambda_value <= 1:
            raise ValueError(
                "lambda_value must be between 0 and 1."
            )

        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError(
                "Invalid epsilon values."
            )

        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_value = lambda_value

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = (
            epsilon_decay_episodes
        )

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

        self.trace_example: list[
            dict[str, Any]
        ] = []

    def epsilon_for_episode(
        self,
        episode: int,
    ) -> float:
        """Return linearly decayed epsilon."""

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

    def greedy_actions(
        self,
        state: State,
    ) -> list[int]:
        """Return all actions tied for the largest Q value."""

        values = self.q_table[state]
        maximum_value = max(values)

        return [
            action
            for action, value in enumerate(values)
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
        """Select an epsilon-greedy action."""

        if self.random_generator.random() < epsilon:
            return self.random_generator.choice(
                list(ACTION_NAMES)
            )

        return self.random_generator.choice(
            self.greedy_actions(state)
        )

    def train(
        self,
        episodes: int,
    ) -> list[dict[str, Any]]:
        """Train with replacing eligibility traces."""

        if episodes <= 0:
            raise ValueError(
                "episodes must be positive."
            )

        self.training_history = []
        self.trace_example = []

        for episode in range(episodes):
            epsilon = self.epsilon_for_episode(
                episode
            )

            state, _ = self.environment.reset(
                seed=self.seed + episode
            )

            action = self.select_action(
                state,
                epsilon,
            )

            traces: dict[TraceKey, float] = {}

            episode_reward = 0.0
            wall_collisions = 0
            penalty_visits = 0
            closed_door_attempts = 0
            key_collected = False
            success = False

            terminated = False
            truncated = False

            episode_step = 0

            while not terminated and not truncated:
                episode_step += 1

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.environment.step(action)

                if terminated or truncated:
                    next_action = None
                    next_q_value = 0.0
                else:
                    next_action = self.select_action(
                        next_state,
                        epsilon,
                    )

                    next_q_value = self.q_table[
                        next_state
                    ][next_action]

                current_q_value = self.q_table[
                    state
                ][action]

                td_error = (
                    reward
                    + self.gamma * next_q_value
                    - current_q_value
                )

                # Replacing trace:
                # The current state-action trace becomes 1.
                traces[(state, action)] = 1.0

                traces_before_decay = {
                    key: value
                    for key, value in traces.items()
                }

                for (
                    trace_state,
                    trace_action,
                ), eligibility in list(
                    traces.items()
                ):
                    self.q_table[
                        trace_state
                    ][trace_action] += (
                        self.alpha
                        * td_error
                        * eligibility
                    )

                    new_eligibility = (
                        self.gamma
                        * self.lambda_value
                        * eligibility
                    )

                    if new_eligibility < 1e-10:
                        del traces[
                            (
                                trace_state,
                                trace_action,
                            )
                        ]
                    else:
                        traces[
                            (
                                trace_state,
                                trace_action,
                            )
                        ] = new_eligibility

                # Save several consecutive steps
                # for the analytical report.
                if episode == 0 and episode_step <= 12:
                    active_trace_records = []

                    for (
                        trace_state,
                        trace_action,
                    ), eligibility in sorted(
                        traces_before_decay.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:8]:
                        active_trace_records.append(
                            {
                                "state": list(
                                    trace_state
                                ),
                                "action": trace_action,
                                "action_name":
                                    ACTION_NAMES[
                                        trace_action
                                    ],
                                "eligibility_before_decay":
                                    eligibility,
                                "eligibility_after_decay":
                                    traces.get(
                                        (
                                            trace_state,
                                            trace_action,
                                        ),
                                        0.0,
                                    ),
                            }
                        )

                    self.trace_example.append(
                        {
                            "episode": episode + 1,
                            "step": episode_step,
                            "state": list(state),
                            "action": action,
                            "action_name":
                                ACTION_NAMES[action],
                            "reward": reward,
                            "next_state": list(
                                next_state
                            ),
                            "next_action":
                                next_action,
                            "next_action_name": (
                                ACTION_NAMES[
                                    next_action
                                ]
                                if next_action
                                is not None
                                else "TERMINAL"
                            ),
                            "old_q": current_q_value,
                            "next_q": next_q_value,
                            "td_error_delta":
                                td_error,
                            "active_trace_count":
                                len(traces),
                            "traces":
                                active_trace_records,
                        }
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

                if "key_collected" in events:
                    key_collected = True

                if "goal_reached" in events:
                    success = True

                episode_reward += reward

                if not terminated and not truncated:
                    state = next_state

                    if next_action is None:
                        raise RuntimeError(
                            "next_action is unexpectedly None."
                        )

                    action = next_action

            episode_record = {
                "episode": episode + 1,
                "lambda": self.lambda_value,
                "epsilon": epsilon,
                "reward": episode_reward,
                "steps": self.environment.step_count,
                "success": int(success),
                "wall_collisions": wall_collisions,
                "penalty_visits": penalty_visits,
                "closed_door_attempts":
                    closed_door_attempts,
                "key_collected": int(key_collected),
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
                    f"lambda={self.lambda_value:.1f} | "
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
        """Extract the greedy policy."""

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
        seed_offset: int = 20_000,
    ) -> dict[str, float]:
        """Evaluate the learned greedy policy."""

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
        """Save episode statistics as CSV."""

        if not self.training_history:
            raise RuntimeError(
                "Train the agent before saving history."
            )

        lambda_name = str(
            self.lambda_value
        ).replace(".", "_")

        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "results"
                / "raw_data"
                / (
                    "sarsa_lambda_"
                    f"{lambda_name}_training.csv"
                )
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    self.training_history[0].keys()
                ),
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
        """Save Q table and configuration."""

        lambda_name = str(
            self.lambda_value
        ).replace(".", "_")

        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "results"
                / "models"
                / (
                    "sarsa_lambda_"
                    f"{lambda_name}.json"
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
            "algorithm": "SARSA(lambda)",
            "trace_type": "replacing",
            "student_id": "40403384",
            "seed": self.seed,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda": self.lambda_value,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_episodes":
                self.epsilon_decay_episodes,
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

    def save_trace_example(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save consecutive delta and trace changes."""

        if not self.trace_example:
            raise RuntimeError(
                "No trace example was recorded."
            )

        lambda_name = str(
            self.lambda_value
        ).replace(".", "_")

        if output_path is None:
            output_path = (
                PROJECT_ROOT
                / "results"
                / "raw_data"
                / (
                    "sarsa_lambda_"
                    f"{lambda_name}_trace_example.json"
                )
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
                self.trace_example,
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

    agent = SARSALambdaAgent(
        environment=environment,
        alpha=0.10,
        gamma=0.95,
        lambda_value=0.70,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=500,
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

    trace_path = (
        agent.save_trace_example()
    )

    print()
    print("SARSA(lambda) training finished.")

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
    print("Trace example saved to:", trace_path)