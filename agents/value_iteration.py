from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


# Allow direct execution from the agents directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environments.maze import ACTION_NAMES, MazeEnv


State = tuple[int, int, int, int]


class ValueIterationAgent:
    """Solve the maze MDP using Value Iteration."""

    def __init__(
        self,
        environment: MazeEnv,
        gamma: float = 0.95,
        theta: float = 1e-8,
        max_iterations: int = 10_000,
    ) -> None:
        if not 0 <= gamma < 1:
            raise ValueError(
                "gamma must be between 0 and 1."
            )

        if theta <= 0:
            raise ValueError(
                "theta must be greater than zero."
            )

        self.environment = environment
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.states = environment.get_all_states()

        self.values: dict[State, float] = {
            state: 0.0
            for state in self.states
        }

        self.policy: dict[State, int | None] = {}

        self.delta_history: list[float] = []
        self.iterations = 0
        self.runtime_seconds = 0.0
        self.converged = False

    def action_value(
        self,
        state: State,
        action: int,
        value_table: dict[State, float] | None = None,
    ) -> float:
        """Calculate the Bellman value of one action."""

        if value_table is None:
            value_table = self.values

        outcomes = self.environment.transition_outcomes(
            state,
            action,
        )

        total_value = 0.0

        for outcome in outcomes:
            probability = float(
                outcome["probability"]
            )

            next_state = outcome["next_state"]
            reward = float(outcome["reward"])
            terminated = bool(outcome["terminated"])

            if terminated:
                future_value = 0.0
            else:
                future_value = value_table[next_state]

            total_value += probability * (
                reward
                + self.gamma * future_value
            )

        return total_value

    def run(self) -> dict[str, Any]:
        """Run synchronous Value Iteration."""

        start_time = time.perf_counter()

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            new_values: dict[State, float] = {}
            maximum_change = 0.0

            for state in self.states:
                if self.environment.is_terminal_state(state):
                    new_values[state] = 0.0
                    continue

                action_values = [
                    self.action_value(
                        state,
                        action,
                        self.values,
                    )
                    for action in ACTION_NAMES
                ]

                best_value = max(action_values)
                new_values[state] = best_value

                state_change = abs(
                    best_value
                    - self.values[state]
                )

                maximum_change = max(
                    maximum_change,
                    state_change,
                )

            self.values = new_values
            self.delta_history.append(maximum_change)
            self.iterations = iteration

            if maximum_change < self.theta:
                self.converged = True
                break

        self.runtime_seconds = (
            time.perf_counter() - start_time
        )

        self.policy = self.extract_policy()

        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "final_delta": self.delta_history[-1],
        }

    def extract_policy(
        self,
    ) -> dict[State, int | None]:
        """Extract the greedy policy from the value table."""

        policy: dict[State, int | None] = {}

        for state in self.states:
            if self.environment.is_terminal_state(state):
                policy[state] = None
                continue

            action_values = {
                action: self.action_value(
                    state,
                    action,
                    self.values,
                )
                for action in ACTION_NAMES
            }

            best_action = max(
                action_values,
                key=action_values.get,
            )

            policy[state] = best_action

        return policy

    def save_results(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        """Save values, policy and convergence information."""

        if not self.policy:
            raise RuntimeError(
                "Run Value Iteration before saving."
            )

        if output_path is None:
            gamma_name = str(self.gamma).replace(
                ".",
                "_",
            )

            output_path = (
                PROJECT_ROOT
                / "results"
                / "models"
                / f"value_iteration_gamma_{gamma_name}.json"
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state_records: list[dict[str, Any]] = []

        for state in sorted(self.states):
            action = self.policy[state]

            state_records.append(
                {
                    "state": list(state),
                    "value": self.values[state],
                    "best_action": action,
                    "best_action_name": (
                        ACTION_NAMES[action]
                        if action is not None
                        else "TERMINAL"
                    ),
                }
            )

        result_data = {
            "algorithm": "Value Iteration",
            "gamma": self.gamma,
            "theta": self.theta,
            "max_iterations": self.max_iterations,
            "converged": self.converged,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "number_of_states": len(self.states),
            "delta_history": self.delta_history,
            "states": state_records,
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path


if __name__ == "__main__":
    environment = MazeEnv(
        transition_seed=8,
        reward_mode="sparse",
        gamma=0.95,
    )

    agent = ValueIterationAgent(
        environment=environment,
        gamma=0.95,
        theta=1e-8,
        max_iterations=10_000,
    )

    summary = agent.run()

    initial_state, _ = environment.reset(seed=8)
    initial_action = agent.policy[initial_state]

    saved_path = agent.save_results()

    print("Value Iteration finished.")
    print("Converged:", summary["converged"])
    print("Iterations:", summary["iterations"])

    print(
        "Runtime:",
        f'{summary["runtime_seconds"]:.4f} seconds',
    )

    print(
        "Final delta:",
        f'{summary["final_delta"]:.12f}',
    )

    print(
        "Number of states:",
        len(agent.states),
    )

    print(
        "Initial state value:",
        f"{agent.values[initial_state]:.4f}",
    )

    print(
        "Best initial action:",
        ACTION_NAMES[initial_action],
    )

    print("Results saved to:", saved_path)