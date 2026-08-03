from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.value_iteration import ValueIterationAgent
from environments.maze import ACTION_NAMES, MazeEnv


GAMMA_VALUES = [0.80, 0.90, 0.95]
THETA = 1e-8
MAX_ITERATIONS = 10_000


def run_experiments() -> list[dict[str, Any]]:
    """Run Value Iteration for different discount factors."""

    experiment_results: list[dict[str, Any]] = []
    policies: dict[float, dict] = {}

    for gamma in GAMMA_VALUES:
        print()
        print("=" * 55)
        print(f"Running Value Iteration with gamma={gamma}")

        environment = MazeEnv(
            transition_seed=8,
            reward_mode="sparse",
            gamma=gamma,
        )

        agent = ValueIterationAgent(
            environment=environment,
            gamma=gamma,
            theta=THETA,
            max_iterations=MAX_ITERATIONS,
        )

        summary = agent.run()

        initial_state, _ = environment.reset(seed=8)
        initial_action = agent.policy[initial_state]

        model_path = agent.save_results()

        policies[gamma] = agent.policy.copy()

        result = {
            "gamma": gamma,
            "theta": THETA,
            "converged": summary["converged"],
            "iterations": summary["iterations"],
            "runtime_seconds": summary["runtime_seconds"],
            "final_delta": summary["final_delta"],
            "number_of_states": len(agent.states),
            "initial_state_value": agent.values[initial_state],
            "initial_best_action": ACTION_NAMES[initial_action],
            "model_path": str(
                model_path.relative_to(PROJECT_ROOT)
            ),
        }

        experiment_results.append(result)

        print("Converged:", result["converged"])
        print("Iterations:", result["iterations"])

        print(
            "Runtime:",
            f'{result["runtime_seconds"]:.4f} seconds',
        )

        print(
            "Initial state value:",
            f'{result["initial_state_value"]:.4f}',
        )

        print(
            "Initial best action:",
            result["initial_best_action"],
        )

    reference_gamma = 0.95
    reference_policy = policies[reference_gamma]

    for result in experiment_results:
        gamma = result["gamma"]
        policy = policies[gamma]

        comparable_states = [
            state
            for state, reference_action
            in reference_policy.items()
            if reference_action is not None
        ]

        matching_actions = sum(
            policy[state] == reference_policy[state]
            for state in comparable_states
        )

        agreement_percentage = (
            100.0
            * matching_actions
            / len(comparable_states)
        )

        result["policy_agreement_with_gamma_0_95"] = (
            agreement_percentage
        )

    return experiment_results


def save_csv(
    results: list[dict[str, Any]],
) -> Path:
    """Save the experiment summary in CSV format."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "value_iteration_gamma_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "gamma",
        "theta",
        "converged",
        "iterations",
        "runtime_seconds",
        "final_delta",
        "number_of_states",
        "initial_state_value",
        "initial_best_action",
        "policy_agreement_with_gamma_0_95",
        "model_path",
    ]

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
        writer.writerows(results)

    return output_path


def save_json(
    results: list[dict[str, Any]],
) -> Path:
    """Save experiment settings and summary in JSON."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "value_iteration_gamma_comparison.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_data = {
        "algorithm": "Value Iteration",
        "student_id": "40403384",
        "seed": 8,
        "reward_mode": "sparse",
        "gamma_values": GAMMA_VALUES,
        "theta": THETA,
        "max_iterations": MAX_ITERATIONS,
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


if __name__ == "__main__":
    results = run_experiments()

    csv_path = save_csv(results)
    json_path = save_json(results)

    print()
    print("=" * 55)
    print("All Value Iteration experiments finished.")
    print("CSV saved to:", csv_path)
    print("JSON saved to:", json_path)

    print()
    print("Summary:")

    for result in results:
        print(
            f'gamma={result["gamma"]:.2f} | '
            f'iterations={result["iterations"]} | '
            f'runtime={result["runtime_seconds"]:.4f}s | '
            f'agreement={result["policy_agreement_with_gamma_0_95"]:.2f}%'
        )