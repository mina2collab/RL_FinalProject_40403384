from __future__ import annotations

import csv
import gc
import sys
import tracemalloc
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SARSALambdaAgent
from agents.value_iteration import ValueIterationAgent
from environments.maze import MazeEnv


OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "algorithm_memory_comparison.csv"
)
def measure_peak_memory(run_function):
    """Measure peak Python memory during one algorithm run."""

    gc.collect()
    tracemalloc.start()

    try:
        result = run_function()

        current_bytes, peak_bytes = (
            tracemalloc.get_traced_memory()
        )
    finally:
        tracemalloc.stop()

    return {
        "result": result,
        "current_memory_kb": (
            current_bytes / 1024
        ),
        "peak_memory_kb": (
            peak_bytes / 1024
        ),
    }
def run_value_iteration():
    """Run Value Iteration for memory measurement."""

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

    agent.run()

    return {
        "states": len(agent.states),
        "value_entries": len(agent.values),
        "policy_entries": len(agent.policy),
    }
def run_q_learning():
    """Run Q-Learning for memory measurement."""

    environment = MazeEnv(
        transition_seed=8,
        reward_mode="sparse",
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

    agent.train(
        episodes=600
    )

    return {
        "states": len(agent.states),
        "q_entries": (
            len(agent.q_table) * 4
        ),
    }
def run_sarsa_lambda():
    """Run SARSA(lambda=0.3) for memory measurement."""

    environment = MazeEnv(
        transition_seed=8,
        reward_mode="sparse",
        gamma=0.95,
    )

    agent = SARSALambdaAgent(
        environment=environment,
        alpha=0.10,
        gamma=0.95,
        lambda_value=0.3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=500,
        seed=8,
    )

    agent.train(
        episodes=600
    )

    return {
        "states": len(agent.states),
        "q_entries": (
            len(agent.q_table) * 4
        ),
        "lambda": 0.3,
    }
def main() -> None:
    """Measure and save peak memory for all algorithms."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiments = [
        (
            "Value Iteration",
            run_value_iteration,
        ),
        (
            "Q-Learning",
            run_q_learning,
        ),
        (
            "SARSA(lambda=0.3)",
            run_sarsa_lambda,
        ),
    ]

    rows = []

    for algorithm_name, run_function in experiments:
        print()
        print("=" * 60)
        print(
            f"Measuring memory: {algorithm_name}"
        )

        measurement = measure_peak_memory(
            run_function
        )

        result = measurement["result"]

        peak_memory_kb = measurement[
            "peak_memory_kb"
        ]

        current_memory_kb = measurement[
            "current_memory_kb"
        ]

        row = {
            "algorithm": algorithm_name,
            "states": result["states"],
            "value_entries": result.get(
                "value_entries",
                0,
            ),
            "policy_entries": result.get(
                "policy_entries",
                0,
            ),
            "q_entries": result.get(
                "q_entries",
                0,
            ),
            "lambda": result.get(
                "lambda",
                "",
            ),
            "current_memory_kb":
                current_memory_kb,
            "peak_memory_kb":
                peak_memory_kb,
            "peak_memory_mb":
                peak_memory_kb / 1024,
        }

        rows.append(row)

        print(
            "Peak memory:",
            f"{peak_memory_kb:.2f} KB",
        )

        print(
            "Peak memory:",
            f"{peak_memory_kb / 1024:.4f} MB",
        )

    fieldnames = [
        "algorithm",
        "states",
        "value_entries",
        "policy_entries",
        "q_entries",
        "lambda",
        "current_memory_kb",
        "peak_memory_kb",
        "peak_memory_mb",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("Memory comparison finished.")
    print(
        "Saved:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )


if __name__ == "__main__":
    main()