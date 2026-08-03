from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.sarsa_lambda import SARSALambdaAgent
from environments.maze import MazeEnv


LAMBDA_VALUES = [0.0, 0.3, 0.7, 0.9]

EPISODES = 600
EVALUATION_EPISODES = 100

ALPHA = 0.10
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 500

BASE_SEED = 8


def average(
    records: list[dict[str, Any]],
    field: str,
) -> float:
    """Return the average value of one field."""

    return statistics.fmean(
        float(record[field])
        for record in records
    )


def run_lambda(
    lambda_value: float,
) -> dict[str, Any]:
    """Train and evaluate one lambda value."""

    print()
    print("=" * 65)

    print(
        "Running SARSA(lambda) with "
        f"lambda={lambda_value}"
    )

    environment = MazeEnv(
        transition_seed=BASE_SEED,
        reward_mode="shaped",
        gamma=GAMMA,
    )

    agent = SARSALambdaAgent(
        environment=environment,
        alpha=ALPHA,
        gamma=GAMMA,
        lambda_value=lambda_value,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_episodes=(
            EPSILON_DECAY_EPISODES
        ),
        seed=BASE_SEED,
    )

    start_time = time.perf_counter()

    history = agent.train(
        episodes=EPISODES
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    evaluation = agent.evaluate(
        episodes=EVALUATION_EPISODES
    )

    first_100 = history[:100]
    last_100 = history[-100:]

    lambda_name = str(
        lambda_value
    ).replace(".", "_")

    history_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / f"sarsa_lambda_{lambda_name}_training.csv"
    )

    model_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / f"sarsa_lambda_{lambda_name}.json"
    )

    trace_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / f"sarsa_lambda_{lambda_name}_trace_example.json"
    )

    agent.save_training_history(
        history_path
    )

    agent.save_model(
        model_path
    )

    agent.save_trace_example(
        trace_path
    )

    result = {
        "lambda": lambda_value,
        "trace_type": "replacing",
        "seed": BASE_SEED,
        "episodes": EPISODES,
        "evaluation_episodes":
            EVALUATION_EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_episodes":
            EPSILON_DECAY_EPISODES,
        "training_seconds":
            training_seconds,
        "first_100_success_rate": (
            100.0
            * average(first_100, "success")
        ),
        "last_100_success_rate": (
            100.0
            * average(last_100, "success")
        ),
        "first_100_average_reward":
            average(first_100, "reward"),
        "last_100_average_reward":
            average(last_100, "reward"),
        "first_100_average_steps":
            average(first_100, "steps"),
        "last_100_average_steps":
            average(last_100, "steps"),
        "last_100_wall_collisions":
            average(
                last_100,
                "wall_collisions",
            ),
        "last_100_penalty_visits":
            average(
                last_100,
                "penalty_visits",
            ),
        "last_100_closed_door_attempts":
            average(
                last_100,
                "closed_door_attempts",
            ),
        "evaluation_success_rate":
            evaluation["success_rate"],
        "evaluation_average_reward":
            evaluation["average_reward"],
        "evaluation_average_steps":
            evaluation["average_steps"],
        "history_path": str(
            history_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "model_path": str(
            model_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "trace_example_path": str(
            trace_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    print()
    print(
        "Lambda finished:",
        lambda_value,
    )

    print(
        "Training time:",
        f"{training_seconds:.4f} seconds",
    )

    print(
        "First 100 success rate:",
        f'{result["first_100_success_rate"]:.2f}%',
    )

    print(
        "Last 100 success rate:",
        f'{result["last_100_success_rate"]:.2f}%',
    )

    print(
        "Last 100 average reward:",
        f'{result["last_100_average_reward"]:.2f}',
    )

    print(
        "Evaluation success rate:",
        f'{result["evaluation_success_rate"]:.2f}%',
    )

    print(
        "Evaluation average reward:",
        f'{result["evaluation_average_reward"]:.2f}',
    )

    print(
        "Evaluation average steps:",
        f'{result["evaluation_average_steps"]:.2f}',
    )

    return result


def save_summary_csv(
    results: list[dict[str, Any]],
) -> Path:
    """Save lambda comparison as CSV."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "sarsa_lambda_comparison.csv"
    )

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
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    return output_path


def save_summary_json(
    results: list[dict[str, Any]],
) -> Path:
    """Save settings and lambda results as JSON."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "sarsa_lambda_comparison.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "algorithm": "SARSA(lambda)",
        "student_id": "40403384",
        "environment_seed": BASE_SEED,
        "reward_mode": "shaped",
        "trace_type": "replacing",
        "lambda_values": LAMBDA_VALUES,
        "results": results,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


if __name__ == "__main__":
    results: list[dict[str, Any]] = []

    for lambda_value in LAMBDA_VALUES:
        result = run_lambda(
            lambda_value
        )

        results.append(result)

    csv_path = save_summary_csv(results)
    json_path = save_summary_json(results)

    print()
    print("=" * 65)

    print(
        "All SARSA(lambda) experiments "
        "finished."
    )

    print("Summary CSV saved to:", csv_path)
    print("Summary JSON saved to:", json_path)

    print()
    print("Final comparison:")

    for result in results:
        print(
            f'lambda={result["lambda"]:.1f} | '
            f'time={result["training_seconds"]:.4f}s | '
            f'last success='
            f'{result["last_100_success_rate"]:.2f}% | '
            f'evaluation success='
            f'{result["evaluation_success_rate"]:.2f}% | '
            f'evaluation reward='
            f'{result["evaluation_average_reward"]:.2f} | '
            f'evaluation steps='
            f'{result["evaluation_average_steps"]:.2f}'
        )