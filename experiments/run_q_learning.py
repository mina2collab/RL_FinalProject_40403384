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


from agents.q_learning import QLearningAgent
from environments.maze import MazeEnv


SCHEDULES = ["linear", "exponential"]

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
    """Return the mean of one numeric field."""

    return statistics.fmean(
        float(record[field])
        for record in records
    )


def run_schedule(
    schedule: str,
) -> dict[str, Any]:
    """Train and evaluate one epsilon schedule."""

    print()
    print("=" * 65)
    print(
        "Running Q-Learning with "
        f"epsilon schedule: {schedule}"
    )

    environment = MazeEnv(
        transition_seed=BASE_SEED,
        reward_mode="shaped",
        gamma=GAMMA,
    )

    agent = QLearningAgent(
        environment=environment,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_episodes=(
            EPSILON_DECAY_EPISODES
        ),
        epsilon_schedule=schedule,
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

    history_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / f"q_learning_{schedule}_training.csv"
    )

    model_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / f"q_learning_{schedule}.json"
    )

    update_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / f"q_learning_{schedule}_update_example.json"
    )

    agent.save_training_history(
        history_path
    )

    agent.save_model(
        model_path
    )

    agent.save_update_example(
        update_path
    )

    result = {
        "schedule": schedule,
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
        "update_example_path": str(
            update_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    print()
    print("Schedule finished:", schedule)

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
    """Save schedule comparison as CSV."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "q_learning_schedule_comparison.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        results[0].keys()
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
        writer.writerows(results)

    return output_path


def save_summary_json(
    results: list[dict[str, Any]],
) -> Path:
    """Save settings and results as JSON."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "q_learning_schedule_comparison.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "algorithm": "Q-Learning",
        "student_id": "40403384",
        "environment_seed": BASE_SEED,
        "reward_mode": "shaped",
        "schedules": SCHEDULES,
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

    for schedule in SCHEDULES:
        result = run_schedule(schedule)
        results.append(result)

    csv_path = save_summary_csv(results)
    json_path = save_summary_json(results)

    print()
    print("=" * 65)
    print(
        "All Q-Learning schedule "
        "experiments finished."
    )

    print("Summary CSV saved to:", csv_path)
    print("Summary JSON saved to:", json_path)

    print()
    print("Final comparison:")

    for result in results:
        print(
            f'{result["schedule"]:11s} | '
            f'time={result["training_seconds"]:.4f}s | '
            f'last success='
            f'{result["last_100_success_rate"]:.2f}% | '
            f'evaluation success='
            f'{result["evaluation_success_rate"]:.2f}% | '
            f'evaluation steps='
            f'{result["evaluation_average_steps"]:.2f}'
        )