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
from transfer.transfer_learning import (
    DIFFERENT_MAP_PATH,
    SIMILAR_MAP_PATH,
    SOURCE_MAP_PATH,
    create_target_maps,
    initialize_target_q_table,
    load_source_q_table,
)


TARGETS = {
    "similar": SIMILAR_MAP_PATH,
    "different": DIFFERENT_MAP_PATH,
}

SCENARIOS = [
    {
        "name": "scratch",
        "mode": "scratch",
        "beta": 0.0,
    },
    {
        "name": "full",
        "mode": "full",
        "beta": 1.0,
    },
    {
        "name": "scaled_0_25",
        "mode": "scaled",
        "beta": 0.25,
    },
    {
        "name": "scaled_0_50",
        "mode": "scaled",
        "beta": 0.50,
    },
    {
        "name": "scaled_0_75",
        "mode": "scaled",
        "beta": 0.75,
    },
    {
        "name": "selective",
        "mode": "selective",
        "beta": 1.0,
    },
]

EPISODES = 400
EVALUATION_EPISODES = 100

ALPHA = 0.10
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 300

BASE_SEED = 8


def average(
    records: list[dict[str, Any]],
    field: str,
) -> float:
    """Return the average of one numeric field."""

    if not records:
        return 0.0

    return statistics.fmean(
        float(record[field])
        for record in records
    )


def rolling_success_episode(
    history: list[dict[str, Any]],
    threshold: float,
    window: int = 50,
) -> int | None:
    """
    Return the first episode whose rolling success rate
    reaches the requested threshold.
    """

    if len(history) < window:
        return None

    for ending_index in range(
        window,
        len(history) + 1,
    ):
        window_records = history[
            ending_index - window:
            ending_index
        ]

        success_rate = average(
            window_records,
            "success",
        )

        if success_rate >= threshold:
            return ending_index

    return None


def save_history(
    history: list[dict[str, Any]],
    target_name: str,
    scenario_name: str,
) -> Path:
    """Save one scenario's episode history."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "transfer"
        / (
            f"{target_name}_"
            f"{scenario_name}_training.csv"
        )
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
                history[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(history)

    return output_path


def run_scenario(
    target_name: str,
    target_path: Path,
    scenario: dict[str, Any],
    source_q_table: dict[
        tuple[int, int, int, int],
        list[float],
    ],
    source_environment: MazeEnv,
) -> dict[str, Any]:
    """Run one transfer-learning scenario."""

    scenario_name = str(
        scenario["name"]
    )

    transfer_mode = str(
        scenario["mode"]
    )

    beta = float(
        scenario["beta"]
    )

    print()
    print("=" * 72)

    print(
        f"Target={target_name} | "
        f"Scenario={scenario_name}"
    )

    target_environment = MazeEnv(
        map_path=target_path,
        transition_seed=BASE_SEED,
        reward_mode="shaped",
        gamma=GAMMA,
    )

    agent = QLearningAgent(
        environment=target_environment,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_episodes=(
            EPSILON_DECAY_EPISODES
        ),
        epsilon_schedule="linear",
        seed=BASE_SEED,
    )

    transfer_summary = (
        initialize_target_q_table(
            target_agent=agent,
            source_q_table=source_q_table,
            mode=transfer_mode,
            source_environment=(
                source_environment
            ),
            beta=beta,
        )
    )

    # Initial performance before target training.
    agent.random_generator.seed(
        BASE_SEED + 1_000
    )

    initial_evaluation = agent.evaluate(
        episodes=EVALUATION_EPISODES,
        seed_offset=30_000,
    )

    # Reset action-selection randomness so that
    # training remains reproducible.
    agent.random_generator.seed(
        BASE_SEED
    )

    training_start = time.perf_counter()

    history = agent.train(
        episodes=EPISODES
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    # Use a common evaluation seed for all scenarios.
    agent.random_generator.seed(
        BASE_SEED + 2_000
    )

    final_evaluation = agent.evaluate(
        episodes=EVALUATION_EPISODES,
        seed_offset=40_000,
    )

    first_50 = history[:50]
    first_100 = history[:100]
    last_100 = history[-100:]

    episode_to_80 = rolling_success_episode(
        history=history,
        threshold=0.80,
        window=50,
    )

    episode_to_90 = rolling_success_episode(
        history=history,
        threshold=0.90,
        window=50,
    )

    history_path = save_history(
        history=history,
        target_name=target_name,
        scenario_name=scenario_name,
    )

    model_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / "transfer"
        / (
            f"{target_name}_"
            f"{scenario_name}.json"
        )
    )

    update_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "transfer"
        / (
            f"{target_name}_"
            f"{scenario_name}_"
            "update_example.json"
        )
    )

    agent.save_model(model_path)
    agent.save_update_example(update_path)

    result = {
        "target": target_name,
        "scenario": scenario_name,
        "transfer_mode": transfer_mode,
        "beta": beta,
        "transferred_states":
            transfer_summary[
                "transferred_states"
            ],
        "transfer_percentage":
            transfer_summary[
                "transfer_percentage"
            ],
        "skipped_changed_neighborhoods":
            transfer_summary[
                "skipped_changed_neighborhoods"
            ],
        "skipped_missing_states":
            transfer_summary[
                "skipped_missing_states"
            ],
        "episodes": EPISODES,
        "training_seconds":
            training_seconds,
        "initial_success_rate":
            initial_evaluation[
                "success_rate"
            ],
        "initial_average_reward":
            initial_evaluation[
                "average_reward"
            ],
        "initial_average_steps":
            initial_evaluation[
                "average_steps"
            ],
        "first_50_success_rate": (
            100.0
            * average(
                first_50,
                "success",
            )
        ),
        "first_100_success_rate": (
            100.0
            * average(
                first_100,
                "success",
            )
        ),
        "first_100_average_reward":
            average(
                first_100,
                "reward",
            ),
        "first_100_average_steps":
            average(
                first_100,
                "steps",
            ),
        "episode_to_80_percent":
            episode_to_80,
        "episode_to_90_percent":
            episode_to_90,
        "last_100_success_rate": (
            100.0
            * average(
                last_100,
                "success",
            )
        ),
        "last_100_average_reward":
            average(
                last_100,
                "reward",
            ),
        "last_100_average_steps":
            average(
                last_100,
                "steps",
            ),
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
        "final_success_rate":
            final_evaluation[
                "success_rate"
            ],
        "final_average_reward":
            final_evaluation[
                "average_reward"
            ],
        "final_average_steps":
            final_evaluation[
                "average_steps"
            ],
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
    print(
        "Transferred states:",
        result["transferred_states"],
    )

    print(
        "Transfer percentage:",
        f'{result["transfer_percentage"]:.2f}%',
    )

    print(
        "Initial success:",
        f'{result["initial_success_rate"]:.2f}%',
    )

    print(
        "First 100 success:",
        f'{result["first_100_success_rate"]:.2f}%',
    )

    print(
        "Last 100 success:",
        f'{result["last_100_success_rate"]:.2f}%',
    )

    print(
        "Final success:",
        f'{result["final_success_rate"]:.2f}%',
    )

    print(
        "Final reward:",
        f'{result["final_average_reward"]:.2f}',
    )

    print(
        "Final steps:",
        f'{result["final_average_steps"]:.2f}',
    )

    return result


def add_transfer_comparisons(
    results: list[dict[str, Any]],
) -> None:
    """
    Compare every transfer mode with scratch on
    the same destination environment.
    """

    for target_name in TARGETS:
        target_results = [
            result
            for result in results
            if result["target"] == target_name
        ]

        scratch_result = next(
            result
            for result in target_results
            if result["scenario"] == "scratch"
        )

        for result in target_results:
            result[
                "initial_success_gain_vs_scratch"
            ] = (
                result["initial_success_rate"]
                - scratch_result[
                    "initial_success_rate"
                ]
            )

            result[
                "first_100_reward_gain_vs_scratch"
            ] = (
                result[
                    "first_100_average_reward"
                ]
                - scratch_result[
                    "first_100_average_reward"
                ]
            )

            result[
                "final_reward_gain_vs_scratch"
            ] = (
                result[
                    "final_average_reward"
                ]
                - scratch_result[
                    "final_average_reward"
                ]
            )

            result[
                "negative_transfer_detected"
            ] = bool(
                result["scenario"] != "scratch"
                and (
                    result[
                        "first_100_reward_gain_vs_scratch"
                    ] < 0
                    or result[
                        "initial_success_gain_vs_scratch"
                    ] < 0
                )
            )


def save_summary_csv(
    results: list[dict[str, Any]],
) -> Path:
    """Save the comparison table as CSV."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "transfer_learning_comparison.csv"
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
    """Save configuration and comparison data."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "transfer_learning_comparison.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_scenarios: dict[
        str,
        dict[str, Any],
    ] = {}

    for target_name in TARGETS:
        target_results = [
            result
            for result in results
            if result["target"] == target_name
        ]

        best_result = max(
            target_results,
            key=lambda result: (
                result["final_success_rate"],
                result["final_average_reward"],
                -result["final_average_steps"],
            ),
        )

        best_scenarios[target_name] = {
            "scenario":
                best_result["scenario"],
            "final_success_rate":
                best_result[
                    "final_success_rate"
                ],
            "final_average_reward":
                best_result[
                    "final_average_reward"
                ],
            "final_average_steps":
                best_result[
                    "final_average_steps"
                ],
        }

    data = {
        "algorithm": "Q-Learning transfer",
        "student_id": "40403384",
        "source_map": str(
            SOURCE_MAP_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "episodes": EPISODES,
        "evaluation_episodes":
            EVALUATION_EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_episodes":
            EPSILON_DECAY_EPISODES,
        "best_scenarios":
            best_scenarios,
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
    # Recreate deterministic target maps.
    create_target_maps()

    source_q_table = load_source_q_table()

    source_environment = MazeEnv(
        map_path=SOURCE_MAP_PATH,
        transition_seed=BASE_SEED,
        reward_mode="shaped",
        gamma=GAMMA,
    )

    all_results: list[
        dict[str, Any]
    ] = []

    for target_name, target_path in TARGETS.items():
        for scenario in SCENARIOS:
            result = run_scenario(
                target_name=target_name,
                target_path=target_path,
                scenario=scenario,
                source_q_table=(
                    source_q_table
                ),
                source_environment=(
                    source_environment
                ),
            )

            all_results.append(result)

    add_transfer_comparisons(
        all_results
    )

    csv_path = save_summary_csv(
        all_results
    )

    json_path = save_summary_json(
        all_results
    )

    print()
    print("=" * 72)
    print(
        "All transfer-learning "
        "experiments finished."
    )

    print(
        "Summary CSV saved to:",
        csv_path,
    )

    print(
        "Summary JSON saved to:",
        json_path,
    )

    print()
    print("Final comparison:")

    for result in all_results:
        print(
            f'{result["target"]:9s} | '
            f'{result["scenario"]:12s} | '
            f'initial='
            f'{result["initial_success_rate"]:6.2f}% | '
            f'first100='
            f'{result["first_100_success_rate"]:6.2f}% | '
            f'final='
            f'{result["final_success_rate"]:6.2f}% | '
            f'reward='
            f'{result["final_average_reward"]:8.2f} | '
            f'steps='
            f'{result["final_average_steps"]:7.2f} | '
            f'negative='
            f'{result["negative_transfer_detected"]}'
        )