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
from agents.sarsa_lambda import SARSALambdaAgent
from environments.maze import ACTION_NAMES, MazeEnv


EPISODES = 600
EVALUATION_EPISODES = 100

ALPHA = 0.10
GAMMA = 0.95
LAMBDA_VALUE = 0.30

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 500

BASE_SEED = 8
RUN_SEEDS = [8, 18, 28]

REWARD_MODES = ["sparse", "shaped"]
ALGORITHMS = ["q_learning", "sarsa_lambda_0_3"]


def average(
    records: list[dict[str, Any]],
    field: str,
) -> float:
    """Return the average value of one history field."""

    return statistics.fmean(
        float(record[field])
        for record in records
    )


def episodes_to_success_rate(
    history: list[dict[str, Any]],
    threshold: float = 0.80,
    window: int = 100,
) -> int:
    """Return the first episode reaching a moving success threshold."""

    if len(history) < window:
        return -1

    for end_index in range(window, len(history) + 1):
        recent_records = history[
            end_index - window:end_index
        ]

        success_rate = statistics.fmean(
            float(record["success"])
            for record in recent_records
        )

        if success_rate >= threshold:
            return end_index

    return -1


def deterministic_evaluate(
    agent: QLearningAgent | SARSALambdaAgent,
    environment: MazeEnv,
    run_seed: int,
) -> dict[str, float]:
    """Evaluate with deterministic greedy tie-breaking."""

    successes = 0
    total_reward = 0.0
    total_steps = 0

    for episode in range(EVALUATION_EPISODES):
        state, _ = environment.reset(
            seed=40_000 + run_seed + episode
        )

        terminated = False
        truncated = False
        episode_success = False

        while not terminated and not truncated:
            action = min(
                agent.greedy_actions(state)
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            total_reward += reward
            state = next_state

            if "goal_reached" in info["events"]:
                episode_success = True

        successes += int(episode_success)
        total_steps += environment.step_count

    return {
        "success_rate": (
            100.0 * successes
            / EVALUATION_EPISODES
        ),
        "average_reward": (
            total_reward
            / EVALUATION_EPISODES
        ),
        "average_steps": (
            total_steps
            / EVALUATION_EPISODES
        ),
    }


def create_agent(
    algorithm: str,
    environment: MazeEnv,
    run_seed: int,
) -> QLearningAgent | SARSALambdaAgent:
    """Create one agent with common settings."""

    if algorithm == "q_learning":
        return QLearningAgent(
            environment=environment,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_episodes=(
                EPSILON_DECAY_EPISODES
            ),
            epsilon_schedule="linear",
            seed=run_seed,
        )

    if algorithm == "sarsa_lambda_0_3":
        return SARSALambdaAgent(
            environment=environment,
            alpha=ALPHA,
            gamma=GAMMA,
            lambda_value=LAMBDA_VALUE,
            epsilon_start=EPSILON_START,
            epsilon_end=EPSILON_END,
            epsilon_decay_episodes=(
                EPSILON_DECAY_EPISODES
            ),
            seed=run_seed,
        )

    raise ValueError(
        f"Unknown algorithm: {algorithm}"
    )


def run_one_experiment(
    algorithm: str,
    reward_mode: str,
    run_seed: int,
) -> tuple[
    dict[str, Any],
    QLearningAgent | SARSALambdaAgent,
]:
    """Train, evaluate and save one experiment."""

    print()
    print("=" * 72)
    print(
        f"Algorithm={algorithm} | "
        f"reward={reward_mode} | "
        f"seed={run_seed}"
    )

    environment = MazeEnv(
        transition_seed=run_seed,
        reward_mode=reward_mode,
        gamma=GAMMA,
    )

    agent = create_agent(
        algorithm=algorithm,
        environment=environment,
        run_seed=run_seed,
    )

    start_time = time.perf_counter()

    history = agent.train(
        episodes=EPISODES
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    evaluation = deterministic_evaluate(
        agent=agent,
        environment=environment,
        run_seed=run_seed,
    )

    first_100 = history[:100]
    last_100 = history[-100:]

    file_prefix = (
        f"reward_comparison_"
        f"{algorithm}_"
        f"{reward_mode}_"
        f"seed_{run_seed}"
    )

    history_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / f"{file_prefix}_training.csv"
    )

    model_path = (
        PROJECT_ROOT
        / "results"
        / "models"
        / f"{file_prefix}.json"
    )

    agent.save_training_history(
        history_path
    )

    agent.save_model(
        model_path
    )

    result = {
        "algorithm": algorithm,
        "reward_mode": reward_mode,
        "seed": run_seed,
        "episodes": EPISODES,
        "evaluation_episodes":
            EVALUATION_EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "lambda": (
            LAMBDA_VALUE
            if algorithm == "sarsa_lambda_0_3"
            else ""
        ),
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
        "episodes_to_80_percent_success":
            episodes_to_success_rate(
                history,
                threshold=0.80,
                window=100,
            ),
        "episodes_to_95_percent_success":
            episodes_to_success_rate(
                history,
                threshold=0.95,
                window=100,
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
    }

    print(
        "Finished | "
        f"time={training_seconds:.2f}s | "
        f"last success="
        f'{result["last_100_success_rate"]:.2f}% | '
        f"evaluation success="
        f'{result["evaluation_success_rate"]:.2f}% | '
        f"evaluation steps="
        f'{result["evaluation_average_steps"]:.2f}'
    )

    return result, agent


def aggregate_results(
    detailed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate mean and standard deviation over seeds."""

    metric_names = [
        "training_seconds",
        "first_100_success_rate",
        "last_100_success_rate",
        "first_100_average_steps",
        "last_100_average_steps",
        "evaluation_success_rate",
        "evaluation_average_reward",
        "evaluation_average_steps",
    ]

    aggregate_records: list[
        dict[str, Any]
    ] = []

    for algorithm in ALGORITHMS:
        for reward_mode in REWARD_MODES:
            group = [
                record
                for record in detailed_results
                if (
                    record["algorithm"]
                    == algorithm
                    and record["reward_mode"]
                    == reward_mode
                )
            ]

            aggregate_record: dict[
                str,
                Any,
            ] = {
                "algorithm": algorithm,
                "reward_mode": reward_mode,
                "number_of_runs": len(group),
                "seeds": ",".join(
                    str(record["seed"])
                    for record in group
                ),
            }

            for metric_name in metric_names:
                values = [
                    float(record[metric_name])
                    for record in group
                ]

                aggregate_record[
                    f"{metric_name}_mean"
                ] = statistics.fmean(values)

                aggregate_record[
                    f"{metric_name}_stdev"
                ] = (
                    statistics.stdev(values)
                    if len(values) > 1
                    else 0.0
                )

            aggregate_records.append(
                aggregate_record
            )

    return aggregate_records


def compare_reward_policies(
    base_agents: dict[
        tuple[str, str],
        QLearningAgent | SARSALambdaAgent,
    ],
) -> list[dict[str, Any]]:
    """Compare sparse and shaped greedy policies."""

    comparisons: list[dict[str, Any]] = []

    for algorithm in ALGORITHMS:
        sparse_agent = base_agents[
            (algorithm, "sparse")
        ]

        shaped_agent = base_agents[
            (algorithm, "shaped")
        ]

        sparse_policy = (
            sparse_agent.greedy_policy()
        )
        shaped_policy = (
            shaped_agent.greedy_policy()
        )

        states = sorted(
            state
            for state in sparse_policy
            if (
                sparse_policy[state] is not None
                and shaped_policy[state]
                is not None
            )
        )

        agreeing_states = [
            state
            for state in states
            if (
                sparse_policy[state]
                == shaped_policy[state]
            )
        ]

        differing_states = [
            state
            for state in states
            if (
                sparse_policy[state]
                != shaped_policy[state]
            )
        ]

        examples = []

        for state in differing_states[:10]:
            sparse_action = sparse_policy[state]
            shaped_action = shaped_policy[state]

            examples.append(
                {
                    "state": list(state),
                    "sparse_action":
                        sparse_action,
                    "sparse_action_name":
                        ACTION_NAMES[
                            int(sparse_action)
                        ],
                    "shaped_action":
                        shaped_action,
                    "shaped_action_name":
                        ACTION_NAMES[
                            int(shaped_action)
                        ],
                }
            )

        comparisons.append(
            {
                "algorithm": algorithm,
                "seed": BASE_SEED,
                "compared_states":
                    len(states),
                "agreeing_states":
                    len(agreeing_states),
                "differing_states":
                    len(differing_states),
                "policy_agreement_percent": (
                    100.0
                    * len(agreeing_states)
                    / len(states)
                ),
                "difference_examples":
                    examples,
            }
        )

    return comparisons


def save_csv(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Save records as CSV."""

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
                records[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(records)


def save_json(
    output_path: Path,
    detailed_results: list[dict[str, Any]],
    aggregate_records: list[dict[str, Any]],
    policy_comparisons: list[
        dict[str, Any]
    ],
) -> None:
    """Save complete experiment information as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "student_id": "40403384",
        "base_seed": BASE_SEED,
        "run_seeds": RUN_SEEDS,
        "episodes": EPISODES,
        "evaluation_episodes":
            EVALUATION_EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "lambda": LAMBDA_VALUE,
        "epsilon_start": EPSILON_START,
        "epsilon_end": EPSILON_END,
        "epsilon_decay_episodes":
            EPSILON_DECAY_EPISODES,
        "evaluation_tie_breaking":
            "minimum greedy action",
        "detailed_results":
            detailed_results,
        "aggregate_results":
            aggregate_records,
        "sparse_vs_shaped_policy":
            policy_comparisons,
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


if __name__ == "__main__":
    detailed_results: list[
        dict[str, Any]
    ] = []

    base_agents: dict[
        tuple[str, str],
        QLearningAgent | SARSALambdaAgent,
    ] = {}

    for algorithm in ALGORITHMS:
        for reward_mode in REWARD_MODES:
            for run_seed in RUN_SEEDS:
                result, agent = (
                    run_one_experiment(
                        algorithm=algorithm,
                        reward_mode=reward_mode,
                        run_seed=run_seed,
                    )
                )

                detailed_results.append(
                    result
                )

                if run_seed == BASE_SEED:
                    base_agents[
                        (
                            algorithm,
                            reward_mode,
                        )
                    ] = agent

    aggregate_records = aggregate_results(
        detailed_results
    )

    policy_comparisons = (
        compare_reward_policies(
            base_agents
        )
    )

    raw_data_directory = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
    )

    detailed_csv_path = (
        raw_data_directory
        / "reward_mode_comparison_detailed.csv"
    )

    aggregate_csv_path = (
        raw_data_directory
        / "reward_mode_comparison_summary.csv"
    )

    json_path = (
        raw_data_directory
        / "reward_mode_comparison.json"
    )

    save_csv(
        detailed_csv_path,
        detailed_results,
    )

    save_csv(
        aggregate_csv_path,
        aggregate_records,
    )

    save_json(
        json_path,
        detailed_results,
        aggregate_records,
        policy_comparisons,
    )

    print()
    print("=" * 72)
    print("All reward comparison experiments finished.")
    print("Detailed CSV:", detailed_csv_path)
    print("Summary CSV:", aggregate_csv_path)
    print("JSON:", json_path)

    print()
    print("Sparse versus shaped policy agreement:")

    for comparison in policy_comparisons:
        print(
            f'{comparison["algorithm"]}: '
            f'{comparison["policy_agreement_percent"]:.2f}%'
        )