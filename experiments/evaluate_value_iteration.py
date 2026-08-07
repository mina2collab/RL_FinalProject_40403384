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
from environments.maze import MazeEnv


GAMMA = 0.95
THETA = 1e-8
MAX_ITERATIONS = 10_000

EVALUATION_EPISODES = 100
BASE_SEED = 8
SEED_OFFSET = 10_000


def evaluate_policy(
    environment: MazeEnv,
    agent: ValueIterationAgent,
) -> dict[str, Any]:
    """Evaluate the converged policy over multiple episodes."""

    successes = 0
    total_reward = 0.0
    total_steps = 0

    episode_records: list[dict[str, Any]] = []

    for episode in range(EVALUATION_EPISODES):
        evaluation_seed = (
            BASE_SEED
            + SEED_OFFSET
            + episode
        )

        state, _ = environment.reset(
            seed=evaluation_seed
        )

        terminated = False
        truncated = False
        episode_success = False
        episode_reward = 0.0

        while not terminated and not truncated:
            action = agent.policy.get(state)

            if action is None:
                raise RuntimeError(
                    "No policy action was found for state: "
                    f"{state}"
                )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = environment.step(action)

            episode_reward += reward
            state = next_state

            if "goal_reached" in info["events"]:
                episode_success = True

        successes += int(episode_success)
        total_reward += episode_reward
        total_steps += environment.step_count

        episode_records.append(
            {
                "episode": episode + 1,
                "seed": evaluation_seed,
                "success": int(episode_success),
                "reward": episode_reward,
                "steps": environment.step_count,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    summary = {
        "algorithm": "Value Iteration",
        "gamma": GAMMA,
        "theta": THETA,
        "reward_mode": "sparse",
        "evaluation_episodes": EVALUATION_EPISODES,
        "base_seed": BASE_SEED,
        "seed_offset": SEED_OFFSET,
        "success_rate": (
            100.0
            * successes
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

    return {
        "summary": summary,
        "episodes": episode_records,
    }


def save_episode_csv(
    episode_records: list[dict[str, Any]],
) -> Path:
    """Save per-episode evaluation results."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "value_iteration_evaluation_episodes.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "episode",
        "seed",
        "success",
        "reward",
        "steps",
        "terminated",
        "truncated",
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
        writer.writerows(episode_records)

    return output_path


def save_summary_csv(
    summary: dict[str, Any],
) -> Path:
    """Save the evaluation summary as CSV."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "value_iteration_evaluation_summary.csv"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary.keys()),
        )

        writer.writeheader()
        writer.writerow(summary)

    return output_path


def save_json(
    result: dict[str, Any],
    convergence_summary: dict[str, Any],
) -> Path:
    """Save settings, convergence, and evaluation results."""

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "value_iteration_evaluation.json"
    )

    output_data = {
        "student_id": "40403384",
        "convergence": convergence_summary,
        "evaluation": result["summary"],
        "episodes": result["episodes"],
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def main() -> None:
    """Run and evaluate Value Iteration."""

    print("=" * 65)
    print("Running Value Iteration evaluation")
    print(f"gamma = {GAMMA}")
    print(f"evaluation episodes = {EVALUATION_EPISODES}")

    environment = MazeEnv(
        transition_seed=BASE_SEED,
        reward_mode="sparse",
        gamma=GAMMA,
    )

    agent = ValueIterationAgent(
        environment=environment,
        gamma=GAMMA,
        theta=THETA,
        max_iterations=MAX_ITERATIONS,
    )

    convergence_summary = agent.run()

    result = evaluate_policy(
        environment=environment,
        agent=agent,
    )

    episode_csv_path = save_episode_csv(
        result["episodes"]
    )

    summary_csv_path = save_summary_csv(
        result["summary"]
    )

    json_path = save_json(
        result=result,
        convergence_summary=convergence_summary,
    )

    summary = result["summary"]

    print()
    print("=" * 65)
    print("Value Iteration evaluation finished.")

    print(
        "Converged:",
        convergence_summary["converged"],
    )

    print(
        "Iterations:",
        convergence_summary["iterations"],
    )

    print(
        "Success rate:",
        f'{summary["success_rate"]:.2f}%',
    )

    print(
        "Average reward:",
        f'{summary["average_reward"]:.2f}',
    )

    print(
        "Average steps:",
        f'{summary["average_steps"]:.2f}',
    )

    print()
    print("Episode CSV:", episode_csv_path)
    print("Summary CSV:", summary_csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()