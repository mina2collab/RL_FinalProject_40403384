from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "results" / "raw_data"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

DETAILED_CSV = (
    RAW_DATA_DIR
    / "reward_mode_comparison_detailed.csv"
)

SUMMARY_CSV = (
    RAW_DATA_DIR
    / "reward_mode_comparison_summary.csv"
)

COMPARISON_JSON = (
    RAW_DATA_DIR
    / "reward_mode_comparison.json"
)

BEHAVIOR_CHECK_CSV = (
    RAW_DATA_DIR
    / "reward_shaping_behavior_check.csv"
)

DISPLAY_NAMES = {
    ("q_learning", "sparse"):
        "Q-Learning\nSparse",
    ("q_learning", "shaped"):
        "Q-Learning\nShaped",
    ("sarsa_lambda_0_3", "sparse"):
        "SARSA(0.3)\nSparse",
    ("sarsa_lambda_0_3", "shaped"):
        "SARSA(0.3)\nShaped",
}


ORDER = [
    ("q_learning", "sparse"),
    ("q_learning", "shaped"),
    ("sarsa_lambda_0_3", "sparse"),
    ("sarsa_lambda_0_3", "shaped"),
]


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    """Read one CSV file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def read_json(
    path: Path,
) -> dict[str, Any]:
    """Read one JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def group_detailed_records(
    records: list[dict[str, str]],
) -> dict[
    tuple[str, str],
    list[dict[str, str]],
]:
    """Group detailed results by algorithm and reward."""

    grouped: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = {}

    for record in records:
        key = (
            record["algorithm"],
            record["reward_mode"],
        )

        grouped.setdefault(key, []).append(
            record
        )

    return grouped


def create_learning_speed_figure(
    detailed_records: list[dict[str, str]],
) -> Path:
    """Plot episodes needed for 80% and 95% success."""

    grouped = group_detailed_records(
        detailed_records
    )

    labels: list[str] = []
    ep80_means: list[float] = []
    ep80_stdevs: list[float] = []
    ep95_means: list[float] = []
    ep95_stdevs: list[float] = []

    for key in ORDER:
        group = grouped[key]

        ep80_values = [
            float(
                record[
                    "episodes_to_80_percent_success"
                ]
            )
            for record in group
            if float(
                record[
                    "episodes_to_80_percent_success"
                ]
            ) >= 0
        ]

        ep95_values = [
            float(
                record[
                    "episodes_to_95_percent_success"
                ]
            )
            for record in group
            if float(
                record[
                    "episodes_to_95_percent_success"
                ]
            ) >= 0
        ]

        labels.append(DISPLAY_NAMES[key])

        ep80_means.append(
            statistics.fmean(ep80_values)
        )
        ep80_stdevs.append(
            statistics.stdev(ep80_values)
            if len(ep80_values) > 1
            else 0.0
        )

        ep95_means.append(
            statistics.fmean(ep95_values)
        )
        ep95_stdevs.append(
            statistics.stdev(ep95_values)
            if len(ep95_values) > 1
            else 0.0
        )

    positions = list(range(len(labels)))
    width = 0.36

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.bar(
        [position - width / 2
         for position in positions],
        ep80_means,
        width=width,
        yerr=ep80_stdevs,
        capsize=5,
        label="80% success",
    )

    axis.bar(
        [position + width / 2
         for position in positions],
        ep95_means,
        width=width,
        yerr=ep95_stdevs,
        capsize=5,
        label="95% success",
    )

    axis.set_title(
        "Learning Speed: Sparse vs Shaped Reward"
    )
    axis.set_xlabel(
        "Algorithm and reward mode"
    )
    axis.set_ylabel(
        "Episodes required"
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "reward_learning_speed.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_path_quality_figure(
    summary_records: list[dict[str, str]],
) -> Path:
    """Plot evaluation path length with standard deviation."""

    record_by_key = {
        (
            record["algorithm"],
            record["reward_mode"],
        ): record
        for record in summary_records
    }

    labels = [
        DISPLAY_NAMES[key]
        for key in ORDER
    ]

    means = [
        float(
            record_by_key[key][
                "evaluation_average_steps_mean"
            ]
        )
        for key in ORDER
    ]

    stdevs = [
        float(
            record_by_key[key][
                "evaluation_average_steps_stdev"
            ]
        )
        for key in ORDER
    ]

    positions = list(range(len(labels)))

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    bars = axis.bar(
        positions,
        means,
        yerr=stdevs,
        capsize=6,
    )

    axis.set_title(
        "Evaluation Path Quality"
    )
    axis.set_xlabel(
        "Algorithm and reward mode"
    )
    axis.set_ylabel(
        "Average evaluation steps"
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, mean in zip(
        bars,
        means,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "reward_path_quality.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_policy_agreement_figure(
    comparison_data: dict[str, Any],
) -> Path:
    """Plot sparse-versus-shaped policy agreement."""

    comparisons = comparison_data[
        "sparse_vs_shaped_policy"
    ]

    labels: list[str] = []
    agreement_values: list[float] = []

    algorithm_labels = {
        "q_learning": "Q-Learning",
        "sarsa_lambda_0_3":
            "SARSA(lambda=0.3)",
    }

    for comparison in comparisons:
        labels.append(
            algorithm_labels[
                comparison["algorithm"]
            ]
        )

        agreement_values.append(
            float(
                comparison[
                    "policy_agreement_percent"
                ]
            )
        )

    positions = list(range(len(labels)))

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    bars = axis.bar(
        positions,
        agreement_values,
    )

    axis.set_title(
        "Sparse vs Shaped Policy Agreement"
    )
    axis.set_xlabel(
        "Algorithm"
    )
    axis.set_ylabel(
        "Policy agreement (%)"
    )
    axis.set_ylim(0, 100)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, agreement in zip(
        bars,
        agreement_values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{agreement:.2f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "reward_policy_agreement.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path

def create_behavior_check_csv() -> Path:
    """Check possible undesirable reward-shaping behavior."""

    training_paths = sorted(
        RAW_DATA_DIR.glob(
            "reward_comparison_*_training.csv"
        )
    )

    output_rows: list[dict[str, Any]] = []

    for path in training_paths:
        name = path.stem

        prefix = "reward_comparison_"
        suffix = "_training"

        if not (
            name.startswith(prefix)
            and name.endswith(suffix)
        ):
            continue

        core = name[
            len(prefix):-len(suffix)
        ]

        before_seed, seed_text = (
            core.rsplit(
                "_seed_",
                1,
            )
        )

        if before_seed.endswith(
            "_shaped"
        ):
            reward_mode = "shaped"
            algorithm = before_seed[
                :-len("_shaped")
            ]
        elif before_seed.endswith(
            "_sparse"
        ):
            reward_mode = "sparse"
            algorithm = before_seed[
                :-len("_sparse")
            ]
        else:
            continue

        records = read_csv(path)

        final_records = records[-100:]

        steps = [
            int(record["steps"])
            for record in final_records
        ]

        rewards = [
            float(record["reward"])
            for record in final_records
        ]

        wall_collisions = [
            int(record["wall_collisions"])
            for record in final_records
        ]

        penalty_visits = [
            int(record["penalty_visits"])
            for record in final_records
        ]

        closed_door_attempts = [
            int(
                record[
                    "closed_door_attempts"
                ]
            )
            for record in final_records
        ]

        success_count = sum(
            int(record["success"])
            for record in final_records
        )

        timeout_count = sum(
            int(record["truncated"])
            for record in final_records
        )

        long_success_count = sum(
            1
            for record in final_records
            if (
                int(record["success"]) == 1
                and int(record["steps"])
                >= 150
            )
        )

        median_reward = statistics.median(
            rewards
        )

        high_reward_long_count = sum(
            1
            for record in final_records
            if (
                int(record["success"]) == 1
                and int(record["steps"])
                >= 150
                and float(record["reward"])
                >= median_reward
            )
        )

        if (
            len(set(steps)) > 1
            and len(set(rewards)) > 1
        ):
            reward_steps_correlation = (
                statistics.correlation(
                    steps,
                    rewards,
                )
            )
        else:
            reward_steps_correlation = 0.0

        output_rows.append(
            {
                "algorithm": algorithm,
                "reward_mode": reward_mode,
                "seed": int(seed_text),
                "window_episodes": 100,
                "success_rate_pct": (
                    100.0
                    * success_count
                    / len(final_records)
                ),
                "mean_steps": statistics.mean(
                    steps
                ),
                "mean_wall_collisions": (
                    statistics.mean(
                        wall_collisions
                    )
                ),
                "mean_penalty_visits": (
                    statistics.mean(
                        penalty_visits
                    )
                ),
                "mean_closed_door_attempts": (
                    statistics.mean(
                        closed_door_attempts
                    )
                ),
                "timeout_rate_pct": (
                    100.0
                    * timeout_count
                    / len(final_records)
                ),
                "long_success_count_ge_150": (
                    long_success_count
                ),
                "high_reward_long_count": (
                    high_reward_long_count
                ),
                "reward_steps_correlation": (
                    reward_steps_correlation
                ),
            }
        )

    fieldnames = [
        "algorithm",
        "reward_mode",
        "seed",
        "window_episodes",
        "success_rate_pct",
        "mean_steps",
        "mean_wall_collisions",
        "mean_penalty_visits",
        "mean_closed_door_attempts",
        "timeout_rate_pct",
        "long_success_count_ge_150",
        "high_reward_long_count",
        "reward_steps_correlation",
    ]

    with BEHAVIOR_CHECK_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    return BEHAVIOR_CHECK_CSV

def main() -> None:
    """Generate all reward-comparison figures."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_records = read_csv(
        DETAILED_CSV
    )

    summary_records = read_csv(
        SUMMARY_CSV
    )

    comparison_data = read_json(
        COMPARISON_JSON
    )

    behavior_check_path = (
        create_behavior_check_csv()
    )
    output_paths = [
        create_learning_speed_figure(
            detailed_records
        ),
        create_path_quality_figure(
            summary_records
        ),
        create_policy_agreement_figure(
            comparison_data
        ),
    ]

    print(
        "Reward comparison figures generated:"
    )

    for output_path in output_paths:
        print(
            output_path.relative_to(
                PROJECT_ROOT
            )
        )


if __name__ == "__main__":
    main()