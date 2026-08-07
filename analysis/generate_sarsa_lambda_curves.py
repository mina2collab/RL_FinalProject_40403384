from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "sarsa_lambda_training_curves.png"
)

INPUT_FILES = {
    "lambda = 0.0": (
        RAW_DATA_DIR
        / "sarsa_lambda_0_0_training.csv"
    ),
    "lambda = 0.3": (
        RAW_DATA_DIR
        / "sarsa_lambda_0_3_training.csv"
    ),
    "lambda = 0.7": (
        RAW_DATA_DIR
        / "sarsa_lambda_0_7_training.csv"
    ),
    "lambda = 0.9": (
        RAW_DATA_DIR
        / "sarsa_lambda_0_9_training.csv"
    ),
}

ROLLING_WINDOW = 50


def rolling_mean(
    values: list[float],
    window: int,
) -> list[float]:
    """Calculate a moving average."""

    averages: list[float] = []

    for index in range(len(values)):
        start_index = max(
            0,
            index - window + 1,
        )

        current_values = values[
            start_index:index + 1
        ]

        averages.append(
            sum(current_values)
            / len(current_values)
        )

    return averages


def load_training_history(
    file_path: Path,
) -> dict[str, list[float]]:
    """Load one SARSA(lambda) training history."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training file was not found:\n"
            f"{file_path}\n\n"
            "Run experiments/run_sarsa_lambda.py first."
        )

    episodes: list[float] = []
    success_values: list[float] = []
    reward_values: list[float] = []

    with file_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "episode",
            "success",
            "reward",
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns
            - available_columns
        )

        if missing_columns:
            raise ValueError(
                "Missing CSV columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        for row in reader:
            episodes.append(
                float(row["episode"])
            )

            success_values.append(
                100.0
                * float(row["success"])
            )

            reward_values.append(
                float(row["reward"])
            )

    return {
        "episodes": episodes,
        "success_rate": rolling_mean(
            success_values,
            ROLLING_WINDOW,
        ),
        "reward": rolling_mean(
            reward_values,
            ROLLING_WINDOW,
        ),
    }


def generate_figure() -> Path:
    """Generate SARSA(lambda) training curves."""

    histories = {
        label: load_training_history(path)
        for label, path in INPUT_FILES.items()
    }

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    success_axis = axes[0]
    reward_axis = axes[1]

    line_styles = {
        "lambda = 0.0": "-",
        "lambda = 0.3": "--",
        "lambda = 0.7": "-.",
        "lambda = 0.9": ":",
    }

    for label, history in histories.items():
        success_axis.plot(
            history["episodes"],
            history["success_rate"],
            label=label,
            linestyle=line_styles[label],
            linewidth=2,
        )

        reward_axis.plot(
            history["episodes"],
            history["reward"],
            label=label,
            linestyle=line_styles[label],
            linewidth=2,
        )

    success_axis.set_title(
        "SARSA(lambda) Training Curves\n"
        "Comparison of Eligibility-Trace Parameters"
    )

    success_axis.set_ylabel(
        f"Success rate (%)\n"
        f"{ROLLING_WINDOW}-episode moving average"
    )

    success_axis.set_ylim(
        -2,
        102,
    )

    success_axis.grid(
        alpha=0.3
    )

    success_axis.legend(
        ncol=2
    )

    reward_axis.set_xlabel(
        "Training episode"
    )

    reward_axis.set_ylabel(
        f"Episode reward\n"
        f"{ROLLING_WINDOW}-episode moving average"
    )

    reward_axis.grid(
        alpha=0.3
    )

    reward_axis.legend(
        ncol=2
    )

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Figure generated successfully."
    )

    print(
        "Output:",
        OUTPUT_PATH,
    )

    for label, history in histories.items():
        print()
        print(label)

        print(
            "Final rolling success rate:",
            f'{history["success_rate"][-1]:.2f}%',
        )

        print(
            "Final rolling reward:",
            f'{history["reward"][-1]:.2f}',
        )

    return OUTPUT_PATH


if __name__ == "__main__":
    generate_figure()