from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw_data"
    / "value_iteration_gamma_comparison.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "value_iteration_convergence.png"
)


def resolve_model_path(
    stored_path: str,
) -> Path:
    """Convert a stored Windows-style relative path to a local path."""

    path_parts = PureWindowsPath(
        stored_path
    ).parts

    return PROJECT_ROOT.joinpath(
        *path_parts
    )


def load_convergence_histories() -> tuple[
    float,
    dict[float, list[float]],
]:
    """Load delta histories for all gamma values."""

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary file was not found:\n"
            f"{SUMMARY_PATH}\n\n"
            "Run experiments/run_value_iteration.py first."
        )

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary_data = json.load(file)

    theta = float(
        summary_data["theta"]
    )

    histories: dict[
        float,
        list[float],
    ] = {}

    for result in summary_data["results"]:
        gamma = float(
            result["gamma"]
        )

        model_path = resolve_model_path(
            result["model_path"]
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file was not found:\n"
                f"{model_path}"
            )

        with model_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            model_data = json.load(file)

        delta_history = [
            float(value)
            for value in model_data[
                "delta_history"
            ]
        ]

        if not delta_history:
            raise ValueError(
                f"delta_history is empty for gamma={gamma}."
            )

        histories[gamma] = delta_history

    return theta, histories


def generate_figure() -> Path:
    """Generate Value Iteration convergence curves."""

    theta, histories = (
        load_convergence_histories()
    )

    figure, axis = plt.subplots(
        figsize=(10, 6.5)
    )

    line_styles = {
        0.80: "-",
        0.90: "--",
        0.95: "-.",
    }

    for gamma in sorted(histories):
        delta_history = histories[gamma]

        iterations = range(
            1,
            len(delta_history) + 1,
        )

        axis.semilogy(
            iterations,
            delta_history,
            label=(
                f"gamma = {gamma:.2f} "
                f"({len(delta_history)} iterations)"
            ),
            linestyle=line_styles.get(
                gamma,
                "-",
            ),
            linewidth=2,
        )

    axis.axhline(
        y=theta,
        linestyle=":",
        linewidth=1.8,
        label=f"Convergence threshold = {theta:.0e}",
    )

    axis.set_title(
        "Value Iteration Convergence\n"
        "Maximum Bellman Update by Iteration"
    )

    axis.set_xlabel(
        "Iteration"
    )

    axis.set_ylabel(
        "Maximum value change (delta, logarithmic scale)"
    )

    axis.grid(
        alpha=0.3,
        which="both",
    )

    axis.legend()

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

    for gamma in sorted(histories):
        delta_history = histories[gamma]

        print()
        print(
            f"gamma = {gamma:.2f}"
        )

        print(
            "Iterations:",
            len(delta_history),
        )

        print(
            "Final delta:",
            f"{delta_history[-1]:.12e}",
        )

    return OUTPUT_PATH


if __name__ == "__main__":
    generate_figure()