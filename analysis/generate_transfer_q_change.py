from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.q_learning import QLearningAgent
from environments.maze import MazeEnv
from transfer.transfer_learning import (
    DIFFERENT_MAP_PATH,
    SOURCE_MAP_PATH,
    initialize_target_q_table,
    load_source_q_table,
)


FINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "transfer"
    / "different_selective.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "transfer_q_change_different_selective.png"
)


def load_saved_q_table(
    model_path: Path,
) -> dict[tuple[int, int, int, int], list[float]]:
    """Load a Q-table from a saved model JSON file."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Final transfer model was not found:\n{model_path}\n"
            "Run experiments/run_transfer_learning.py first."
        )

    with model_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        model_data = json.load(file)

    q_table: dict[
        tuple[int, int, int, int],
        list[float],
    ] = {}

    for record in model_data["q_table"]:
        state = tuple(int(value) for value in record["state"])

        q_table[state] = [
            float(value)
            for value in record["q_values"]
        ]

    return q_table


def build_initial_selective_q_table(
    source_environment: MazeEnv,
    target_environment: MazeEnv,
) -> tuple[
    dict[tuple[int, int, int, int], list[float]],
    dict[str, object],
]:
    """Reconstruct the Q-table immediately after selective transfer."""

    source_q_table = load_source_q_table()

    target_agent = QLearningAgent(
        environment=target_environment,
        alpha=0.10,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_episodes=300,
        epsilon_schedule="linear",
        seed=8,
    )

    transfer_summary = initialize_target_q_table(
        target_agent=target_agent,
        source_q_table=source_q_table,
        mode="selective",
        source_environment=source_environment,
        beta=1.0,
    )

    initial_q_table = {
        state: values.copy()
        for state, values in target_agent.q_table.items()
    }

    return initial_q_table, transfer_summary


def generate_q_change_figure() -> Path:
    """Generate a heatmap of max-Q change before and after training."""

    source_environment = MazeEnv(
        map_path=SOURCE_MAP_PATH,
        transition_seed=8,
        reward_mode="shaped",
        gamma=0.95,
    )

    target_environment = MazeEnv(
        map_path=DIFFERENT_MAP_PATH,
        transition_seed=8,
        reward_mode="shaped",
        gamma=0.95,
    )

    (
        initial_q_table,
        transfer_summary,
    ) = build_initial_selective_q_table(
        source_environment=source_environment,
        target_environment=target_environment,
    )

    final_q_table = load_saved_q_table(
        FINAL_MODEL_PATH
    )

    has_key = 0
    gate_phase = 0

    change_grid = np.full(
        (
            target_environment.size,
            target_environment.size,
        ),
        np.nan,
        dtype=float,
    )

    changed_policy_count = 0
    compared_state_count = 0

    for row in range(target_environment.size):
        for column in range(target_environment.size):
            state = (
                row,
                column,
                has_key,
                gate_phase,
            )

            if (
                state not in initial_q_table
                or state not in final_q_table
            ):
                continue

            initial_values = initial_q_table[state]
            final_values = final_q_table[state]

            change_grid[row, column] = (
                max(final_values)
                - max(initial_values)
            )

            initial_action = min(
                action
                for action, value in enumerate(initial_values)
                if np.isclose(
                    value,
                    max(initial_values),
                    rtol=1e-12,
                    atol=1e-12,
                )
            )

            final_action = min(
                action
                for action, value in enumerate(final_values)
                if np.isclose(
                    value,
                    max(final_values),
                    rtol=1e-12,
                    atol=1e-12,
                )
            )

            compared_state_count += 1

            if initial_action != final_action:
                changed_policy_count += 1

    finite_changes = np.abs(
        change_grid[np.isfinite(change_grid)]
    )

    if finite_changes.size == 0:
        raise RuntimeError(
            "No comparable states were found."
        )

    color_limit = float(
        np.percentile(finite_changes, 95)
    )

    if color_limit < 1e-9:
        color_limit = 1.0

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    color_map = matplotlib.colormaps[
        "coolwarm"
    ].copy()

    color_map.set_bad("black")

    image = axis.imshow(
        change_grid,
        cmap=color_map,
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
    )

    color_bar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    color_bar.set_label(
        "Change in max Q: after training - before training"
    )

    for row in range(target_environment.size):
        for column in range(target_environment.size):
            symbol = target_environment.grid[row][column]

            if symbol in {
                target_environment.empty_symbol,
                target_environment.wall_symbol,
            }:
                continue

            axis.text(
                column,
                row,
                symbol,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    changed_percentage = (
        100.0
        * changed_policy_count
        / compared_state_count
        if compared_state_count
        else 0.0
    )

    axis.set_title(
        "Selective Transfer on Different Target\n"
        "Max-Q Change Before vs After Target Training\n"
        f"has_key={has_key}, gate_phase={gate_phase}"
    )

    axis.set_xlabel("Column")
    axis.set_ylabel("Row")

    axis.set_xticks(
        range(target_environment.size)
    )

    axis.set_yticks(
        range(target_environment.size)
    )

    axis.tick_params(
        axis="both",
        labelsize=7,
    )

    axis.text(
        0.5,
        -0.11,
        (
            f"Transferred states: "
            f'{transfer_summary["transferred_states"]} | '
            f"Greedy action changed in "
            f"{changed_policy_count}/{compared_state_count} states "
            f"({changed_percentage:.2f}%)"
        ),
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
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

    print("Figure generated successfully.")
    print("Output:", OUTPUT_PATH)
    print(
        "Transferred states:",
        transfer_summary["transferred_states"],
    )
    print(
        "Changed greedy actions:",
        changed_policy_count,
        "/",
        compared_state_count,
    )
    print(
        "Changed policy percentage:",
        f"{changed_percentage:.2f}%",
    )

    return OUTPUT_PATH


if __name__ == "__main__":
    generate_q_change_figure()