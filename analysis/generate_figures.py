from __future__ import annotations

import ast
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.maze import MazeEnv


FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MAP = (
    ROOT
    / "environments"
    / "maps"
    / "source_map.json"
)

VALUE_MODEL = (
    ROOT
    / "results"
    / "models"
    / "value_iteration_gamma_0_95.json"
)

Q_MODEL = (
    ROOT
    / "results"
    / "models"
    / "q_learning_linear.json"
)

TRANSFER_CSV = (
    ROOT
    / "results"
    / "raw_data"
    / "transfer_learning_comparison.csv"
)

ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
}


def parse_state(value: Any) -> tuple[int, int, int, int]:
    """Convert a saved state to a tuple."""

    if isinstance(value, (list, tuple)):
        items = value

    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple)):
                items = parsed
            else:
                items = value.split(",")

        except (ValueError, SyntaxError):
            items = (
                value.replace("(", "")
                .replace(")", "")
                .split(",")
            )

    else:
        raise ValueError(
            f"Unsupported state value: {value}"
        )

    if len(items) != 4:
        raise ValueError(
            f"Invalid state: {value}"
        )

    return tuple(
        int(item)
        for item in items
    )


def load_json(path: Path) -> dict:
    """Load a JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_model(
    path: Path,
) -> tuple[
    dict[tuple[int, int, int, int], float],
    dict[tuple[int, int, int, int], int | None],
]:
    """Load values and policy from a saved model."""

    data = load_json(path)

    values = {}
    policy = {}

    if "q_table" in data:
        records = data["q_table"]

        for record in records:
            state = parse_state(
                record["state"]
            )

            q_values = [
                float(value)
                for value in record.get(
                    "q_values",
                    [],
                )
            ]

            action = record.get(
                "best_action"
            )

            if action is None and q_values:
                action = max(
                    range(len(q_values)),
                    key=lambda index: q_values[index],
                )

            values[state] = (
                max(q_values)
                if q_values
                else 0.0
            )

            policy[state] = (
                int(action)
                if action is not None
                else None
            )

        return values, policy

    if "states" in data:
        records = data["states"]

        for record in records:
            state = parse_state(
                record["state"]
            )

            value = record.get(
                "value",
                record.get(
                    "state_value",
                    record.get("v", 0.0),
                ),
            )

            action = record.get(
                "best_action",
                record.get(
                    "policy_action",
                    record.get("action"),
                ),
            )

            values[state] = float(value)

            policy[state] = (
                int(action)
                if action is not None
                else None
            )

        return values, policy

    raise ValueError(
        f"Unknown model format: {path}"
    )


def create_environment() -> MazeEnv:
    """Create the source maze environment."""

    return MazeEnv(
        map_path=SOURCE_MAP,
        transition_seed=8,
        reward_mode="shaped",
        gamma=0.95,
    )


def is_wall(
    environment: MazeEnv,
    row: int,
    column: int,
) -> bool:
    """Check whether a cell is a wall."""

    return (
        environment.grid[row][column]
        == environment.wall_symbol
    )


def map_label(
    environment: MazeEnv,
    row: int,
    column: int,
) -> str:
    """Return a short cell label."""

    symbol = environment.grid[row][column]

    labels = {
        environment.start_symbol: "S",
        environment.key_symbol: "K",
        environment.door_symbol: "D",
        environment.goal_symbol: "G",
        environment.penalty_symbol: "P",
        environment.gate_symbol: "T",
    }

    return labels.get(symbol, "")


def prepare_axis(
    axis,
    environment: MazeEnv,
    title: str,
) -> None:
    """Apply common maze formatting."""

    axis.set_title(title)

    axis.set_xticks(
        range(environment.size)
    )
    axis.set_yticks(
        range(environment.size)
    )

    axis.set_xticklabels([])
    axis.set_yticklabels([])

    axis.set_xlim(
        -0.5,
        environment.size - 0.5,
    )

    axis.set_ylim(
        environment.size - 0.5,
        -0.5,
    )

    axis.grid(
        True,
        linewidth=0.5,
        alpha=0.4,
    )

    axis.set_aspect("equal")


def draw_map_labels(
    axis,
    environment: MazeEnv,
) -> None:
    """Draw walls and important map symbols."""

    for row in range(environment.size):
        for column in range(
            environment.size
        ):
            if is_wall(
                environment,
                row,
                column,
            ):
                axis.text(
                    column,
                    row,
                    "■",
                    ha="center",
                    va="center",
                    fontsize=15,
                )

                continue

            label = map_label(
                environment,
                row,
                column,
            )

            if label:
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )


def action_for_cell(
    policy: dict,
    row: int,
    column: int,
    has_key: int = 0,
    gate_phase: int = 0,
) -> int | None:
    """Find a representative action for one cell."""

    state = (
        row,
        column,
        has_key,
        gate_phase,
    )

    if state in policy:
        return policy[state]

    candidates = [
        (
            saved_state,
            action,
        )
        for saved_state, action in policy.items()
        if saved_state[0] == row
        and saved_state[1] == column
        and saved_state[2] == has_key
    ]

    if not candidates:
        candidates = [
            (
                saved_state,
                action,
            )
            for saved_state, action in policy.items()
            if saved_state[0] == row
            and saved_state[1] == column
        ]

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0][3]
    )

    return candidates[0][1]


def generate_value_heatmap(
    environment: MazeEnv,
    values: dict,
) -> Path:
    """Generate the Value Iteration heatmap."""

    matrix = []

    for row in range(environment.size):
        matrix_row = []

        for column in range(
            environment.size
        ):
            if is_wall(
                environment,
                row,
                column,
            ):
                matrix_row.append(
                    math.nan
                )

                continue

            cell_values = [
                value
                for state, value in values.items()
                if state[0] == row
                and state[1] == column
            ]

            matrix_row.append(
                max(cell_values)
                if cell_values
                else math.nan
            )

        matrix.append(matrix_row)

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    image = axis.imshow(matrix)

    prepare_axis(
        axis,
        environment,
        (
            "Value Iteration State-Value Heatmap\n"
            "Maximum over key and gate states"
        ),
    )

    draw_map_labels(
        axis,
        environment,
    )

    figure.colorbar(
        image,
        ax=axis,
        label="State value",
    )

    output_path = (
        FIGURES_DIR
        / "value_iteration_heatmap.png"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def generate_policy_figure(
    environment: MazeEnv,
    policy: dict,
) -> Path:
    """Generate the Value Iteration policy arrows."""

    background = []

    for row in range(environment.size):
        background_row = []

        for column in range(
            environment.size
        ):
            background_row.append(
                math.nan
                if is_wall(
                    environment,
                    row,
                    column,
                )
                else 0.0
            )

        background.append(background_row)

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    axis.imshow(
        background,
        alpha=0.15,
    )

    prepare_axis(
        axis,
        environment,
        (
            "Value Iteration Greedy Policy\n"
            "has_key=0, gate_phase=0"
        ),
    )

    draw_map_labels(
        axis,
        environment,
    )

    for row in range(environment.size):
        for column in range(
            environment.size
        ):
            if is_wall(
                environment,
                row,
                column,
            ):
                continue

            if (
                environment.grid[row][column]
                == environment.goal_symbol
            ):
                continue

            action = action_for_cell(
                policy,
                row,
                column,
            )

            if action in ARROWS:
                axis.text(
                    column,
                    row,
                    ARROWS[action],
                    ha="center",
                    va="center",
                    fontsize=13,
                )

    output_path = (
        FIGURES_DIR
        / "value_iteration_policy.png"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def simulate_visitation(
    policy: dict,
    episodes: int = 200,
) -> tuple[MazeEnv, Counter]:
    """Count visited cells during greedy evaluation."""

    environment = create_environment()

    visits = Counter()

    for episode in range(episodes):
        state, _ = environment.reset(
            seed=8 + episode
        )

        visits[
            (
                state[0],
                state[1],
            )
        ] += 1

        while not environment.done:
            action = policy.get(state)

            if action is None:
                break

            (
                state,
                _,
                terminated,
                truncated,
                _,
            ) = environment.step(action)

            visits[
                (
                    state[0],
                    state[1],
                )
            ] += 1

            if terminated or truncated:
                break

    return environment, visits


def generate_visitation_heatmap(
    policy: dict,
) -> Path:
    """Generate a Q-Learning visitation heatmap."""

    environment, visits = (
        simulate_visitation(
            policy,
            episodes=200,
        )
    )

    matrix = []

    for row in range(environment.size):
        matrix_row = []

        for column in range(
            environment.size
        ):
            if is_wall(
                environment,
                row,
                column,
            ):
                matrix_row.append(
                    math.nan
                )
            else:
                matrix_row.append(
                    visits[
                        (
                            row,
                            column,
                        )
                    ]
                )

        matrix.append(matrix_row)

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    image = axis.imshow(matrix)

    prepare_axis(
        axis,
        environment,
        (
            "Q-Learning State Visitation Heatmap\n"
            "200 greedy evaluation episodes"
        ),
    )

    draw_map_labels(
        axis,
        environment,
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Visit count",
    )

    output_path = (
        FIGURES_DIR
        / "q_learning_visitation.png"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def generate_policy_difference(
    environment: MazeEnv,
    value_policy: dict,
    q_policy: dict,
) -> Path:
    """Compare Value Iteration and Q-Learning."""

    matrix = []

    same_count = 0
    comparison_count = 0

    for row in range(environment.size):
        matrix_row = []

        for column in range(
            environment.size
        ):
            if is_wall(
                environment,
                row,
                column,
            ):
                matrix_row.append(
                    math.nan
                )

                continue

            first_action = action_for_cell(
                value_policy,
                row,
                column,
            )

            second_action = action_for_cell(
                q_policy,
                row,
                column,
            )

            if (
                first_action is None
                or second_action is None
            ):
                matrix_row.append(
                    math.nan
                )

                continue

            comparison_count += 1

            actions_match = (
                first_action
                == second_action
            )

            if actions_match:
                same_count += 1

            matrix_row.append(
                0.0
                if actions_match
                else 1.0
            )

        matrix.append(matrix_row)

    agreement = (
        100.0
        * same_count
        / comparison_count
        if comparison_count
        else 0.0
    )

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    image = axis.imshow(
        matrix,
        vmin=0.0,
        vmax=1.0,
    )

    prepare_axis(
        axis,
        environment,
        (
            "Policy Difference: Value Iteration vs Q-Learning\n"
            f"Agreement = {agreement:.2f}%"
        ),
    )

    draw_map_labels(
        axis,
        environment,
    )

    for row in range(environment.size):
        for column in range(
            environment.size
        ):
            value = matrix[row][column]

            if math.isnan(value):
                continue

            if value == 1.0:
                axis.text(
                    column,
                    row,
                    "×",
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        ticks=[0.0, 1.0],
    )

    colorbar.ax.set_yticklabels(
        [
            "Same",
            "Different",
        ]
    )

    output_path = (
        FIGURES_DIR
        / "policy_difference.png"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def find_field(
    row: dict[str, str],
    names: tuple[str, ...],
) -> str | None:
    """Find a CSV column using possible names."""

    normalized = {
        key.lower(): key
        for key in row
    }

    for name in names:
        if name in row:
            return name

        if name.lower() in normalized:
            return normalized[
                name.lower()
            ]

    return None


def read_text(
    row: dict[str, str],
    names: tuple[str, ...],
    default: str,
) -> str:
    """Read a text field from a CSV row."""

    field = find_field(
        row,
        names,
    )

    if field is None:
        return default

    return row.get(
        field,
        default,
    ).strip()


def read_number(
    row: dict[str, str],
    names: tuple[str, ...],
) -> float:
    """Read a numeric field from a CSV row."""

    field = find_field(
        row,
        names,
    )

    if field is None:
        return 0.0

    value = (
        row.get(field, "0")
        .replace("%", "")
        .strip()
    )

    try:
        number = float(value)
    except ValueError:
        return 0.0

    if 0.0 <= number <= 1.0:
        number *= 100.0

    return number


def generate_transfer_figure() -> Path | None:
    """Compare transfer performance before and after training."""

    if not TRANSFER_CSV.exists():
        print(
            "Transfer comparison CSV was not found."
        )

        return None

    with TRANSFER_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    labels = []
    initial_values = []
    final_values = []

    for row in rows:
        target = read_text(
            row,
            (
                "target",
                "target_type",
                "environment",
            ),
            "target",
        )

        scenario = read_text(
            row,
            (
                "scenario",
                "method",
                "transfer_scenario",
            ),
            "scenario",
        )

        initial_success = read_number(
            row,
            (
                "initial_success_rate",
                "initial_success",
                "initial",
            ),
        )

        final_success = read_number(
            row,
            (
                "final_success_rate",
                "evaluation_success_rate",
                "final_success",
                "final",
            ),
        )

        labels.append(
            f"{target}\n{scenario}"
        )

        initial_values.append(
            initial_success
        )

        final_values.append(
            final_success
        )

    positions = list(
        range(len(labels))
    )

    width = 0.38

    figure, axis = plt.subplots(
        figsize=(15, 7)
    )

    axis.bar(
        [
            position - width / 2
            for position in positions
        ],
        initial_values,
        width=width,
        label="Initial success",
    )

    axis.bar(
        [
            position + width / 2
            for position in positions
        ],
        final_values,
        width=width,
        label="Final success",
    )

    axis.set_title(
        "Transfer Learning: Before and After Training"
    )

    axis.set_ylabel(
        "Success rate (%)"
    )

    axis.set_ylim(
        0,
        105,
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.grid(
        axis="y",
        alpha=0.35,
    )

    axis.legend()

    output_path = (
        FIGURES_DIR
        / "transfer_before_after.png"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main() -> None:
    """Generate all final project figures."""

    environment = create_environment()

    (
        value_values,
        value_policy,
    ) = load_model(VALUE_MODEL)

    (
        _,
        q_policy,
    ) = load_model(Q_MODEL)

    generated_files = []

    generated_files.append(
        generate_value_heatmap(
            environment,
            value_values,
        )
    )

    generated_files.append(
        generate_policy_figure(
            environment,
            value_policy,
        )
    )

    generated_files.append(
        generate_visitation_heatmap(
            q_policy
        )
    )

    generated_files.append(
        generate_policy_difference(
            environment,
            value_policy,
            q_policy,
        )
    )

    transfer_figure = (
        generate_transfer_figure()
    )

    if transfer_figure is not None:
        generated_files.append(
            transfer_figure
        )

    print()
    print(
        "Figure generation finished."
    )

    for path in generated_files:
        print(
            "Saved:",
            path,
        )


if __name__ == "__main__":
    main()