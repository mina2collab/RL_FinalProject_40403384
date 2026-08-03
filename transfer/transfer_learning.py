from __future__ import annotations

import copy
import json
import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.q_learning import QLearningAgent
from environments.maze import ACTION_NAMES, MazeEnv


State = tuple[int, int, int, int]

SOURCE_MAP_PATH = (
    PROJECT_ROOT
    / "environments"
    / "maps"
    / "source_map.json"
)

SIMILAR_MAP_PATH = (
    PROJECT_ROOT
    / "environments"
    / "maps"
    / "target_similar.json"
)

DIFFERENT_MAP_PATH = (
    PROJECT_ROOT
    / "environments"
    / "maps"
    / "target_different.json"
)

SOURCE_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "models"
    / "q_learning_linear.json"
)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    """Save a dictionary as JSON."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path


def shortest_path(
    grid: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    wall_symbol: str,
    door_symbol: str,
    allow_door: bool,
) -> list[tuple[int, int]] | None:
    """Return one shortest BFS path."""

    blocked_symbols = {wall_symbol}

    if not allow_door:
        blocked_symbols.add(door_symbol)

    rows = len(grid)
    columns = len(grid[0])

    queue = deque([start])

    parent: dict[
        tuple[int, int],
        tuple[int, int] | None,
    ] = {
        start: None,
    }

    while queue:
        current = queue.popleft()

        if current == goal:
            path: list[tuple[int, int]] = []

            node: tuple[int, int] | None = current

            while node is not None:
                path.append(node)
                node = parent[node]

            path.reverse()
            return path

        row, column = current

        for row_change, column_change in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            next_row = row + row_change
            next_column = column + column_change

            next_position = (
                next_row,
                next_column,
            )

            if not (
                0 <= next_row < rows
                and 0 <= next_column < columns
            ):
                continue

            if next_position in parent:
                continue

            if (
                grid[next_row][next_column]
                in blocked_symbols
            ):
                continue

            parent[next_position] = current
            queue.append(next_position)

    return None


def reachable_cells(
    grid: list[list[str]],
    start: tuple[int, int],
    wall_symbol: str,
    door_symbol: str,
    allow_door: bool,
) -> set[tuple[int, int]]:
    """Return all cells reachable from a start cell."""

    blocked_symbols = {wall_symbol}

    if not allow_door:
        blocked_symbols.add(door_symbol)

    rows = len(grid)
    columns = len(grid[0])

    queue = deque([start])
    visited = {start}

    while queue:
        row, column = queue.popleft()

        for row_change, column_change in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            next_row = row + row_change
            next_column = column + column_change

            next_position = (
                next_row,
                next_column,
            )

            if not (
                0 <= next_row < rows
                and 0 <= next_column < columns
            ):
                continue

            if next_position in visited:
                continue

            if (
                grid[next_row][next_column]
                in blocked_symbols
            ):
                continue

            visited.add(next_position)
            queue.append(next_position)

    return visited


def move_interior_walls(
    grid: list[list[str]],
    movement_ratio: float,
    random_generator: random.Random,
    wall_symbol: str,
    empty_symbol: str,
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
    """
    Move a percentage of interior walls.

    The number of removed walls equals the number of
    newly created walls.
    """

    size = len(grid)

    wall_cells = [
        (row, column)
        for row in range(1, size - 1)
        for column in range(1, size - 1)
        if grid[row][column] == wall_symbol
    ]

    empty_cells = [
        (row, column)
        for row in range(1, size - 1)
        for column in range(1, size - 1)
        if grid[row][column] == empty_symbol
    ]

    move_count = max(
        1,
        round(
            movement_ratio * len(wall_cells)
        ),
    )

    if move_count > len(empty_cells):
        raise RuntimeError(
            "Not enough empty cells for moving walls."
        )

    removed_walls = set(
        random_generator.sample(
            wall_cells,
            move_count,
        )
    )

    added_walls = set(
        random_generator.sample(
            empty_cells,
            move_count,
        )
    )

    for row, column in removed_walls:
        grid[row][column] = empty_symbol

    for row, column in added_walls:
        grid[row][column] = wall_symbol

    return removed_walls, added_walls


def validate_target_map(
    grid: list[list[str]],
    start: tuple[int, int],
    key: tuple[int, int],
    goal: tuple[int, int],
    wall_symbol: str,
    door_symbol: str,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
] | None:
    """Validate Start -> Key -> Goal using BFS."""

    start_to_key = shortest_path(
        grid=grid,
        start=start,
        goal=key,
        wall_symbol=wall_symbol,
        door_symbol=door_symbol,
        allow_door=False,
    )

    if start_to_key is None:
        return None

    key_to_goal = shortest_path(
        grid=grid,
        start=key,
        goal=goal,
        wall_symbol=wall_symbol,
        door_symbol=door_symbol,
        allow_door=True,
    )

    if key_to_goal is None:
        return None

    return start_to_key, key_to_goal


def build_target_map(
    source_data: dict[str, Any],
    target_type: str,
    seed: int,
) -> dict[str, Any]:
    """Create one reproducible destination map."""

    if target_type not in {
        "similar",
        "different",
    }:
        raise ValueError(
            "target_type must be similar or different."
        )

    symbols = source_data["symbols"]

    wall_symbol = symbols["wall"]
    empty_symbol = symbols["empty"]
    key_symbol = symbols["key"]
    penalty_symbol = symbols["penalty"]
    door_symbol = symbols["door"]

    source_positions = source_data["positions"]

    start = tuple(source_positions["start"])
    source_key = tuple(source_positions["key"])
    goal = tuple(source_positions["goal"])
    door = tuple(source_positions["door"])
    periodic_gate = tuple(
        source_positions["periodic_gate"]
    )

    source_grid = [
        list(row)
        for row in source_data["grid"]
    ]

    original_interior_walls = {
        (row, column)
        for row in range(1, len(source_grid) - 1)
        for column in range(1, len(source_grid) - 1)
        if source_grid[row][column] == wall_symbol
    }

    movement_ratio = (
        0.15
        if target_type == "similar"
        else 0.40
    )

    for attempt in range(500):
        random_generator = random.Random(
            seed + 997 * attempt
        )

        grid = copy.deepcopy(source_grid)

        removed_walls, added_walls = (
            move_interior_walls(
                grid=grid,
                movement_ratio=movement_ratio,
                random_generator=random_generator,
                wall_symbol=wall_symbol,
                empty_symbol=empty_symbol,
            )
        )

        key = source_key

        if target_type == "different":
            # Remove the old key.
            grid[source_key[0]][source_key[1]] = (
                empty_symbol
            )

            reachable_without_key = reachable_cells(
                grid=grid,
                start=start,
                wall_symbol=wall_symbol,
                door_symbol=door_symbol,
                allow_door=False,
            )

            forbidden_positions = {
                start,
                source_key,
                goal,
                door,
                periodic_gate,
            }

            forbidden_positions.update(
                tuple(position)
                for position
                in source_positions["penalties"]
            )

            key_candidates = [
                position
                for position in reachable_without_key
                if position not in forbidden_positions
                and grid[position[0]][position[1]]
                == empty_symbol
                and (
                    abs(position[0] - source_key[0])
                    + abs(position[1] - source_key[1])
                    >= 5
                )
            ]

            if not key_candidates:
                continue

            key = random_generator.choice(
                key_candidates
            )

            grid[key[0]][key[1]] = key_symbol

            # Add three new penalty cells.
            penalty_candidates = [
                (row, column)
                for row in range(1, len(grid) - 1)
                for column in range(1, len(grid) - 1)
                if grid[row][column] == empty_symbol
                and (row, column)
                not in {
                    start,
                    key,
                    goal,
                    door,
                    periodic_gate,
                }
            ]

            if len(penalty_candidates) < 3:
                continue

            new_penalties = random_generator.sample(
                penalty_candidates,
                3,
            )

            for row, column in new_penalties:
                grid[row][column] = penalty_symbol

        validation = validate_target_map(
            grid=grid,
            start=start,
            key=key,
            goal=goal,
            wall_symbol=wall_symbol,
            door_symbol=door_symbol,
        )

        if validation is None:
            continue

        start_to_key, key_to_goal = validation

        target_data = copy.deepcopy(source_data)

        target_data["seed"] = seed

        target_data["grid"] = [
            "".join(row)
            for row in grid
        ]

        target_data["positions"]["key"] = list(key)

        target_data["positions"]["penalties"] = [
            [row, column]
            for row in range(len(grid))
            for column in range(len(grid[0]))
            if grid[row][column] == penalty_symbol
        ]

        final_interior_walls = {
            (row, column)
            for row in range(1, len(grid) - 1)
            for column in range(1, len(grid) - 1)
            if grid[row][column] == wall_symbol
        }

        wall_count = sum(
            cell == wall_symbol
            for row in grid
            for cell in row
        )

        target_data["validation"] = {
            "wall_count": wall_count,
            "wall_ratio": (
                wall_count
                / (len(grid) * len(grid[0]))
            ),
            "path_start_to_key_length": (
                len(start_to_key) - 1
            ),
            "path_key_to_goal_length": (
                len(key_to_goal) - 1
            ),
        }

        moved_wall_count = len(
            original_interior_walls
            - final_interior_walls
        )

        changed_wall_cells = len(
            original_interior_walls
            ^ final_interior_walls
        )

        target_data["transfer_metadata"] = {
            "target_type": target_type,
            "generation_attempt": attempt + 1,
            "movement_ratio_requested":
                movement_ratio,
            "moved_wall_count":
                moved_wall_count,
            "moved_wall_percentage": (
                100.0
                * moved_wall_count
                / len(original_interior_walls)
            ),
            "changed_wall_cells":
                changed_wall_cells,
            "added_wall_count":
                len(added_walls),
            "removed_wall_count":
                len(removed_walls),
            "key_moved":
                key != source_key,
            "new_penalty_count": (
                0
                if target_type == "similar"
                else 3
            ),
        }

        return target_data

    raise RuntimeError(
        f"Could not generate a valid {target_type} map."
    )


def create_target_maps() -> dict[str, Path]:
    """Generate and save both target environments."""

    if not SOURCE_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Source map not found: {SOURCE_MAP_PATH}"
        )

    source_data = load_json(
        SOURCE_MAP_PATH
    )

    similar_data = build_target_map(
        source_data=source_data,
        target_type="similar",
        seed=108,
    )

    different_data = build_target_map(
        source_data=source_data,
        target_type="different",
        seed=208,
    )

    save_json(
        similar_data,
        SIMILAR_MAP_PATH,
    )

    save_json(
        different_data,
        DIFFERENT_MAP_PATH,
    )

    return {
        "similar": SIMILAR_MAP_PATH,
        "different": DIFFERENT_MAP_PATH,
    }


def load_source_q_table(
    model_path: str | Path = SOURCE_MODEL_PATH,
) -> dict[State, list[float]]:
    """Load the source Q table from a saved model."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            "Source Q-Learning model was not found: "
            f"{model_path}"
        )

    model_data = load_json(model_path)

    q_table: dict[State, list[float]] = {}

    for record in model_data["q_table"]:
        state: State = tuple(record["state"])

        q_table[state] = [
            float(value)
            for value in record["q_values"]
        ]

    return q_table


def neighborhood_signature(
    environment: MazeEnv,
    state: State,
    radius: int = 1,
) -> tuple[str, ...]:
    """Return the local grid neighborhood of a state."""

    row, column, _, _ = state

    signature: list[str] = []

    for row_change in range(
        -radius,
        radius + 1,
    ):
        for column_change in range(
            -radius,
            radius + 1,
        ):
            neighbor_row = row + row_change
            neighbor_column = (
                column + column_change
            )

            if not (
                0 <= neighbor_row < environment.size
                and 0 <= neighbor_column
                < environment.size
            ):
                signature.append("OUTSIDE")
            else:
                signature.append(
                    environment.grid[
                        neighbor_row
                    ][neighbor_column]
                )

    return tuple(signature)


def initialize_target_q_table(
    target_agent: QLearningAgent,
    source_q_table: dict[
        State,
        list[float],
    ],
    mode: str,
    source_environment: MazeEnv,
    beta: float = 0.50,
) -> dict[str, Any]:
    """
    Initialize a target Q table.

    Modes:
        scratch
        full
        scaled
        selective
    """

    if mode not in {
        "scratch",
        "full",
        "scaled",
        "selective",
    }:
        raise ValueError(
            "Invalid transfer mode."
        )

    if not 0 <= beta <= 1:
        raise ValueError(
            "beta must be between zero and one."
        )

    transferred_states = 0
    skipped_changed_neighborhoods = 0
    skipped_missing_states = 0

    if mode == "scratch":
        return {
            "mode": mode,
            "beta": 0.0,
            "transferred_states": 0,
            "total_target_states":
                len(target_agent.q_table),
            "transfer_percentage": 0.0,
            "skipped_changed_neighborhoods": 0,
            "skipped_missing_states": 0,
        }

    for state in target_agent.q_table:
        source_values = source_q_table.get(state)

        if source_values is None:
            skipped_missing_states += 1
            continue

        if mode == "selective":
            source_signature = (
                neighborhood_signature(
                    source_environment,
                    state,
                )
            )

            target_signature = (
                neighborhood_signature(
                    target_agent.environment,
                    state,
                )
            )

            if source_signature != target_signature:
                skipped_changed_neighborhoods += 1
                continue

        scale = beta if mode == "scaled" else 1.0

        target_agent.q_table[state] = [
            scale * value
            for value in source_values
        ]

        transferred_states += 1

    return {
        "mode": mode,
        "beta": (
            beta
            if mode == "scaled"
            else 1.0
        ),
        "transferred_states":
            transferred_states,
        "total_target_states":
            len(target_agent.q_table),
        "transfer_percentage": (
            100.0
            * transferred_states
            / len(target_agent.q_table)
        ),
        "skipped_changed_neighborhoods":
            skipped_changed_neighborhoods,
        "skipped_missing_states":
            skipped_missing_states,
    }


def expected_immediate_reward(
    environment: MazeEnv,
    state: State,
    action: int,
) -> float:
    """Calculate expected one-step reward."""

    outcomes = environment.transition_outcomes(
        state,
        action,
    )

    return sum(
        float(outcome["probability"])
        * float(outcome["reward"])
        for outcome in outcomes
    )


def find_negative_transfer_candidate(
    source_q_table: dict[
        State,
        list[float],
    ],
    target_environment: MazeEnv,
    source_environment: MazeEnv,
) -> dict[str, Any] | None:
    """
    Find a state where the source greedy action is poor
    after the local environment has changed.
    """

    best_candidate: dict[str, Any] | None = None
    largest_gap = 0.0

    for state in target_environment.get_all_states():
        if target_environment.is_terminal_state(state):
            continue

        source_values = source_q_table.get(state)

        if source_values is None:
            continue

        source_action = max(
            range(len(source_values)),
            key=lambda action: source_values[action],
        )

        target_rewards = {
            action: expected_immediate_reward(
                target_environment,
                state,
                action,
            )
            for action in ACTION_NAMES
        }

        target_best_action = max(
            target_rewards,
            key=target_rewards.get,
        )

        reward_gap = (
            target_rewards[target_best_action]
            - target_rewards[source_action]
        )

        local_changed = (
            neighborhood_signature(
                source_environment,
                state,
            )
            != neighborhood_signature(
                target_environment,
                state,
            )
        )

        if (
            local_changed
            and target_best_action != source_action
            and reward_gap > largest_gap
        ):
            largest_gap = reward_gap

            best_candidate = {
                "state": list(state),
                "source_q_values":
                    source_values,
                "source_greedy_action":
                    source_action,
                "source_greedy_action_name":
                    ACTION_NAMES[source_action],
                "target_best_immediate_action":
                    target_best_action,
                "target_best_immediate_action_name":
                    ACTION_NAMES[
                        target_best_action
                    ],
                "target_expected_rewards": {
                    ACTION_NAMES[action]:
                        reward
                    for action, reward
                    in target_rewards.items()
                },
                "immediate_reward_gap":
                    reward_gap,
                "local_neighborhood_changed":
                    local_changed,
            }

    return best_candidate


def demonstrate_transfer_initialization() -> Path:
    """Generate maps and prepare transfer scenarios."""

    target_paths = create_target_maps()

    source_environment = MazeEnv(
        map_path=SOURCE_MAP_PATH,
        transition_seed=8,
        reward_mode="shaped",
        gamma=0.95,
    )

    source_q_table = load_source_q_table()

    summaries: list[dict[str, Any]] = []
    negative_candidates: dict[str, Any] = {}

    scenario_settings = [
        ("scratch", 0.0),
        ("full", 1.0),
        ("scaled", 0.25),
        ("scaled", 0.50),
        ("scaled", 0.75),
        ("selective", 1.0),
    ]

    for target_name, target_path in target_paths.items():
        target_environment = MazeEnv(
            map_path=target_path,
            transition_seed=8,
            reward_mode="shaped",
            gamma=0.95,
        )

        for mode, beta in scenario_settings:
            agent = QLearningAgent(
                environment=target_environment,
                alpha=0.10,
                gamma=0.95,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_episodes=500,
                epsilon_schedule="linear",
                seed=8,
            )

            transfer_summary = (
                initialize_target_q_table(
                    target_agent=agent,
                    source_q_table=source_q_table,
                    mode=mode,
                    source_environment=(
                        source_environment
                    ),
                    beta=beta,
                )
            )

            transfer_summary["target"] = target_name

            summaries.append(
                transfer_summary
            )

        negative_candidate = (
            find_negative_transfer_candidate(
                source_q_table=source_q_table,
                target_environment=(
                    target_environment
                ),
                source_environment=(
                    source_environment
                ),
            )
        )

        negative_candidates[target_name] = (
            negative_candidate
        )

    output_data = {
        "student_id": "40403384",
        "source_map": str(
            SOURCE_MAP_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "source_model": str(
            SOURCE_MODEL_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "target_maps": {
            name: str(
                path.relative_to(
                    PROJECT_ROOT
                )
            )
            for name, path in target_paths.items()
        },
        "transfer_scenarios": summaries,
        "negative_transfer_candidates":
            negative_candidates,
    }

    output_path = (
        PROJECT_ROOT
        / "results"
        / "raw_data"
        / "transfer_initialization_summary.json"
    )

    save_json(
        output_data,
        output_path,
    )

    return output_path


if __name__ == "__main__":
    summary_path = (
        demonstrate_transfer_initialization()
    )

    similar_data = load_json(
        SIMILAR_MAP_PATH
    )

    different_data = load_json(
        DIFFERENT_MAP_PATH
    )

    print("Transfer-learning preparation finished.")

    print()
    print("Similar target:")

    print(
        "Moved walls:",
        f'{similar_data["transfer_metadata"]["moved_wall_percentage"]:.2f}%',
    )

    print(
        "Key moved:",
        similar_data["transfer_metadata"]["key_moved"],
    )

    print(
        "Start -> Key path:",
        similar_data["validation"][
            "path_start_to_key_length"
        ],
    )

    print(
        "Key -> Goal path:",
        similar_data["validation"][
            "path_key_to_goal_length"
        ],
    )

    print()
    print("Different target:")

    print(
        "Moved walls:",
        f'{different_data["transfer_metadata"]["moved_wall_percentage"]:.2f}%',
    )

    print(
        "Key moved:",
        different_data["transfer_metadata"]["key_moved"],
    )

    print(
        "New penalties:",
        different_data["transfer_metadata"][
            "new_penalty_count"
        ],
    )

    print(
        "Start -> Key path:",
        different_data["validation"][
            "path_start_to_key_length"
        ],
    )

    print(
        "Key -> Goal path:",
        different_data["validation"][
            "path_key_to_goal_length"
        ],
    )

    print()
    print("Summary saved to:", summary_path)