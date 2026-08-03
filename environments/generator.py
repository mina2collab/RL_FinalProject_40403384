from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path


STUDENT_ID = "40403384"
BASE_SEED = int(STUDENT_ID[-2])
MAZE_SIZE = 15 + (BASE_SEED % 4)

WALL = "#"
EMPTY = "."
START = "S"
KEY = "K"
DOOR = "D"
GOAL = "G"
PENALTY = "P"
PERIODIC_GATE = "T"


def line_path(
    start: tuple[int, int],
    end: tuple[int, int],
) -> set[tuple[int, int]]:
    """Create a horizontal or vertical path between two cells."""

    row1, col1 = start
    row2, col2 = end
    cells: set[tuple[int, int]] = set()

    if row1 == row2:
        for col in range(min(col1, col2), max(col1, col2) + 1):
            cells.add((row1, col))

    elif col1 == col2:
        for row in range(min(row1, row2), max(row1, row2) + 1):
            cells.add((row, col1))

    else:
        raise ValueError(
            "line_path only accepts horizontal or vertical segments."
        )

    return cells


def bfs(
    grid: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    allow_door: bool,
) -> list[tuple[int, int]] | None:
    """Return one shortest path or None when no path exists."""

    rows = len(grid)
    cols = len(grid[0])

    blocked = {WALL}

    if not allow_door:
        blocked.add(DOOR)

    queue = deque([start])

    parent: dict[
        tuple[int, int],
        tuple[int, int] | None,
    ] = {start: None}

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

        row, col = current

        for row_change, col_change in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            new_row = row + row_change
            new_col = col + col_change
            next_cell = (new_row, new_col)

            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            if next_cell in parent:
                continue

            if grid[new_row][new_col] in blocked:
                continue

            parent[next_cell] = current
            queue.append(next_cell)

    return None


def generate_maze(seed: int = BASE_SEED) -> dict:
    """Generate a reproducible dynamic maze."""

    random_generator = random.Random(seed)
    size = MAZE_SIZE

    grid = [
        [EMPTY for _ in range(size)]
        for _ in range(size)
    ]

    # Create boundary walls.
    for index in range(size):
        grid[0][index] = WALL
        grid[size - 1][index] = WALL
        grid[index][0] = WALL
        grid[index][size - 1] = WALL

    start = (1, 1)
    key = (3, size - 3)
    door = (size // 2, size // 2)
    goal = (size - 2, size - 2)
    periodic_gate = (3, size // 2)

    # This barrier makes passing through the door necessary.
    barrier_row = size // 2

    for col in range(1, size - 1):
        grid[barrier_row][col] = WALL

    grid[door[0]][door[1]] = DOOR

    # Guaranteed route:
    # Start -> Key -> Door -> Goal
    reserved_cells: set[tuple[int, int]] = set()

    reserved_cells |= line_path(start, (3, 1))
    reserved_cells |= line_path((3, 1), key)

    reserved_cells |= line_path(key, (6, key[1]))
    reserved_cells |= line_path(
        (6, key[1]),
        (6, door[1]),
    )
    reserved_cells |= line_path(
        (6, door[1]),
        door,
    )

    reserved_cells |= line_path(
        door,
        (size - 2, door[1]),
    )
    reserved_cells |= line_path(
        (size - 2, door[1]),
        goal,
    )

    # Create reproducible random interior walls.
    for row in range(1, size - 1):
        for col in range(1, size - 1):
            cell = (row, col)

            if cell in reserved_cells:
                continue

            if row == barrier_row:
                continue

            if random_generator.random() < 0.10:
                grid[row][col] = WALL

    # Place important cells.
    grid[start[0]][start[1]] = START
    grid[key[0]][key[1]] = KEY
    grid[door[0]][door[1]] = DOOR
    grid[goal[0]][goal[1]] = GOAL

    grid[periodic_gate[0]][periodic_gate[1]] = PERIODIC_GATE

    # Select five reproducible penalty cells.
    penalty_candidates = [
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if grid[row][col] == EMPTY
        and (row, col) not in reserved_cells
    ]

    if len(penalty_candidates) < 5:
        raise RuntimeError(
            "Not enough free cells for penalties."
        )

    penalty_cells = random_generator.sample(
        penalty_candidates,
        5,
    )

    for row, col in penalty_cells:
        grid[row][col] = PENALTY

    # Validate Start -> Key while the door is closed.
    path_start_to_key = bfs(
        grid=grid,
        start=start,
        goal=key,
        allow_door=False,
    )

    # Validate Key -> Goal while the door is available.
    path_key_to_goal = bfs(
        grid=grid,
        start=key,
        goal=goal,
        allow_door=True,
    )

    if path_start_to_key is None:
        raise RuntimeError(
            "No valid path from start to key."
        )

    if path_key_to_goal is None:
        raise RuntimeError(
            "No valid path from key to goal."
        )

    wall_count = sum(
        cell == WALL
        for row in grid
        for cell in row
    )

    wall_ratio = wall_count / (size * size)

    if wall_ratio < 0.15:
        raise RuntimeError(
            "Wall ratio is below 15 percent."
        )

    maze_data = {
        "student_id": STUDENT_ID,
        "seed": seed,
        "size": size,

        "symbols": {
            "wall": WALL,
            "empty": EMPTY,
            "start": START,
            "key": KEY,
            "door": DOOR,
            "goal": GOAL,
            "penalty": PENALTY,
            "periodic_gate": PERIODIC_GATE,
        },

        "positions": {
            "start": list(start),
            "key": list(key),
            "door": list(door),
            "goal": list(goal),
            "periodic_gate": list(periodic_gate),
            "penalties": [
                list(cell)
                for cell in penalty_cells
            ],
        },

        "dynamic_feature": {
            "type": "periodic_gate",
            "period": 2,
            "open_phase": 0,
        },

        "validation": {
            "wall_count": wall_count,
            "wall_ratio": wall_ratio,
            "path_start_to_key_length":
                len(path_start_to_key) - 1,
            "path_key_to_goal_length":
                len(path_key_to_goal) - 1,
        },

        "grid": [
            "".join(row)
            for row in grid
        ],
    }

    return maze_data


def save_maze(maze_data: dict) -> Path:
    """Save the final map in JSON format."""

    output_path = (
        Path(__file__).resolve().parent
        / "maps"
        / "source_map.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            maze_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def print_maze(maze_data: dict) -> None:
    """Print the maze in the Python shell."""

    for row in maze_data["grid"]:
        print(row)


if __name__ == "__main__":
    maze = generate_maze()
    saved_path = save_maze(maze)

    print_maze(maze)

    print()
    print("Maze saved to:", saved_path)

    print(
        "Wall ratio:",
        f'{maze["validation"]["wall_ratio"]:.2%}',
    )

    print(
        "Start -> Key path length:",
        maze["validation"]["path_start_to_key_length"],
    )

    print(
        "Key -> Goal path length:",
        maze["validation"]["path_key_to_goal_length"],
    )