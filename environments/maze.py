from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


# State = (row, column, has_key, gate_phase)
State = tuple[int, int, int, int]


# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}

ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Two perpendicular directions for every intended action.
PERPENDICULAR_ACTIONS = {
    UP: (LEFT, RIGHT),
    DOWN: (RIGHT, LEFT),
    LEFT: (DOWN, UP),
    RIGHT: (UP, DOWN),
}


class MazeEnv:
    """Dynamic stochastic maze environment."""

    def __init__(
        self,
        map_path: str | Path | None = None,
        transition_seed: int = 8,
        reward_mode: str = "sparse",
        gamma: float = 0.95,
    ) -> None:
        if reward_mode not in {"sparse", "shaped"}:
            raise ValueError(
                "reward_mode must be 'sparse' or 'shaped'."
            )

        if map_path is None:
            map_path = (
                Path(__file__).resolve().parent
                / "maps"
                / "source_map.json"
            )

        self.map_path = Path(map_path)

        if not self.map_path.exists():
            raise FileNotFoundError(
                f"Map file was not found: {self.map_path}"
            )

        with self.map_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.maze_data = json.load(file)

        self.size = int(self.maze_data["size"])
        self.grid = [
            list(row)
            for row in self.maze_data["grid"]
        ]

        symbols = self.maze_data["symbols"]

        self.wall_symbol = symbols["wall"]
        self.empty_symbol = symbols["empty"]
        self.start_symbol = symbols["start"]
        self.key_symbol = symbols["key"]
        self.door_symbol = symbols["door"]
        self.goal_symbol = symbols["goal"]
        self.penalty_symbol = symbols["penalty"]
        self.gate_symbol = symbols["periodic_gate"]

        positions = self.maze_data["positions"]

        self.start_position = tuple(positions["start"])
        self.key_position = tuple(positions["key"])
        self.door_position = tuple(positions["door"])
        self.goal_position = tuple(positions["goal"])
        self.gate_position = tuple(
            positions["periodic_gate"]
        )

        dynamic_feature = self.maze_data["dynamic_feature"]

        self.gate_period = int(
            dynamic_feature["period"]
        )

        self.gate_open_phase = int(
            dynamic_feature["open_phase"]
        )

        self.reward_mode = reward_mode
        self.gamma = gamma

        self.random_generator = random.Random(
            transition_seed
        )

        self.passable_cell_count = sum(
            cell != self.wall_symbol
            for row in self.grid
            for cell in row
        )

        # Recommended limit: three times the passable cells.
        self.max_steps = 3 * self.passable_cell_count

        self.position = self.start_position
        self.has_key = False
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        self.event_log: list[dict[str, Any]] = []

    @property
    def gate_phase(self) -> int:
        """Return the current periodic-gate phase."""

        return self.step_count % self.gate_period

    def is_gate_open(
        self,
        phase: int | None = None,
    ) -> bool:
        """Check whether the periodic gate is open."""

        if phase is None:
            phase = self.gate_phase

        return phase == self.gate_open_phase

    def get_state(self) -> State:
        """Return the current Markov state."""

        row, column = self.position

        return (
            row,
            column,
            int(self.has_key),
            self.gate_phase,
        )

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[State, dict[str, Any]]:
        """Start a new episode."""

        if seed is not None:
            self.random_generator.seed(seed)

        self.position = self.start_position
        self.has_key = False
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        self.event_log = []

        state = self.get_state()

        info = {
            "event": "episode_reset",
            "max_steps": self.max_steps,
            "gate_open": self.is_gate_open(),
            "reward_mode": self.reward_mode,
        }

        return state, info

    def _sample_actual_action(
        self,
        intended_action: int,
    ) -> int:
        """
        Execute intended action with probability 0.8.

        Each perpendicular action is executed with
        probability 0.1.
        """

        random_value = self.random_generator.random()

        perpendicular_1, perpendicular_2 = (
            PERPENDICULAR_ACTIONS[intended_action]
        )

        if random_value < 0.8:
            return intended_action

        if random_value < 0.9:
            return perpendicular_1

        return perpendicular_2

    def _candidate_position(
        self,
        position: tuple[int, int],
        action: int,
    ) -> tuple[int, int]:
        """Calculate the candidate position."""

        row, column = position
        row_change, column_change = ACTION_DELTAS[action]

        return (
            row + row_change,
            column + column_change,
        )

    def _inside_map(
        self,
        position: tuple[int, int],
    ) -> bool:
        """Check whether a position is inside the map."""

        row, column = position

        return (
            0 <= row < self.size
            and 0 <= column < self.size
        )

    def _cell(
        self,
        position: tuple[int, int],
    ) -> str:
        """Return the symbol of a map cell."""

        row, column = position
        return self.grid[row][column]

    def _potential(self, state: State) -> float:
        """
        Potential used for potential-based reward shaping.

        Before collecting the key, distance to the key is used.
        After collecting it, distance to the goal is used.
        """

        row, column, has_key, _ = state

        if has_key:
            target_row, target_column = self.goal_position
        else:
            target_row, target_column = self.key_position

        distance = (
            abs(row - target_row)
            + abs(column - target_column)
        )

        maximum_distance = 2 * (self.size - 1)

        return float(maximum_distance - distance)

    def _simulate_transition(
        self,
        state: State,
        actual_action: int,
    ) -> tuple[State, float, bool, list[str]]:
        """
        Simulate one deterministic transition.

        This function does not modify the real environment.
        """

        if self.is_terminal_state(state):
            return state, 0.0, True, ["terminal_self_loop"]

        row, column, has_key_value, phase = state

        current_position = (row, column)
        has_key = bool(has_key_value)

        next_phase = (phase + 1) % self.gate_period

        candidate = self._candidate_position(
            current_position,
            actual_action,
        )

        new_position = current_position
        new_has_key = has_key
        terminated = False

        # Every action has a small movement cost.
        reward = -1.0
        events: list[str] = []

        if not self._inside_map(candidate):
            reward -= 4.0
            events.append("wall_collision")

        else:
            candidate_cell = self._cell(candidate)

            if candidate_cell == self.wall_symbol:
                reward -= 4.0
                events.append("wall_collision")

            elif (
                candidate_cell == self.door_symbol
                and not has_key
            ):
                reward -= 7.0
                events.append("closed_door_attempt")

            elif (
                candidate_cell == self.gate_symbol
                and not self.is_gate_open(phase)
            ):
                reward -= 3.0
                events.append("periodic_gate_closed")

            else:
                new_position = candidate

                if candidate_cell == self.penalty_symbol:
                    reward -= 9.0
                    events.append("penalty_cell")

                elif (
                    candidate_cell == self.key_symbol
                    and not has_key
                ):
                    new_has_key = True
                    reward += 21.0
                    events.append("key_collected")

                elif candidate_cell == self.door_symbol:
                    events.append("door_passed")

                elif candidate_cell == self.gate_symbol:
                    events.append("periodic_gate_passed")

                elif (
                    candidate_cell == self.goal_symbol
                    and new_has_key
                ):
                    reward += 101.0
                    terminated = True
                    events.append("goal_reached")

                else:
                    events.append("normal_move")

        next_state: State = (
            new_position[0],
            new_position[1],
            int(new_has_key),
            next_phase,
        )

        if self.reward_mode == "shaped":
            shaping_reward = 0.20 * (
                self.gamma * self._potential(next_state)
                - self._potential(state)
            )

            reward += shaping_reward
            events.append("shaping_reward")

        return next_state, reward, terminated, events

    def step(
        self,
        action: int,
    ) -> tuple[
        State,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """
        Execute one stochastic environment step.

        Returns:
            next_state
            reward
            terminated
            truncated
            info
        """

        if action not in ACTION_NAMES:
            raise ValueError(
                "Action must be 0, 1, 2, or 3."
            )

        if self.done:
            raise RuntimeError(
                "Episode is finished. Call reset() first."
            )

        current_state = self.get_state()
        current_gate_open = self.is_gate_open()

        actual_action = self._sample_actual_action(action)

        (
            simulated_next_state,
            reward,
            terminated,
            events,
        ) = self._simulate_transition(
            current_state,
            actual_action,
        )

        self.position = (
            simulated_next_state[0],
            simulated_next_state[1],
        )

        self.has_key = bool(
            simulated_next_state[2]
        )

        self.step_count += 1

        truncated = False

        if (
            not terminated
            and self.step_count >= self.max_steps
        ):
            truncated = True
            reward -= 20.0
            events.append("step_limit_reached")

        next_state = self.get_state()

        self.episode_reward += reward
        self.done = terminated or truncated

        step_record = {
            "step": self.step_count,
            "state": current_state,
            "intended_action": ACTION_NAMES[action],
            "actual_action": ACTION_NAMES[actual_action],
            "next_state": next_state,
            "reward": reward,
            "events": events,
            "gate_open_before_action": current_gate_open,
            "gate_open_after_action": self.is_gate_open(),
            "terminated": terminated,
            "truncated": truncated,
        }

        self.event_log.append(step_record)

        info = {
            "intended_action": ACTION_NAMES[action],
            "actual_action": ACTION_NAMES[actual_action],
            "events": events,
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "has_key": self.has_key,
            "gate_open": self.is_gate_open(),
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    def is_terminal_state(self, state: State) -> bool:
        """Check whether a state is a successful terminal state."""

        row, column, has_key, _ = state

        return (
            (row, column) == self.goal_position
            and bool(has_key)
        )

    def transition_outcomes(
        self,
        state: State,
        intended_action: int,
    ) -> list[dict[str, Any]]:
        """
        Return the full stochastic transition model.

        This method will later be used by Value Iteration.
        """

        if intended_action not in ACTION_NAMES:
            raise ValueError(
                "Action must be 0, 1, 2, or 3."
            )

        perpendicular_1, perpendicular_2 = (
            PERPENDICULAR_ACTIONS[intended_action]
        )

        possible_actions = (
            (0.8, intended_action),
            (0.1, perpendicular_1),
            (0.1, perpendicular_2),
        )

        aggregated: dict[
            tuple[State, float, bool],
            float,
        ] = {}

        for probability, actual_action in possible_actions:
            (
                next_state,
                reward,
                terminated,
                _,
            ) = self._simulate_transition(
                state,
                actual_action,
            )

            transition_key = (
                next_state,
                round(reward, 12),
                terminated,
            )

            aggregated[transition_key] = (
                aggregated.get(transition_key, 0.0)
                + probability
            )

        outcomes: list[dict[str, Any]] = []

        for transition_key, probability in aggregated.items():
            next_state, reward, terminated = transition_key

            outcomes.append(
                {
                    "probability": probability,
                    "next_state": next_state,
                    "reward": reward,
                    "terminated": terminated,
                }
            )

        return outcomes

    def get_all_states(self) -> list[State]:
        """Return all non-wall states."""

        states: list[State] = []

        for row in range(self.size):
            for column in range(self.size):
                if self.grid[row][column] == self.wall_symbol:
                    continue

                for has_key in (0, 1):
                    for phase in range(self.gate_period):
                        states.append(
                            (
                                row,
                                column,
                                has_key,
                                phase,
                            )
                        )

        return states

    def render_text(self) -> str:
        """Return a text representation of the maze."""

        rendered_rows: list[str] = []

        for row_index, row in enumerate(self.grid):
            rendered_row: list[str] = []

            for column_index, cell in enumerate(row):
                if (
                    row_index,
                    column_index,
                ) == self.position:
                    rendered_row.append("A")
                else:
                    rendered_row.append(cell)

            rendered_rows.append(
                "".join(rendered_row)
            )

        return "\n".join(rendered_rows)

    def save_event_log(
        self,
        output_path: str | Path,
    ) -> Path:
        """Save episode events as a JSON file."""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.event_log,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return output_path


if __name__ == "__main__":
    environment = MazeEnv(
        transition_seed=8,
        reward_mode="sparse",
    )

    initial_state, initial_info = environment.reset(
        seed=8
    )

    print("Environment loaded successfully.")
    print("Initial state:", initial_state)
    print("Maximum steps:", initial_info["max_steps"])
    print("Gate open:", initial_info["gate_open"])
    print("Number of MDP states:",
          len(environment.get_all_states()))

    print()
    print("Maze with agent:")
    print(environment.render_text())

    print()
    print("Transition outcomes for action RIGHT:")

    outcomes = environment.transition_outcomes(
        initial_state,
        RIGHT,
    )

    probability_sum = 0.0

    for outcome in outcomes:
        probability_sum += outcome["probability"]
        print(outcome)

    print()
    print("Probability sum:", probability_sum)