import sys
import unittest
from pathlib import Path


# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from environments.maze import (
    DOWN,
    RIGHT,
    UP,
    MazeEnv,
)


class TestMazeEnvironment(unittest.TestCase):

    def setUp(self) -> None:
        """Create a fresh environment before every test."""

        self.env = MazeEnv(
            transition_seed=8,
            reward_mode="sparse",
        )

    def test_reset_returns_initial_state(self) -> None:
        state, info = self.env.reset(seed=8)

        self.assertEqual(state, (1, 1, 0, 0))
        self.assertFalse(self.env.has_key)
        self.assertTrue(info["gate_open"])
        self.assertEqual(
            info["max_steps"],
            3 * self.env.passable_cell_count,
        )

    def test_transition_probabilities_sum_to_one(self) -> None:
        state, _ = self.env.reset(seed=8)

        outcomes = self.env.transition_outcomes(
            state,
            RIGHT,
        )

        probability_sum = sum(
            outcome["probability"]
            for outcome in outcomes
        )

        probabilities = sorted(
            outcome["probability"]
            for outcome in outcomes
        )

        self.assertAlmostEqual(
            probability_sum,
            1.0,
            places=10,
        )

        self.assertEqual(
            probabilities,
            [0.1, 0.1, 0.8],
        )

    def test_wall_collision_keeps_agent_in_place(self) -> None:
        self.env.reset(seed=1)

        next_state, reward, terminated, truncated, info = (
            self.env.step(UP)
        )

        self.assertEqual(
            next_state[:2],
            self.env.start_position,
        )

        self.assertIn(
            "wall_collision",
            info["events"],
        )

        self.assertEqual(reward, -5.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_key_is_collected(self) -> None:
        self.env.reset(seed=1)

        key_row, key_column = self.env.key_position

        self.env.position = (
            key_row,
            key_column - 1,
        )

        self.env.has_key = False
        self.env.step_count = 0
        self.env.done = False
        self.env.random_generator.seed(1)

        next_state, reward, terminated, truncated, info = (
            self.env.step(RIGHT)
        )

        self.assertEqual(
            next_state[:2],
            self.env.key_position,
        )

        self.assertEqual(next_state[2], 1)
        self.assertTrue(self.env.has_key)

        self.assertIn(
            "key_collected",
            info["events"],
        )

        self.assertGreater(reward, 0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_closed_door_blocks_agent_without_key(self) -> None:
        self.env.reset(seed=1)

        door_row, door_column = self.env.door_position

        self.env.position = (
            door_row - 1,
            door_column,
        )

        self.env.has_key = False
        self.env.step_count = 0
        self.env.done = False
        self.env.random_generator.seed(1)

        previous_position = self.env.position

        next_state, reward, terminated, truncated, info = (
            self.env.step(DOWN)
        )

        self.assertEqual(
            next_state[:2],
            previous_position,
        )

        self.assertIn(
            "closed_door_attempt",
            info["events"],
        )

        self.assertEqual(reward, -8.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_periodic_gate_blocks_agent_when_closed(self) -> None:
        self.env.reset(seed=1)

        gate_row, gate_column = self.env.gate_position

        self.env.position = (
            gate_row,
            gate_column - 1,
        )

        self.env.has_key = False

        # Phase 1 is closed because open_phase is 0.
        self.env.step_count = 1
        self.env.done = False
        self.env.random_generator.seed(1)

        previous_position = self.env.position

        next_state, reward, terminated, truncated, info = (
            self.env.step(RIGHT)
        )

        self.assertEqual(
            next_state[:2],
            previous_position,
        )

        self.assertIn(
            "periodic_gate_closed",
            info["events"],
        )

        self.assertEqual(reward, -4.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()