import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.q_learning import QLearningAgent
from environments.maze import MazeEnv
from transfer.transfer_learning import (
    DIFFERENT_MAP_PATH,
    SIMILAR_MAP_PATH,
    SOURCE_MAP_PATH,
    create_target_maps,
    initialize_target_q_table,
    load_json,
    load_source_q_table,
    validate_target_map,
)


class TestTransferLearning(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Create deterministic target maps once."""

        create_target_maps()

        cls.source_data = load_json(
            SOURCE_MAP_PATH
        )

        cls.similar_data = load_json(
            SIMILAR_MAP_PATH
        )

        cls.different_data = load_json(
            DIFFERENT_MAP_PATH
        )

        cls.source_environment = MazeEnv(
            map_path=SOURCE_MAP_PATH,
            transition_seed=8,
            reward_mode="shaped",
            gamma=0.95,
        )

        cls.similar_environment = MazeEnv(
            map_path=SIMILAR_MAP_PATH,
            transition_seed=8,
            reward_mode="shaped",
            gamma=0.95,
        )

        cls.different_environment = MazeEnv(
            map_path=DIFFERENT_MAP_PATH,
            transition_seed=8,
            reward_mode="shaped",
            gamma=0.95,
        )

        cls.source_q_table = (
            load_source_q_table()
        )

    @staticmethod
    def create_agent(
        environment: MazeEnv,
    ) -> QLearningAgent:
        """Create a target Q-Learning agent."""

        return QLearningAgent(
            environment=environment,
            alpha=0.10,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=300,
            epsilon_schedule="linear",
            seed=8,
        )

    def test_target_map_files_exist(self) -> None:
        self.assertTrue(
            SIMILAR_MAP_PATH.exists()
        )

        self.assertTrue(
            DIFFERENT_MAP_PATH.exists()
        )

    def test_similar_map_keeps_main_positions_and_is_valid(
        self,
    ) -> None:
        source_positions = (
            self.source_data["positions"]
        )

        similar_positions = (
            self.similar_data["positions"]
        )

        self.assertEqual(
            similar_positions["start"],
            source_positions["start"],
        )

        self.assertEqual(
            similar_positions["key"],
            source_positions["key"],
        )

        self.assertEqual(
            similar_positions["goal"],
            source_positions["goal"],
        )

        symbols = self.similar_data["symbols"]

        validation = validate_target_map(
            grid=[
                list(row)
                for row
                in self.similar_data["grid"]
            ],
            start=tuple(
                similar_positions["start"]
            ),
            key=tuple(
                similar_positions["key"]
            ),
            goal=tuple(
                similar_positions["goal"]
            ),
            wall_symbol=symbols["wall"],
            door_symbol=symbols["door"],
        )

        self.assertIsNotNone(validation)

    def test_different_map_changes_key_and_adds_penalties(
        self,
    ) -> None:
        source_positions = (
            self.source_data["positions"]
        )

        different_positions = (
            self.different_data["positions"]
        )

        self.assertNotEqual(
            different_positions["key"],
            source_positions["key"],
        )

        self.assertEqual(
            len(
                different_positions[
                    "penalties"
                ]
            ),
            len(
                source_positions[
                    "penalties"
                ]
            ) + 3,
        )

        moved_percentage = (
            self.different_data[
                "transfer_metadata"
            ]["moved_wall_percentage"]
        )

        self.assertGreaterEqual(
            moved_percentage,
            35.0,
        )

        symbols = self.different_data[
            "symbols"
        ]

        validation = validate_target_map(
            grid=[
                list(row)
                for row
                in self.different_data["grid"]
            ],
            start=tuple(
                different_positions["start"]
            ),
            key=tuple(
                different_positions["key"]
            ),
            goal=tuple(
                different_positions["goal"]
            ),
            wall_symbol=symbols["wall"],
            door_symbol=symbols["door"],
        )

        self.assertIsNotNone(validation)

    def test_scratch_transfer_keeps_zero_q_values(
        self,
    ) -> None:
        agent = self.create_agent(
            self.similar_environment
        )

        summary = initialize_target_q_table(
            target_agent=agent,
            source_q_table=(
                self.source_q_table
            ),
            mode="scratch",
            source_environment=(
                self.source_environment
            ),
            beta=0.0,
        )

        self.assertEqual(
            summary["transferred_states"],
            0,
        )

        for values in agent.q_table.values():
            self.assertTrue(
                all(value == 0.0 for value in values)
            )

    def test_full_transfer_copies_shared_states(
        self,
    ) -> None:
        agent = self.create_agent(
            self.similar_environment
        )

        shared_states = set(
            agent.q_table
        ).intersection(
            self.source_q_table
        )

        summary = initialize_target_q_table(
            target_agent=agent,
            source_q_table=(
                self.source_q_table
            ),
            mode="full",
            source_environment=(
                self.source_environment
            ),
            beta=1.0,
        )

        self.assertEqual(
            summary["transferred_states"],
            len(shared_states),
        )

        sample_state = next(
            iter(shared_states)
        )

        self.assertEqual(
            agent.q_table[sample_state],
            self.source_q_table[
                sample_state
            ],
        )

    def test_scaled_transfer_multiplies_q_values(
        self,
    ) -> None:
        agent = self.create_agent(
            self.similar_environment
        )

        beta = 0.50

        shared_states = set(
            agent.q_table
        ).intersection(
            self.source_q_table
        )

        sample_state = next(
            iter(shared_states)
        )

        initialize_target_q_table(
            target_agent=agent,
            source_q_table=(
                self.source_q_table
            ),
            mode="scaled",
            source_environment=(
                self.source_environment
            ),
            beta=beta,
        )

        expected_values = [
            beta * value
            for value
            in self.source_q_table[
                sample_state
            ]
        ]

        for actual, expected in zip(
            agent.q_table[sample_state],
            expected_values,
        ):
            self.assertAlmostEqual(
                actual,
                expected,
            )

    def test_selective_transfer_skips_changed_regions(
        self,
    ) -> None:
        agent = self.create_agent(
            self.different_environment
        )

        shared_state_count = len(
            set(agent.q_table).intersection(
                self.source_q_table
            )
        )

        summary = initialize_target_q_table(
            target_agent=agent,
            source_q_table=(
                self.source_q_table
            ),
            mode="selective",
            source_environment=(
                self.source_environment
            ),
            beta=1.0,
        )

        self.assertGreater(
            summary["transferred_states"],
            0,
        )

        self.assertLess(
            summary["transferred_states"],
            shared_state_count,
        )

        self.assertGreater(
            summary[
                "skipped_changed_neighborhoods"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()