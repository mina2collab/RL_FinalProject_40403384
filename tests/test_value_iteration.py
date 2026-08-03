import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.value_iteration import ValueIterationAgent
from environments.maze import ACTION_NAMES, MazeEnv


class TestValueIteration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Run Value Iteration once for all tests."""

        cls.env = MazeEnv(
            transition_seed=8,
            reward_mode="sparse",
            gamma=0.95,
        )

        cls.agent = ValueIterationAgent(
            environment=cls.env,
            gamma=0.95,
            theta=1e-6,
            max_iterations=2_000,
        )

        cls.summary = cls.agent.run()

    def test_algorithm_converges(self) -> None:
        self.assertTrue(self.summary["converged"])

        self.assertLess(
            self.summary["final_delta"],
            self.agent.theta,
        )

    def test_value_table_contains_all_states(self) -> None:
        self.assertEqual(
            len(self.agent.values),
            len(self.env.get_all_states()),
        )

        self.assertEqual(
            len(self.agent.values),
            564,
        )

    def test_policy_contains_valid_actions(self) -> None:
        for state, action in self.agent.policy.items():
            if self.env.is_terminal_state(state):
                self.assertIsNone(action)
            else:
                self.assertIn(action, ACTION_NAMES)

    def test_terminal_state_value_is_zero(self) -> None:
        goal_row, goal_column = self.env.goal_position

        for phase in range(self.env.gate_period):
            terminal_state = (
                goal_row,
                goal_column,
                1,
                phase,
            )

            self.assertEqual(
                self.agent.values[terminal_state],
                0.0,
            )

    def test_initial_state_has_policy(self) -> None:
        initial_state, _ = self.env.reset(seed=8)

        self.assertIn(initial_state, self.agent.policy)

        self.assertIn(
            self.agent.policy[initial_state],
            ACTION_NAMES,
        )


if __name__ == "__main__":
    unittest.main()