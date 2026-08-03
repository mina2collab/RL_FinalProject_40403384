import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.q_learning import QLearningAgent
from environments.maze import RIGHT, MazeEnv


class TestQLearning(unittest.TestCase):

    def setUp(self) -> None:
        """Create a fresh environment and agent."""

        self.env = MazeEnv(
            transition_seed=8,
            reward_mode="sparse",
            gamma=0.95,
        )

        self.agent = QLearningAgent(
            environment=self.env,
            alpha=0.10,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=100,
            epsilon_schedule="linear",
            seed=8,
        )

    def test_q_table_contains_all_states(self) -> None:
        self.assertEqual(
            len(self.agent.q_table),
            len(self.env.get_all_states()),
        )

        self.assertEqual(
            len(self.agent.q_table),
            564,
        )

        for values in self.agent.q_table.values():
            self.assertEqual(len(values), 4)

    def test_linear_epsilon_schedule(self) -> None:
        self.assertAlmostEqual(
            self.agent.epsilon_for_episode(0),
            1.0,
        )

        self.assertAlmostEqual(
            self.agent.epsilon_for_episode(50),
            0.525,
        )

        self.assertAlmostEqual(
            self.agent.epsilon_for_episode(100),
            0.05,
        )

        self.assertAlmostEqual(
            self.agent.epsilon_for_episode(200),
            0.05,
        )

    def test_exponential_epsilon_schedule(self) -> None:
        exponential_agent = QLearningAgent(
            environment=self.env,
            alpha=0.10,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=100,
            epsilon_schedule="exponential",
            seed=8,
        )

        self.assertAlmostEqual(
            exponential_agent.epsilon_for_episode(0),
            1.0,
        )

        self.assertAlmostEqual(
            exponential_agent.epsilon_for_episode(100),
            0.05,
            places=10,
        )

        self.assertAlmostEqual(
            exponential_agent.epsilon_for_episode(200),
            0.05,
            places=10,
        )

    def test_greedy_action_selects_best_q_value(self) -> None:
        state, _ = self.env.reset(seed=8)

        self.agent.q_table[state] = [
            1.0,
            2.0,
            5.0,
            3.0,
        ]

        selected_action = self.agent.select_action(
            state=state,
            epsilon=0.0,
        )

        self.assertEqual(selected_action, 2)

    def test_q_learning_update_formula(self) -> None:
        state, _ = self.env.reset(seed=8)

        next_state = (
            state[0],
            state[1] + 1,
            state[2],
            1,
        )

        self.agent.q_table[state][RIGHT] = 2.0

        self.agent.q_table[next_state] = [
            1.0,
            2.0,
            3.0,
            4.0,
        ]

        update = self.agent.update_q_value(
            state=state,
            action=RIGHT,
            reward=-1.0,
            next_state=next_state,
            terminated=False,
            truncated=False,
        )

        expected_target = (
            -1.0
            + 0.95 * 4.0
        )

        expected_new_q = (
            2.0
            + 0.10 * (expected_target - 2.0)
        )

        self.assertAlmostEqual(
            update["target"],
            expected_target,
        )

        self.assertAlmostEqual(
            update["new_q"],
            expected_new_q,
        )

        self.assertAlmostEqual(
            self.agent.q_table[state][RIGHT],
            expected_new_q,
        )


if __name__ == "__main__":
    unittest.main()