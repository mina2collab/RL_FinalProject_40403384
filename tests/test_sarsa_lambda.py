import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.sarsa_lambda import SARSALambdaAgent
from environments.maze import MazeEnv


class TestSARSALambda(unittest.TestCase):

    def setUp(self) -> None:
        """Create a fresh environment and SARSA agent."""

        self.env = MazeEnv(
            transition_seed=8,
            reward_mode="sparse",
            gamma=0.95,
        )

        self.agent = SARSALambdaAgent(
            environment=self.env,
            alpha=0.10,
            gamma=0.95,
            lambda_value=0.70,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=100,
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

    def test_greedy_action_selects_best_q_value(self) -> None:
        state, _ = self.env.reset(seed=8)

        self.agent.q_table[state] = [
            1.0,
            7.0,
            3.0,
            2.0,
        ]

        selected_action = self.agent.select_action(
            state=state,
            epsilon=0.0,
        )

        self.assertEqual(selected_action, 1)

    def test_training_records_episode_and_traces(self) -> None:
        history = self.agent.train(
            episodes=1,
        )

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["episode"], 1)
        self.assertEqual(history[0]["lambda"], 0.70)

        self.assertGreater(
            len(self.agent.trace_example),
            0,
        )

        first_step = self.agent.trace_example[0]

        self.assertIn(
            "td_error_delta",
            first_step,
        )

        self.assertGreater(
            len(first_step["traces"]),
            0,
        )

    def test_trace_decay_uses_gamma_times_lambda(self) -> None:
        self.agent.train(
            episodes=1,
        )

        first_step = self.agent.trace_example[0]
        first_trace = first_step["traces"][0]

        expected_decay = (
            self.agent.gamma
            * self.agent.lambda_value
        )

        self.assertAlmostEqual(
            first_trace["eligibility_before_decay"],
            1.0,
        )

        self.assertAlmostEqual(
            first_trace["eligibility_after_decay"],
            expected_decay,
        )

    def test_lambda_zero_removes_trace_after_update(self) -> None:
        zero_lambda_agent = SARSALambdaAgent(
            environment=self.env,
            alpha=0.10,
            gamma=0.95,
            lambda_value=0.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_episodes=100,
            seed=8,
        )

        zero_lambda_agent.train(
            episodes=1,
        )

        first_step = zero_lambda_agent.trace_example[0]
        first_trace = first_step["traces"][0]

        self.assertEqual(
            first_trace["eligibility_after_decay"],
            0.0,
        )

        self.assertEqual(
            first_step["active_trace_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()