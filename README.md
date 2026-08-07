# RL Final Project - Dynamic Maze

**Student ID:** 40403384  
**Base Seed:** 8  
**Maze Size:** 15 x 15  
**Course:** Reinforcement Learning

This project implements and evaluates reinforcement-learning agents in a stochastic dynamic maze.

The project includes:

- Value Iteration
- Q-Learning
- SARSA(lambda) with replacing eligibility traces
- Transfer Learning with Q-Learning
- Reward shaping experiments
- A Pygame graphical interface
- Reproducible experiments
- Saved models, figures, configurations, and raw results
- Unit tests

---

## 1. Environment

The environment is a 15 x 15 dynamic maze generated using the student-specific seed.

```text
Student ID = 40403384
Base seed = 8
Maze size = 15 + (8 % 4) = 15
```

The maze contains walls, normal cells, penalty cells, a start position, a key, a locked door, a goal, and a periodic gate.

The agent must start from the initial position, collect the key, pass through the locked door, and reach the goal.

### State representation

The Markov state is represented as:

```text
(row, column, has_key, gate_phase)
```

The environment contains 141 passable cells, and the complete state representation contains 564 states.

### Actions

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

### Stochastic transitions

For every selected action:

- Intended action: probability `0.8`
- First perpendicular action: probability `0.1`
- Second perpendicular action: probability `0.1`

A collision with a wall leaves the agent in the same position and produces a penalty.

### Reward modes

Two reward definitions are supported:

- `sparse`
- `shaped`

Potential-based reward shaping uses:

```text
F(s,s') = 0.20 * [gamma * Phi(s') - Phi(s)]
```

Before collecting the key, the potential is based on Manhattan distance to the key. After collecting the key, it is based on Manhattan distance to the goal.

---

## 2. Algorithms

### Value Iteration

Value Iteration uses the complete transition model and implements Bellman updates from scratch.

The following discount factors are compared:

```text
gamma = 0.80
gamma = 0.90
gamma = 0.95
```

The saved results include the number of iterations, runtime, final delta, state-value table, greedy policy, and agreement with the reference policy.

### Q-Learning

Q-Learning is implemented as an off-policy, model-free algorithm with an epsilon-greedy behavior policy.

Two epsilon schedules are compared:

- Linear decay
- Exponential decay

The linear schedule produced better evaluation reward and fewer steps in the main schedule-comparison experiment.

Training CSV files also record episode-level environment events, including normal moves, wall collisions, penalty visits, closed-door attempts, successful door passes, periodic-gate events, key collection, goal reaching, and step-limit termination.

### SARSA(lambda)

SARSA(lambda) is implemented as an on-policy algorithm using replacing eligibility traces.

The following lambda values are evaluated:

```text
lambda = 0.0
lambda = 0.3
lambda = 0.7
lambda = 0.9
```

In the current experiments, `lambda = 0.3` provides the strongest overall balance between success rate, reward, path quality, and stability.

For a sample episode, TD error and eligibility-trace information is saved for inspection.

---

## 3. Transfer Learning

Transfer learning is performed using Q-Learning. The learned source Q-table is transferred to two destination environments.

### Similar target environment

- Start, key, and goal remain fixed.
- Two of fourteen interior walls are moved.
- The resulting moved-wall percentage is `14.29%`.
- This is the closest feasible discrete value to the requested 15-20% range.

### Different target environment

- `39.29%` of interior walls are moved.
- The key position is changed.
- Three new penalty cells are added.
- The generated map is validated using BFS.

### Transfer scenarios

The evaluated scenarios are:

1. Training from scratch
2. Full Q-table transfer
3. Scaled transfer with `beta = 0.25`
4. Scaled transfer with `beta = 0.50`
5. Scaled transfer with `beta = 0.75`
6. Selective transfer

Selective transfer copies only states whose local neighborhoods are unchanged.

A state-level negative-transfer example is also recorded. At state `(3, 11, 0, 0)`, the transferred greedy action was initially `UP`, while the changed target structure favored `RIGHT`. After further target training, the greedy action was corrected to `RIGHT`.

---

## 4. Project Structure

```text
RL_FinalProject_40403384/
|-- agents/
|   |-- value_iteration.py
|   |-- q_learning.py
|   `-- sarsa_lambda.py
|
|-- analysis/
|   |-- generate_figures.py
|   |-- generate_q_learning_curve.py
|   |-- generate_sarsa_lambda_curves.py
|   |-- generate_value_iteration_convergence.py
|   |-- generate_transfer_q_change.py
|   |-- generate_reward_comparison_figures.py
|   `-- measure_algorithm_memory.py
|
|-- configs/
|   `-- experiment_config.json
|
|-- environments/
|   |-- generator.py
|   |-- maze.py
|   `-- maps/
|       |-- source_map.json
|       |-- target_similar.json
|       `-- target_different.json
|
|-- experiments/
|   |-- run_value_iteration.py
|   |-- evaluate_value_iteration.py
|   |-- run_q_learning.py
|   |-- run_sarsa_lambda.py
|   |-- run_reward_comparison.py
|   `-- run_transfer_learning.py
|
|-- gui/
|   `-- maze_gui.py
|
|-- transfer/
|   `-- transfer_learning.py
|
|-- tests/
|   |-- test_maze.py
|   |-- test_value_iteration.py
|   |-- test_q_learning.py
|   |-- test_sarsa_lambda.py
|   `-- test_transfer_learning.py
|
|-- results/
|   |-- figures/
|   |-- models/
|   `-- raw_data/
|
|-- report/
|   `-- RL_Final_Report_40403384.docx
|
|-- main.py
|-- requirements.txt
|-- README.md
`-- report.pdf
```

---

## 5. Requirements

The project was tested with:

```text
Python 3.12.7
Windows 10
```

Required Python packages are listed in `requirements.txt`:

```text
pygame==2.6.1
matplotlib==3.9.2
python-docx==1.2.0
```

The reinforcement-learning algorithms are implemented from scratch and do not use external reinforcement-learning libraries.

`pygame` is used for the graphical interface, `matplotlib` is used for experimental figures, and `python-docx` is included for document-related project utilities.

---

## 6. Installation

Open a terminal in the project root directory and run:

```bash
python -m pip install -r requirements.txt
```

Verify Pygame with:

```bash
python -c "import pygame; print(pygame.version.ver)"
```

---

## 7. Running the Project

Show the project configuration:

```bash
python main.py
```

Generate the source maze:

```bash
python environments/generator.py
```

Run Value Iteration experiments:

```bash
python experiments/run_value_iteration.py
python experiments/evaluate_value_iteration.py
```

Run Q-Learning experiments:

```bash
python experiments/run_q_learning.py
```

Run SARSA(lambda) experiments:

```bash
python experiments/run_sarsa_lambda.py
```

Run sparse-vs-shaped reward comparison experiments:

```bash
python experiments/run_reward_comparison.py
```

Generate transfer-learning maps and run transfer experiments:

```bash
python transfer/transfer_learning.py
python experiments/run_transfer_learning.py
```

Run the Pygame interface:

```bash
python gui/maze_gui.py
```

---

## 8. Graphical Interface

The Pygame interface supports source, similar, and different maps; multiple learned policies; start/pause; single-step execution; reset; model and map selection; animation-speed control; agent-trail visualization; current state; reward; key and gate status; intended and actual actions; and final success/failure status.

### Keyboard controls

| Key | Action |
|---|---|
| `Space` | Start or pause |
| `N` | Single step |
| `R` | Reset |
| `M` | Next model |
| `E` | Next map |
| `+` | Faster |
| `-` | Slower |
| `Esc` | Exit |

---

## 9. Running Tests

Run all unit tests from the project root:

```bash
python -m unittest discover -s tests -v
```

Current result:

```text
Ran 29 tests
OK
```

The tests cover environment behavior, transition probabilities, Value Iteration convergence, Q-Learning updates and epsilon schedules, SARSA eligibility traces, transfer-map generation and validation, and transfer strategies.

---

## 10. Reproducing the Results

All main experiment settings and hyperparameters are documented in:

```text
configs/experiment_config.json
```

To reproduce the main experimental results:

```bash
python -m pip install -r requirements.txt
python environments/generator.py
python experiments/run_value_iteration.py
python experiments/evaluate_value_iteration.py
python experiments/run_q_learning.py
python experiments/run_sarsa_lambda.py
python experiments/run_reward_comparison.py
python transfer/transfer_learning.py
python experiments/run_transfer_learning.py
```

To regenerate figures and additional analysis outputs:

```bash
python analysis/generate_figures.py
python analysis/generate_value_iteration_convergence.py
python analysis/generate_q_learning_curve.py
python analysis/generate_sarsa_lambda_curves.py
python analysis/generate_reward_comparison_figures.py
python analysis/generate_transfer_q_change.py
python analysis/measure_algorithm_memory.py
```

To run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

Generated outputs are stored under:

```text
results/raw_data/
results/models/
results/figures/
```

All project paths are calculated relative to the repository root and do not depend on a specific local username or absolute directory.

---

## 11. Reproducibility and Saved Data

The project uses deterministic seeds for map generation, environment transitions, agent exploration, evaluation episodes, and transfer-map generation.

The base seed is:

```text
8
```

Raw numerical results are stored in `results/raw_data/`, learned models and policies in `results/models/`, and generated figures in `results/figures/`.

Q-Learning training-state visitation counts are saved separately and are used to generate the training visitation heatmap.

The reward-shaping behavior analysis checks for undesirable effects such as long loops, reward farming, excessive hazard avoidance, and timeouts in the final training episodes.

Peak Python-tracked memory usage is measured with `tracemalloc` in a separate sparse-reward comparison run. These values represent peak Python-managed allocations, not total operating-system process RAM.

The memory comparison output is stored in:

```text
results/raw_data/algorithm_memory_comparison.csv
```

---

## 12. Main Experimental Findings

- Value Iteration with `gamma = 0.95` achieved a `100%` evaluation success rate, an average reward of `65.74`, and an average path length of `47.60` steps.
- For Q-Learning, the linear epsilon schedule achieved a `100%` evaluation success rate with an average reward of `9.70` and an average path length of `78.67` steps.
- The exponential epsilon schedule also achieved a `100%` evaluation success rate, but with a lower average reward of `-32.54` and a longer average path length of `83.65` steps.
- Among the tested SARSA(lambda) settings, `lambda = 0.3` achieved a `100%` evaluation success rate with an average reward of `56.02` and an average path length of `49.84` steps.
- Reward shaping showed no evidence of persistent reward farming or long-loop behavior in the final training episodes. In the behavior-check output, `high_reward_long_count = 0`.
- Training-state visitation counts are collected directly during Q-Learning training and are used for the visitation heatmap.
- Transfer learning substantially improves performance on the similar target environment.
- Negative transfer can occur when source knowledge is transferred to a locally changed target region.
- At state `(3, 11, 0, 0)`, the transferred greedy action was initially `UP`, while the target environment favored `RIGHT`; after further target training, the greedy action was corrected to `RIGHT`.
- Selective transfer reduces the risk of copying inappropriate Q-values from structurally changed local regions.
- Peak Python-tracked memory measured with `tracemalloc` was approximately `0.1348 MB` for Value Iteration, `0.7397 MB` for Q-Learning, and `0.7132 MB` for SARSA(lambda=0.3).
- All 29 unit tests pass after the final implementation changes.

---

## 13. Use of AI Assistance

An AI assistant was used as a development aid for code-structure suggestions, debugging support, test-case suggestions, and documentation drafting. All suggestions were reviewed, modified, executed, and validated by the student.

| Use | AI suggestion | Student change | Reason |
|---|---|---|---|
| Similar-map generation | Move approximately 18% of interior walls using ceiling | Replaced ceiling with rounding, producing two moved walls (`14.29%`) | Three of fourteen walls would be `21.43%`, outside the intended range |
| Different-map generation | Move the key to a random valid position | Added a minimum key-to-goal distance and BFS validation before accepting the map | The initial random suggestion placed the key only one step from the goal and made the task too easy |

The final implementation was validated with reproducible experiments and 29 unit tests.

---

## 14. Repository

Public GitHub repository:

https://github.com/mina2collab/RL_FinalProject_40403384

---

## Author

Student ID: **40403384**
