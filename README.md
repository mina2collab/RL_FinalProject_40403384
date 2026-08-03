# RL Final Project — Dynamic Maze

**Student ID:** 40403384  
**Base Seed:** 8  
**Maze Size:** 15 × 15  
**Course:** Reinforcement Learning  

This project implements and evaluates reinforcement-learning agents in a stochastic dynamic maze.

The project includes:

- Value Iteration
- Q-Learning
- SARSA(λ) with replacing eligibility traces
- Transfer Learning with Q-Learning
- A Pygame graphical interface
- Reproducible experiments
- Saved models and raw results
- Unit tests

---

## 1. Environment

The environment is a 15 × 15 dynamic maze generated using the student-specific seed.

```text
Student ID = 40403384
Base seed = 8
Maze size = 15 + (8 % 4) = 15
```

The maze contains:

- Walls
- Normal cells
- Penalty cells
- Start position
- Key
- Locked door
- Goal
- Periodic gate

The agent must:

1. Start from the initial position.
2. Collect the key.
3. Pass through the locked door.
4. Reach the goal.

### State representation

The Markov state is represented as:

```text
(row, column, has_key, gate_phase)
```

Where:

- `row` and `column` are the agent position.
- `has_key` shows whether the key has been collected.
- `gate_phase` represents the current phase of the periodic gate.

Including the gate phase and key status preserves the Markov property.

### Actions

The agent has four actions:

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

The shaped reward provides intermediate feedback for useful behavior while preserving the main task objective.

---

## 2. Algorithms

### Value Iteration

Value Iteration uses the complete transition model and implements Bellman updates from scratch.

The following discount factors are compared:

```text
γ = 0.80
γ = 0.90
γ = 0.95
```

The saved results include:

- Number of iterations
- Runtime
- Final delta
- State-value table
- Greedy policy
- Agreement with the reference policy

### Q-Learning

Q-Learning is implemented as an off-policy, model-free algorithm with an ε-greedy behavior policy.

Two epsilon schedules are compared:

- Linear decay
- Exponential decay

The linear schedule produced better evaluation reward and fewer steps in the current experiments.

### SARSA(λ)

SARSA(λ) is implemented as an on-policy algorithm using replacing eligibility traces.

The following λ values are evaluated:

```text
λ = 0.0
λ = 0.3
λ = 0.7
λ = 0.9
```

In the current results, `λ = 0.3` provides the best balance between success rate, reward, path length, and training time.

For a sample episode, consecutive changes in TD error `δ` and eligibility traces `E` are saved as JSON.

---

## 3. Transfer Learning

Transfer learning is performed using Q-Learning.

The learned Q-table from the source maze is transferred to two destination environments.

### Similar target environment

- Start, key, and goal remain fixed.
- Two of fourteen interior walls are moved.
- The resulting moved-wall percentage is `14.29%`.
- This is the closest feasible discrete value to the requested 15–20% range.

### Different target environment

- `39.29%` of interior walls are moved.
- The key position is changed.
- Three new penalty cells are added.
- The generated map is validated using BFS.

### Transfer scenarios

The following scenarios are evaluated:

1. Training from scratch
2. Full Q-table transfer
3. Scaled transfer with `β = 0.25`
4. Scaled transfer with `β = 0.50`
5. Scaled transfer with `β = 0.75`
6. Selective transfer

Selective transfer copies only states whose local neighborhoods are unchanged.

Current observations:

- On the similar map, scaled transfer with `β = 0.75` gives the strongest final result.
- On the different map, transferred policies reach strong final performance.
- Some full and scaled transfers initially perform poorly, demonstrating negative transfer.
- Selective transfer improves initial safety by skipping changed local regions.

---

## 4. Project Structure

```text
RL_FinalProject_40403384/
├── agents/
│   ├── value_iteration.py
│   ├── q_learning.py
│   └── sarsa_lambda.py
│
├── environments/
│   ├── generator.py
│   ├── maze.py
│   └── maps/
│       ├── source_map.json
│       ├── target_similar.json
│       └── target_different.json
│
├── experiments/
│   ├── run_value_iteration.py
│   ├── run_q_learning.py
│   ├── run_sarsa_lambda.py
│   └── run_transfer_learning.py
│
├── gui/
│   └── maze_gui.py
│
├── transfer/
│   └── transfer_learning.py
│
├── tests/
│   ├── test_maze.py
│   ├── test_value_iteration.py
│   ├── test_q_learning.py
│   ├── test_sarsa_lambda.py
│   └── test_transfer_learning.py
│
├── results/
│   ├── models/
│   └── raw_data/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 5. Requirements

The project was tested with:

```text
Python 3.12.7
Pygame 2.6.1
Windows 10
```

The reinforcement-learning algorithms use only the Python standard library and were implemented without external RL libraries.

---

## 6. Installation

Open a terminal in the project root directory.

Install the required package:

```bash
python -m pip install -r requirements.txt
```

Verify the Pygame installation:

```bash
python -c "import pygame; print(pygame.version.ver)"
```

---

## 7. Running the Project

### Show the project configuration

```bash
python main.py
```

Expected configuration:

```text
Student ID: 40403384
Base seed: 8
Maze size: 15
```

### Generate the source maze

```bash
python environments/generator.py
```

The generated map is saved at:

```text
environments/maps/source_map.json
```

### Run Value Iteration

```bash
python agents/value_iteration.py
```

Run the gamma comparison:

```bash
python experiments/run_value_iteration.py
```

### Run Q-Learning

```bash
python agents/q_learning.py
```

Compare linear and exponential epsilon schedules:

```bash
python experiments/run_q_learning.py
```

### Run SARSA(λ)

```bash
python agents/sarsa_lambda.py
```

Compare all required λ values:

```bash
python experiments/run_sarsa_lambda.py
```

### Generate transfer-learning maps

```bash
python transfer/transfer_learning.py
```

### Run transfer-learning experiments

```bash
python experiments/run_transfer_learning.py
```

This experiment runs all transfer scenarios on both destination maps.

---

## 8. Graphical Interface

Run the Pygame interface with:

```bash
python gui/maze_gui.py
```

The interface supports:

- Source, similar, and different maps
- Multiple learned policies
- Start and pause
- Single-step execution
- Episode reset
- Model selection
- Map selection
- Animation speed control
- Agent trail visualization
- Current state
- Step count
- Episode reward
- Key status
- Gate status
- Intended and actual actions
- Final success or failure status

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

Current test result:

```text
Ran 29 tests
OK
```

The tests cover:

- Environment reset and state representation
- Stochastic transition probabilities
- Wall collisions
- Key collection
- Locked door behavior
- Periodic gate behavior
- Value Iteration convergence
- Q-Learning updates and epsilon schedules
- SARSA eligibility traces
- Transfer-map generation
- BFS map validation
- Full, scaled, selective, and scratch transfer

---

## 10. Reproducing the Results

To reproduce all main results, run these commands in order:

```bash
python -m pip install -r requirements.txt
python environments/generator.py
python experiments/run_value_iteration.py
python experiments/run_q_learning.py
python experiments/run_sarsa_lambda.py
python transfer/transfer_learning.py
python experiments/run_transfer_learning.py
python -m unittest discover -s tests -v
```

Generated outputs are stored under:

```text
results/raw_data/
results/models/
```

Raw experiment data are stored as CSV and JSON files so that the results can be inspected and reproduced.

All paths are calculated relative to the project directory and do not depend on a specific computer username or absolute local path.

---

## 11. Reproducibility

The project uses deterministic seeds for:

- Map generation
- Environment transitions
- Agent exploration
- Evaluation episodes
- Transfer-map generation

The base seed is derived from the penultimate digit of the student ID:

```text
Base seed = 8
```

Every algorithm uses the saved source map to ensure that comparisons are performed under the same environment.

---

## 12. Main Experimental Findings

- Value Iteration with `γ = 0.95` is used as the main reference policy.
- Linear epsilon decay performs better than exponential decay for Q-Learning.
- SARSA with `λ = 0.3` provides the best overall balance in the tested settings.
- Transfer learning substantially improves performance on the similar destination map.
- Full or scaled transfer can produce negative transfer when the destination structure changes significantly.
- Selective transfer reduces the risk of copying knowledge from changed local regions.
- Reward shaping accelerates learning, but its effect must be evaluated together with policy quality and path length.

---

## 13. Use of AI Assistance

An AI assistant was used as a development aid for:

- Initial code structure suggestions
- Debugging support
- Test-case suggestions
- Documentation drafting

All generated suggestions were reviewed, modified, executed, and validated by the student.

Examples of suggestions that required correction:

| Initial suggestion | Problem | Student correction |
|---|---|---|
| Move approximately 18% of similar-map walls using ceiling | Three of fourteen walls produced `21.43%`, outside the preferred range | Replaced ceiling with rounding, producing the closest feasible value of `14.29%` |
| Randomly move the key in the different map | The generated key was only one step from the goal, making the task too easy | Added a minimum key-to-goal distance constraint and regenerated the map |

The final implementation was validated using reproducible experiments and 29 unit tests.

---

## 14. Repository

Public GitHub repository:

```text
https://github.com/mina2collab/RL_FinalProject_40403384
```

---

## Author

Student ID: **40403384**