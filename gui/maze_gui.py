from __future__ import annotations

import json
import sys
from pathlib import Path

import pygame


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.maze import MazeEnv


State = tuple[int, int, int, int]


MAPS = [
    {
        "name": "Source Map",
        "path": (
            ROOT
            / "environments"
            / "maps"
            / "source_map.json"
        ),
        "models": [
            (
                "Q-Learning",
                ROOT
                / "results"
                / "models"
                / "q_learning_linear.json",
                "shaped",
            ),
            (
                "SARSA lambda=0.3",
                ROOT
                / "results"
                / "models"
                / "sarsa_lambda_0_3.json",
                "shaped",
            ),
            (
                "Value Iteration",
                ROOT
                / "results"
                / "models"
                / "value_iteration_gamma_0_95.json",
                "sparse",
            ),
        ],
    },
    {
        "name": "Similar Target",
        "path": (
            ROOT
            / "environments"
            / "maps"
            / "target_similar.json"
        ),
        "models": [
            (
                "Scaled beta=0.75",
                ROOT
                / "results"
                / "models"
                / "transfer"
                / "similar_scaled_0_75.json",
                "shaped",
            ),
            (
                "Selective Transfer",
                ROOT
                / "results"
                / "models"
                / "transfer"
                / "similar_selective.json",
                "shaped",
            ),
            (
                "Scratch",
                ROOT
                / "results"
                / "models"
                / "transfer"
                / "similar_scratch.json",
                "shaped",
            ),
        ],
    },
    {
        "name": "Different Target",
        "path": (
            ROOT
            / "environments"
            / "maps"
            / "target_different.json"
        ),
        "models": [
            (
                "Selective Transfer",
                ROOT
                / "results"
                / "models"
                / "transfer"
                / "different_selective.json",
                "shaped",
            ),
            (
                "Scaled beta=0.75",
                ROOT
                / "results"
                / "models"
                / "transfer"
                / "different_scaled_0_75.json",
                "shaped",
            ),
            (
                "Scratch",
                ROOT
                / "results"
                / "models"
                / "transfer"
                / "different_scratch.json",
                "shaped",
            ),
        ],
    },
]


COLORS = {
    "background": (235, 238, 242),
    "panel": (247, 248, 250),
    "wall": (40, 45, 55),
    "empty": (250, 250, 250),
    "start": (83, 180, 95),
    "key": (255, 202, 40),
    "door_closed": (133, 87, 35),
    "door_open": (59, 168, 150),
    "goal": (45, 155, 75),
    "penalty": (220, 75, 75),
    "gate_closed": (123, 88, 196),
    "gate_open": (75, 165, 230),
    "agent": (25, 95, 190),
    "trail": (175, 205, 245),
    "grid": (205, 210, 218),
    "text": (30, 35, 45),
    "muted": (90, 98, 110),
    "button": (58, 68, 82),
    "button_hover": (75, 88, 105),
    "white": (255, 255, 255),
}


def load_policy(
    path: Path,
) -> dict[State, int | None]:
    """
    Load a policy from Q-Learning, SARSA,
    transfer learning, or Value Iteration.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    policy: dict[State, int | None] = {}

    if "q_table" in data:
        for record in data["q_table"]:
            state: State = tuple(
                record["state"]
            )

            action = record.get(
                "best_action"
            )

            if (
                action is None
                and any(record["q_values"])
            ):
                action = max(
                    range(
                        len(record["q_values"])
                    ),
                    key=lambda index: (
                        record["q_values"][index]
                    ),
                )

            policy[state] = action

        return policy

    if "states" in data:
        for record in data["states"]:
            state: State = tuple(
                record["state"]
            )

            policy[state] = record[
                "best_action"
            ]

        return policy

    raise ValueError(
        "Unknown model format."
    )


class MazeGUI:
    """Pygame visualization of learned policies."""

    def __init__(self) -> None:
        pygame.init()

        pygame.display.set_caption(
            "Dynamic Maze - RL Project"
        )

        self.cell = 42
        self.margin = 20

        self.panel_x = (
            self.margin * 2
            + 15 * self.cell
        )

        self.screen = pygame.display.set_mode(
            (1060, 690)
        )

        self.clock = pygame.time.Clock()

        self.title_font = pygame.font.SysFont(
            "segoeui",
            24,
            bold=True,
        )

        self.font = pygame.font.SysFont(
            "segoeui",
            17,
        )

        self.small_font = pygame.font.SysFont(
            "segoeui",
            14,
        )

        self.map_index = 0
        self.model_index = 0

        self.speed_values = [
            1000,
            600,
            350,
            200,
            100,
            50,
        ]

        self.speed_index = 2
        self.last_tick = 0
        self.running = False
        self.status = "Ready"
        self.last_info: dict = {}

        self.trail: list[
            tuple[int, int]
        ] = []

        self.buttons = [
            (
                "Start / Pause",
                "toggle",
                pygame.Rect(
                    self.panel_x,
                    250,
                    160,
                    38,
                ),
            ),
            (
                "Single Step",
                "step",
                pygame.Rect(
                    self.panel_x + 172,
                    250,
                    160,
                    38,
                ),
            ),
            (
                "Reset",
                "reset",
                pygame.Rect(
                    self.panel_x,
                    300,
                    160,
                    38,
                ),
            ),
            (
                "Next Model",
                "model",
                pygame.Rect(
                    self.panel_x + 172,
                    300,
                    160,
                    38,
                ),
            ),
            (
                "Next Map",
                "map",
                pygame.Rect(
                    self.panel_x,
                    350,
                    160,
                    38,
                ),
            ),
            (
                "Faster",
                "faster",
                pygame.Rect(
                    self.panel_x + 172,
                    350,
                    160,
                    38,
                ),
            ),
            (
                "Slower",
                "slower",
                pygame.Rect(
                    self.panel_x,
                    400,
                    160,
                    38,
                ),
            ),
        ]

        self.load_current_selection()

    @property
    def map_data(self) -> dict:
        """Return the selected map information."""

        return MAPS[self.map_index]

    @property
    def model_data(
        self,
    ) -> tuple[str, Path, str]:
        """Return the selected model information."""

        return self.map_data[
            "models"
        ][self.model_index]

    def load_current_selection(
        self,
    ) -> None:
        """Load the selected environment and model."""

        (
            model_name,
            model_path,
            reward_mode,
        ) = self.model_data

        self.env = MazeEnv(
            map_path=self.map_data["path"],
            transition_seed=8,
            reward_mode=reward_mode,
            gamma=0.95,
        )

        try:
            self.policy = load_policy(
                model_path
            )

            self.status = (
                f"Loaded: {model_name}"
            )

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
        ) as error:
            self.policy = {}
            self.status = str(error)

        self.reset()

    def reset(self) -> None:
        """Reset the current episode."""

        state, info = self.env.reset(
            seed=8
        )

        self.running = False
        self.last_info = info

        self.trail = [
            (
                state[0],
                state[1],
            )
        ]

        if self.policy:
            self.status = "Ready"

    def step(self) -> None:
        """Execute one action from the learned policy."""

        if self.env.done:
            self.running = False
            return

        state = self.env.get_state()

        action = self.policy.get(
            state
        )

        if action is None:
            self.running = False

            if self.env.is_terminal_state(
                state
            ):
                self.status = (
                    "Goal reached."
                )
            else:
                self.status = (
                    "No action for this state."
                )

            return

        (
            next_state,
            _,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        self.last_info = info

        self.trail.append(
            (
                next_state[0],
                next_state[1],
            )
        )

        events = info.get(
            "events",
            [],
        )

        if events:
            self.status = ", ".join(
                events
            )
        else:
            self.status = "Moving"

        if terminated:
            self.status = (
                "Success: goal reached."
            )

        elif truncated:
            self.status = (
                "Stopped: step limit."
            )

        if terminated or truncated:
            self.running = False

    def perform(
        self,
        action: str,
    ) -> None:
        """Execute a GUI control action."""

        if action == "toggle":
            if (
                self.policy
                and not self.env.done
            ):
                self.running = (
                    not self.running
                )

        elif action == "step":
            self.running = False
            self.step()

        elif action == "reset":
            self.reset()

        elif action == "model":
            model_count = len(
                self.map_data["models"]
            )

            self.model_index = (
                self.model_index + 1
            ) % model_count

            self.load_current_selection()

        elif action == "map":
            self.map_index = (
                self.map_index + 1
            ) % len(MAPS)

            self.model_index = 0
            self.load_current_selection()

        elif action == "faster":
            self.speed_index = min(
                self.speed_index + 1,
                len(self.speed_values) - 1,
            )

        elif action == "slower":
            self.speed_index = max(
                self.speed_index - 1,
                0,
            )

    def handle_events(self) -> bool:
        """Process keyboard, mouse and close events."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                key_actions = {
                    pygame.K_SPACE: "toggle",
                    pygame.K_n: "step",
                    pygame.K_r: "reset",
                    pygame.K_m: "model",
                    pygame.K_e: "map",
                    pygame.K_PLUS: "faster",
                    pygame.K_EQUALS: "faster",
                    pygame.K_KP_PLUS:
                        "faster",
                    pygame.K_MINUS: "slower",
                    pygame.K_KP_MINUS:
                        "slower",
                }

                if event.key == pygame.K_ESCAPE:
                    return False

                if event.key in key_actions:
                    self.perform(
                        key_actions[event.key]
                    )

            if (
                event.type
                == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                for (
                    _,
                    action,
                    rectangle,
                ) in self.buttons:
                    if rectangle.collidepoint(
                        event.pos
                    ):
                        self.perform(action)
                        break

        return True

    def draw_label(
        self,
        text: str,
        rectangle: pygame.Rect,
    ) -> None:
        """Draw a centered cell label."""

        surface = self.small_font.render(
            text,
            True,
            COLORS["text"],
        )

        self.screen.blit(
            surface,
            surface.get_rect(
                center=rectangle.center
            ),
        )

    def draw_maze(self) -> None:
        """Draw the maze, trail and agent."""

        visited = set(
            self.trail[:-1]
        )

        for row in range(
            self.env.size
        ):
            for column in range(
                self.env.size
            ):
                rectangle = pygame.Rect(
                    self.margin
                    + column * self.cell,
                    self.margin
                    + row * self.cell,
                    self.cell,
                    self.cell,
                )

                symbol = self.env.grid[
                    row
                ][column]

                color = COLORS["empty"]
                label = ""

                if (
                    symbol
                    == self.env.wall_symbol
                ):
                    color = COLORS["wall"]

                elif (
                    symbol
                    == self.env.start_symbol
                ):
                    color = COLORS["start"]
                    label = "S"

                elif (
                    symbol
                    == self.env.key_symbol
                ):
                    color = COLORS["key"]

                    if not self.env.has_key:
                        label = "K"

                elif (
                    symbol
                    == self.env.door_symbol
                ):
                    if self.env.has_key:
                        color = COLORS[
                            "door_open"
                        ]

                        label = "OPEN"
                    else:
                        color = COLORS[
                            "door_closed"
                        ]

                        label = "D"

                elif (
                    symbol
                    == self.env.goal_symbol
                ):
                    color = COLORS["goal"]
                    label = "G"

                elif (
                    symbol
                    == self.env.penalty_symbol
                ):
                    color = COLORS["penalty"]
                    label = "P"

                elif (
                    symbol
                    == self.env.gate_symbol
                ):
                    gate_open = (
                        self.env.is_gate_open()
                    )

                    if gate_open:
                        color = COLORS[
                            "gate_open"
                        ]

                        label = "OPEN"
                    else:
                        color = COLORS[
                            "gate_closed"
                        ]

                        label = "GATE"

                elif (
                    row,
                    column,
                ) in visited:
                    color = COLORS["trail"]

                pygame.draw.rect(
                    self.screen,
                    color,
                    rectangle,
                )

                pygame.draw.rect(
                    self.screen,
                    COLORS["grid"],
                    rectangle,
                    1,
                )

                if label:
                    self.draw_label(
                        label,
                        rectangle,
                    )

        row, column = (
            self.env.position
        )

        center = (
            self.margin
            + column * self.cell
            + self.cell // 2,
            self.margin
            + row * self.cell
            + self.cell // 2,
        )

        pygame.draw.circle(
            self.screen,
            COLORS["agent"],
            center,
            self.cell // 3,
        )

        pygame.draw.circle(
            self.screen,
            COLORS["white"],
            center,
            self.cell // 3,
            2,
        )

    def text(
        self,
        value: str,
        x: int,
        y: int,
        *,
        small: bool = False,
        muted: bool = False,
    ) -> None:
        """Draw one text line."""

        font = (
            self.small_font
            if small
            else self.font
        )

        color = (
            COLORS["muted"]
            if muted
            else COLORS["text"]
        )

        surface = font.render(
            value,
            True,
            color,
        )

        self.screen.blit(
            surface,
            (x, y),
        )

    def draw_panel(self) -> None:
        """Draw information and controls."""

        panel = pygame.Rect(
            self.panel_x - 12,
            12,
            370,
            660,
        )

        pygame.draw.rect(
            self.screen,
            COLORS["panel"],
            panel,
            border_radius=12,
        )

        title = self.title_font.render(
            "Dynamic Maze",
            True,
            COLORS["text"],
        )

        self.screen.blit(
            title,
            (
                self.panel_x,
                28,
            ),
        )

        model_name, _, _ = (
            self.model_data
        )

        self.text(
            "Environment:",
            self.panel_x,
            72,
            muted=True,
        )

        self.text(
            self.map_data["name"],
            self.panel_x,
            96,
        )

        self.text(
            "Policy:",
            self.panel_x,
            128,
            muted=True,
        )

        self.text(
            model_name,
            self.panel_x,
            152,
        )

        self.text(
            f"State: {self.env.get_state()}",
            self.panel_x,
            188,
            small=True,
        )

        self.text(
            (
                f"Steps: "
                f"{self.env.step_count}"
                f" / {self.env.max_steps}"
            ),
            self.panel_x,
            210,
            small=True,
        )

        self.text(
            (
                "Reward: "
                f"{self.env.episode_reward:.2f}"
            ),
            self.panel_x,
            230,
            small=True,
        )

        mouse_position = (
            pygame.mouse.get_pos()
        )

        for (
            label,
            _,
            rectangle,
        ) in self.buttons:
            if rectangle.collidepoint(
                mouse_position
            ):
                color = COLORS[
                    "button_hover"
                ]
            else:
                color = COLORS["button"]

            pygame.draw.rect(
                self.screen,
                color,
                rectangle,
                border_radius=7,
            )

            label_surface = (
                self.small_font.render(
                    label,
                    True,
                    COLORS["white"],
                )
            )

            self.screen.blit(
                label_surface,
                label_surface.get_rect(
                    center=rectangle.center
                ),
            )

        self.text(
            (
                "Animation delay: "
                f"{self.speed_values[self.speed_index]}"
                " ms"
            ),
            self.panel_x,
            455,
            small=True,
        )

        self.text(
            (
                "Key: "
                + (
                    "collected"
                    if self.env.has_key
                    else "not collected"
                )
            ),
            self.panel_x,
            478,
            small=True,
        )

        self.text(
            (
                "Gate: "
                + (
                    "open"
                    if self.env.is_gate_open()
                    else "closed"
                )
            ),
            self.panel_x,
            500,
            small=True,
        )

        self.text(
            (
                "Intended: "
                f"{self.last_info.get('intended_action', '-')}"
            ),
            self.panel_x,
            525,
            small=True,
        )

        self.text(
            (
                "Actual: "
                f"{self.last_info.get('actual_action', '-')}"
            ),
            self.panel_x,
            547,
            small=True,
        )

        status = self.status

        if len(status) > 45:
            status = (
                status[:42] + "..."
            )

        self.text(
            "Status:",
            self.panel_x,
            575,
            small=True,
            muted=True,
        )

        self.text(
            status,
            self.panel_x,
            596,
            small=True,
        )

        self.text(
            (
                "Space start/pause | "
                "N step | R reset"
            ),
            self.panel_x,
            628,
            small=True,
            muted=True,
        )

        self.text(
            (
                "M model | E map | "
                "+/- speed | Esc exit"
            ),
            self.panel_x,
            648,
            small=True,
            muted=True,
        )

    def update(self) -> None:
        """Advance the animation."""

        if not self.running:
            return

        current_time = (
            pygame.time.get_ticks()
        )

        delay = self.speed_values[
            self.speed_index
        ]

        if (
            current_time - self.last_tick
            >= delay
        ):
            self.step()

            self.last_tick = (
                current_time
            )

    def run(self) -> None:
        """Run the Pygame window."""

        active = True

        while active:
            active = self.handle_events()

            self.update()

            self.screen.fill(
                COLORS["background"]
            )

            self.draw_maze()
            self.draw_panel()

            pygame.display.flip()

            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    MazeGUI().run()