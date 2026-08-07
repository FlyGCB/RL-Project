"""
Paddle Duel RL Arena
====================

A lightweight two-player Pong/Paddle-style environment for teaching Reinforcement Learning.

The environment intentionally avoids external RL libraries. It follows a small Gym-like API:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

For competitions, the same environment can pit two submitted agents against each other:

    obs_left, obs_right, info = env.reset_duel(seed=0)
    obs_left, obs_right, reward_left, reward_right, terminated, truncated, info = env.step_duel(left_action, right_action)

Levels
------
level=1: Solo paddle practice against a wall, tabular state.
level=2: Duel against a scripted opponent, tabular state.
level=3: Duel against an opponent zoo, tabular state.
level=4: Same mechanics as level 3, but the learning agent receives pixels only.
level=5: Same mechanics as level 4, but intended for two learned agents with pixel observations.

Actions
-------
0 = move up
1 = move down
2 = stay

Design notes
------------
- One step represents one simulated video frame.
- By default, a rally can last up to 60 seconds at 30 FPS, so 1800 steps.
- An episode is one rally: it terminates as soon as the first player scores.
- If nobody scores before the time limit, the rally times out; cumulative reward decides the winner.
- The ball speed gradually increases over simulated time, so defense becomes harder.
- The displayed score is the cumulative reward collected by each side.
- Student agents should be implemented as objects with a select_action(...) method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union, Any
import math
import random

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
except Exception:  # pragma: no cover
    plt = None
    FuncAnimation = None
    HTML = None

# ---------------------------------------------------------------------------
# Public action constants
# ---------------------------------------------------------------------------

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_STAY = 2
UP = ACTION_UP
DOWN = ACTION_DOWN
STAY = ACTION_STAY

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_STAY: "STAY",
}

Action = int
TabularState = Tuple[int, ...]
Observation = Union[TabularState, np.ndarray]


class AgentProtocol(Protocol):
    """Minimal interface expected by helper functions and competitions."""

    name: str

    def reset(self) -> None:
        ...

    def select_action(self, observation: Observation, info: Optional[Dict[str, Any]] = None, training: bool = False) -> int:
        ...


class BaseAgent:
    """Base class used by all agents in the project.

    Students may inherit from this class, but they only need to implement
    select_action(...). A submitted agent should never directly modify the
    environment. It should only observe, decide, and return one action.
    """

    name = "BaseAgent"

    def reset(self) -> None:
        """Called at the beginning of each episode/rally."""
        return None

    def select_action(self, observation: Observation, info: Optional[Dict[str, Any]] = None, training: bool = False) -> int:
        """Return one action: ACTION_UP, ACTION_DOWN, or ACTION_STAY."""
        raise NotImplementedError("Students must implement select_action(...).")


class RandomAgent(BaseAgent):
    """Baseline agent that samples actions uniformly at random."""

    def __init__(
        self,
        side: str = "left",
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        assert side in ["left", "right"], "side must be 'left' or 'right'"

        self.side = side
        self.name = name if name is not None else f"RandomAgent_{side}"
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called at the beginning of an episode."""
        pass

    def select_action(
        self,
        observation: Observation,
        info: Optional[Dict[str, Any]] = None,
        training: bool = False,
    ) -> int:
        return int(self.rng.integers(3))


class TrackingAgent(BaseAgent):
    """Simple hand-coded baseline: move paddle toward the ball.

    This agent works with tabular observations and with info dictionaries. It is
    useful as a scripted opponent and as a sanity-check baseline.
    """

    def __init__(self, side: str = "left", noise: float = 0.05, seed: Optional[int] = None, name: Optional[str] = None):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.name = name or f"TrackingAgent_{side}"

    def select_action(self, observation: Observation, info: Optional[Dict[str, Any]] = None, training: bool = False) -> int:
        if self.rng.random() < self.noise:
            return int(self.rng.integers(3))

        if info is not None:
            if self.side == "left":
                paddle_y = float(info["left_paddle_y"])
                ball_coming = float(info["ball_vx"]) < 0
            else:
                paddle_y = float(info["right_paddle_y"])
                ball_coming = float(info["ball_vx"]) > 0
            paddle_h = float(info["paddle_height"])
            ball_y = float(info["ball_y"])
        elif isinstance(observation, tuple):
            # Tabular state: (left_bin, right_bin, ball_x_bin, ball_y_bin, vx_bin, vy_bin, speed_bin)
            left_bin, right_bin, _, ball_y_bin, vx_bin, _, _ = observation
            paddle_y = float(left_bin if self.side == "left" else right_bin)
            paddle_h = 2.0
            ball_y = float(ball_y_bin)
            ball_coming = vx_bin == 0 if self.side == "left" else vx_bin == 1
        else:
            # Pixel-only agents should not usually use this baseline.
            return ACTION_STAY

        if not ball_coming and self.rng.random() < 0.80:
            return ACTION_STAY

        paddle_center = paddle_y + paddle_h / 2.0
        if ball_y < paddle_center - 1:
            return ACTION_UP
        if ball_y > paddle_center + 1:
            return ACTION_DOWN
        return ACTION_STAY


@dataclass
class Episode:
    """Container returned by record_episode(...).

    frames:
        Rendered RGB frames, useful for replay.
    observations:
        Observations returned by the environment before/after actions.
    actions:
        Actions selected by the learning agent.
    rewards:
        Rewards received after each action.
    infos:
        Info dictionaries returned by the environment.
    total_reward:
        Sum of rewards for the left/learning agent.
    terminated:
        True when a rally ended naturally because one player scored.
    truncated:
        True when a rally ended because the 60-second time limit was reached before any point was scored.
    """

    frames: List[np.ndarray]
    observations: List[Observation]
    actions: List[int]
    rewards: List[float]
    infos: List[Dict[str, Any]]
    total_reward: float
    terminated: bool
    truncated: bool




@dataclass
class DuelEpisode:
    """Container returned by record_duel_episode(...).

    This is useful for Level 5 and teacher competitions where both paddles are
    controlled by agent objects. It stores side-specific observations, actions,
    and rewards, plus rendered frames for replay/debugging.
    """

    frames: List[np.ndarray]
    left_observations: List[Observation]
    right_observations: List[Observation]
    left_actions: List[int]
    right_actions: List[int]
    left_rewards: List[float]
    right_rewards: List[float]
    infos: List[Dict[str, Any]]
    left_total_reward: float
    right_total_reward: float
    winner: Optional[str]
    terminated: bool
    truncated: bool

class PaddleDuelEnv:
    """Paddle Duel environment.

    Parameters
    ----------
    level:
        1: solo paddle vs wall, tabular state.
        2: duel vs one scripted opponent, tabular state.
        3: duel vs opponent zoo, tabular state.
        4: duel vs opponent zoo, pixel-only state.
        5: level 4 mechanics with two learned pixel agents in duel mode.
    width_px, height_px:
        Board size in pixels. A 640x360 board is large enough to look good but
        still cheap to render on a laptop.
    max_seconds:
        Simulated episode length. Default is 60 seconds.
    fps:
        Simulated frames per second. max_steps = max_seconds * fps.
    win_score:
        Deprecated compatibility argument from earlier drafts. In v5, an episode is
        exactly one rally and therefore terminates as soon as either player scores.
        The value of win_score is ignored.
    ball_speed_start, ball_speed_max, ball_speedup_per_second:
        Ball speed schedule. The speed increases gently over simulated time.
    obs_mode:
        "tabular" or "pixels". If None, inferred from level.
    opponent:
        For level 2/3/4. One of "random", "weak", "tracking", "strong", "zoo".
    render_scale:
        Downscale factor for pixel observations. Pixel observations crop the play area and exclude the score bar. Use render_scale=1 for full-resolution pixels or render_scale=2 for a CPU-friendly high-resolution state.
    seed:
        Optional initial random seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        level: int = 1,
        width_px: int = 640,
        height_px: int = 360,
        paddle_width: int = 14,
        paddle_height: int = 76,
        paddle_speed: float = 7.0,
        max_seconds: float = 60.0,
        fps: int = 30,
        win_score: Optional[int] = None,
        ball_radius: int = 7,
        ball_speed_start: float = 9.45,
        ball_speed_max: float = 18.9,
        ball_speedup_per_second: float = 0.2475,
        ball_speedup_on_hit: float = 0.225,
        point_reward: float = 10.0,
        hit_reward: float = 0.06,
        obs_mode: Optional[str] = None,
        opponent: Optional[str] = None,
        render_scale: int = 2,
        seed: Optional[int] = None,
    ):
        if level not in (1, 2, 3, 4, 5):
            raise ValueError("level must be 1, 2, 3, 4, or 5")
        self.level = int(level)
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.paddle_width = int(paddle_width)
        self.paddle_height = int(paddle_height)
        self.paddle_speed = float(paddle_speed)
        self.max_seconds = float(max_seconds)
        self.fps = int(fps)
        self.max_steps = int(round(self.max_seconds * self.fps))
        self.win_score = win_score
        self.ball_radius = int(ball_radius)
        self.ball_speed_start = float(ball_speed_start)
        self.ball_speed_max = float(ball_speed_max)
        self.ball_speedup_per_second = float(ball_speedup_per_second)
        self.ball_speedup_on_hit = float(ball_speedup_on_hit)
        self.point_reward = float(point_reward)
        self.hit_reward = float(hit_reward)
        self.obs_mode = obs_mode or ("pixels" if self.level in (4, 5) else "tabular")
        if self.obs_mode not in ("tabular", "pixels"):
            raise ValueError("obs_mode must be 'tabular' or 'pixels'")
        if self.level in (4, 5) and self.obs_mode != "pixels":
            raise ValueError("levels 4 and 5 are pixels-only by design")
        self.opponent = opponent or ("wall" if self.level == 1 else ("zoo" if self.level in (3, 4, 5) else "weak"))
        self.render_scale = int(render_scale)
        if self.render_scale < 1:
            raise ValueError("render_scale must be >= 1")

        self.left_x = 42.0
        self.right_x = self.width_px - 42.0 - self.paddle_width
        # Top area is reserved for the live score, so it does not overlap the play area.
        self.top = 42.0
        self.bottom = self.height_px - 18.0
        self.min_paddle_y = self.top
        self.max_paddle_y = self.bottom - self.paddle_height

        self.rng = np.random.default_rng(seed)
        self.py_random = random.Random(seed)
        self.reset(seed=seed)

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def action_space_n(self) -> int:
        return 3

    def observation_description(self) -> str:
        """Human-readable description of the current observation mode."""
        if self.obs_mode == "pixels":
            obs_h = int((self.bottom - self.top) // self.render_scale)
            obs_w = self.width_px // self.render_scale
            return (
                f"Pixel observation: RGB image with shape approximately ({obs_h}, {obs_w}, 3). "
                "The image shows only the play area: left paddle, right paddle, ball, borders, and centre line. "
                "The live score bar is deliberately excluded from the pixel state so agents cannot overfit to the scoreboard. "
                "Agents must extract useful information from pixels, either with a CNN or with helper features."
            )
        return (
            "Tabular observation tuple: "
            "(left_paddle_y_bin, right_paddle_y_bin, ball_x_bin, ball_y_bin, ball_vx_sign, ball_vy_sign, speed_bin). "
            "The bins discretize continuous pixel positions, making the state small enough for Q-tables."
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[Observation, Dict[str, Any]]:
        """Reset the environment for the usual single-learning-agent setup."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.py_random = random.Random(seed)

        self.steps = 0
        self.elapsed_seconds = 0.0
        # left_points/right_points are traditional Pong points. In v5 an episode is one rally,
        # so at the end they are either 1-0, 0-1, or 0-0 after a timeout draw.
        # left_score/right_score are the displayed cumulative rewards used for ranking.
        self.left_points = 0
        self.right_points = 0
        self.left_total_reward = 0.0
        self.right_total_reward = 0.0
        self.left_score = 0.0
        self.right_score = 0.0
        self.left_hits = 0
        self.right_hits = 0
        self.done_reason = None
        self.winner = None
        self.last_scorer = None
        self.first_score_step = {"left": None, "right": None}

        center_y = (self.top + self.bottom - self.paddle_height) / 2.0
        self.left_paddle_y = center_y
        self.right_paddle_y = center_y

        self.active_opponent = self._sample_opponent_type()
        self._reset_ball(direction=-1 if self.level > 1 else 1)

        obs = self._get_obs(side="left")
        info = self._get_info()
        return obs, info

    def reset_duel(self, seed: Optional[int] = None) -> Tuple[Observation, Observation, Dict[str, Any]]:
        """Reset environment and return observations for both paddles.

        This is used by the teacher competition notebook. Left and right agents
        receive side-adjusted observations: for the right agent, the x-axis and
        velocity signs are mirrored so the right agent can use similar logic.
        """
        obs_left, info = self.reset(seed=seed)
        obs_right = self._get_obs(side="right")
        return obs_left, obs_right, info

    def _sample_opponent_type(self) -> str:
        if self.opponent == "zoo":
            return self.py_random.choices(
                ["random", "weak", "tracking", "strong"],
                weights=[0.15, 0.35, 0.35, 0.15],
                k=1,
            )[0]
        return self.opponent

    def _reset_ball(self, direction: Optional[int] = None) -> None:
        self.ball_x = self.width_px / 2.0
        self.ball_y = float(self.rng.uniform(self.top + 40, self.bottom - 40))
        if direction is None:
            direction = int(self.rng.choice([-1, 1]))
        angle = float(self.rng.uniform(-0.55, 0.55))
        speed = self.ball_speed_start
        self.ball_vx = float(direction * speed * math.cos(angle))
        self.ball_vy = float(speed * math.sin(angle))
        # Avoid nearly flat openings every time.
        if abs(self.ball_vy) < 0.5:
            self.ball_vy += float(self.rng.choice([-0.8, 0.8]))

    def _current_speed_limit(self) -> float:
        return min(
            self.ball_speed_max,
            self.ball_speed_start + self.ball_speedup_per_second * self.elapsed_seconds,
        )

    def _renormalize_ball_speed(self, extra: float = 0.0) -> None:
        speed = math.sqrt(self.ball_vx ** 2 + self.ball_vy ** 2)
        target = min(self.ball_speed_max, max(speed, self._current_speed_limit()) + extra)
        if speed <= 1e-6:
            self.ball_vx = target
            self.ball_vy = 0.0
        else:
            self.ball_vx *= target / speed
            self.ball_vy *= target / speed

    def _move_paddle(self, y: float, action: int, speed_scale: float = 1.0) -> float:
        if action == ACTION_UP:
            y -= self.paddle_speed * speed_scale
        elif action == ACTION_DOWN:
            y += self.paddle_speed * speed_scale
        elif action == ACTION_STAY:
            pass
        else:
            raise ValueError(f"Unknown action: {action}. Use 0=up, 1=down, 2=stay.")
        return float(np.clip(y, self.min_paddle_y, self.max_paddle_y))

    def _scripted_opponent_action(self) -> int:
        if self.level == 1:
            return ACTION_STAY
        kind = self.active_opponent
        if kind == "random":
            return int(self.rng.choice([ACTION_UP, ACTION_DOWN, ACTION_STAY], p=[0.28, 0.28, 0.44]))

        opponent_center = self.right_paddle_y + self.paddle_height / 2.0
        target_y = self.ball_y
        ball_coming = self.ball_vx > 0

        if kind == "weak":
            if not ball_coming and self.rng.random() < 0.80:
                return ACTION_STAY
            if self.rng.random() < 0.45:
                return ACTION_STAY
        elif kind == "tracking":
            if not ball_coming and self.rng.random() < 0.50:
                return ACTION_STAY
            if self.rng.random() < 0.10:
                return int(self.rng.choice([ACTION_UP, ACTION_DOWN, ACTION_STAY]))
        elif kind == "strong":
            if not ball_coming and self.rng.random() < 0.25:
                return ACTION_STAY
            if self.rng.random() < 0.03:
                return int(self.rng.choice([ACTION_UP, ACTION_DOWN, ACTION_STAY]))
        else:
            raise ValueError(f"Unknown opponent type: {kind}")

        if target_y < opponent_center - 4:
            return ACTION_UP
        if target_y > opponent_center + 4:
            return ACTION_DOWN
        return ACTION_STAY

    def step(self, action: int) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        """Advance one step with a learning agent controlling the left paddle.

        The right paddle is controlled by the environment's scripted opponent.
        """
        right_action = self._scripted_opponent_action()
        obs_left, _, reward_left, _, terminated, truncated, info = self.step_duel(action, right_action)
        return obs_left, reward_left, terminated, truncated, info

    def step_duel(
        self,
        left_action: int,
        right_action: int,
    ) -> Tuple[Observation, Observation, float, float, bool, bool, Dict[str, Any]]:
        """Advance one frame with two externally controlled paddles.

        This method is for two-agent control and competitions. It returns side-adjusted observations
        for both agents and side-specific rewards.
        """
        self.steps += 1
        self.elapsed_seconds = self.steps / self.fps

        # Tiny movement penalty used as a draw/tie-breaker.
        # Staying still is not penalized; only moving up/down consumes effort.
        reward_left = -0.001 if int(left_action) in (ACTION_UP, ACTION_DOWN) else 0.0
        reward_right = -0.001 if int(right_action) in (ACTION_UP, ACTION_DOWN) else 0.0
        terminated = False
        truncated = False

        self.left_paddle_y = self._move_paddle(self.left_paddle_y, int(left_action), speed_scale=1.0)
        self.right_paddle_y = self._move_paddle(self.right_paddle_y, int(right_action), speed_scale=1.0)

        # Gradually increase ball speed over time, without changing its direction.
        self._renormalize_ball_speed(extra=0.0)

        next_x = self.ball_x + self.ball_vx
        next_y = self.ball_y + self.ball_vy

        # Top/bottom wall bounce.
        if next_y - self.ball_radius < self.top:
            next_y = self.top + self.ball_radius
            self.ball_vy = abs(self.ball_vy)
        elif next_y + self.ball_radius > self.bottom:
            next_y = self.bottom - self.ball_radius
            self.ball_vy = -abs(self.ball_vy)

        # Left paddle collision / right scores.
        if self.ball_vx < 0 and next_x - self.ball_radius <= self.left_x + self.paddle_width:
            if self._paddle_hits(self.left_paddle_y, next_y):
                self.left_hits += 1
                reward_left += self.hit_reward
                next_x = self.left_x + self.paddle_width + self.ball_radius + 0.5
                self.ball_vx = abs(self.ball_vx)
                self._adjust_angle_after_hit(self.left_paddle_y, next_y, direction=1)
                self._renormalize_ball_speed(extra=self.ball_speedup_on_hit)
            elif next_x + self.ball_radius < 0:
                # Right scores. In v5, one episode is one rally, so the episode
                # terminates immediately after the first score. There is no
                # separate rally-win bonus.
                self.right_points += 1
                reward_right += self.point_reward
                self._register_score("right")
                self.done_reason = "right_scored"
                self.winner = "right"
                terminated = True

        # Right paddle collision / left scores.
        elif self.ball_vx > 0 and next_x + self.ball_radius >= self.right_x:
            if self.level == 1:
                # Right side is a wall in solo practice.
                next_x = self.right_x - self.ball_radius - 0.5
                self.ball_vx = -abs(self.ball_vx)
                self._renormalize_ball_speed(extra=self.ball_speedup_on_hit * 0.5)
            elif self._paddle_hits(self.right_paddle_y, next_y):
                self.right_hits += 1
                reward_right += self.hit_reward
                next_x = self.right_x - self.ball_radius - 0.5
                self.ball_vx = -abs(self.ball_vx)
                self._adjust_angle_after_hit(self.right_paddle_y, next_y, direction=-1)
                self._renormalize_ball_speed(extra=self.ball_speedup_on_hit)
            elif next_x - self.ball_radius > self.width_px:
                # Left scores. In v5, one episode is one rally, so the episode
                # terminates immediately after the first score. There is no
                # separate rally-win bonus.
                self.left_points += 1
                reward_left += self.point_reward
                self._register_score("left")
                self.done_reason = "left_scored"
                self.winner = "left"
                terminated = True

        self.ball_x = float(next_x)
        self.ball_y = float(next_y)

        # Level 1 uses hits as the main success signal, but still times out at the configured duration.
        if self.level == 1 and self.left_hits >= 10:
            # Do not terminate immediately by default; keeping the rally alive is useful.
            reward_left += 0.02

        # Timeout means nobody managed to score during the rally.
        # There is no additional terminal reward on timeout.
        # The official decider is the cumulative reward collected up to the end
        # of the rally. This includes the current step reward.
        if self.steps >= self.max_steps and not terminated:
            truncated = True
            self.done_reason = "timeout"
            provisional_left = self.left_total_reward + float(reward_left)
            provisional_right = self.right_total_reward + float(reward_right)
            if provisional_left > provisional_right:
                self.winner = "left"
            elif provisional_right > provisional_left:
                self.winner = "right"
            else:
                self.winner = None

        self.left_total_reward += float(reward_left)
        self.right_total_reward += float(reward_right)
        self.left_score = float(self.left_total_reward)
        self.right_score = float(self.right_total_reward)

        obs_left = self._get_obs(side="left")
        obs_right = self._get_obs(side="right")
        info = self._get_info()
        return obs_left, obs_right, float(reward_left), float(reward_right), terminated, truncated, info

    def _register_score(self, side: str) -> None:
        self.last_scorer = side
        if self.first_score_step[side] is None:
            self.first_score_step[side] = self.steps

    def _winner_from_score(self) -> Optional[str]:
        """Return the winner from traditional Pong points.

        In v5 this is mostly a helper for diagnostics: the episode terminates
        immediately after the first point, while timeout with 0-0 uses cumulative reward as the decider.
        """
        if self.left_points > self.right_points:
            return "left"
        if self.right_points > self.left_points:
            return "right"
        return None

    def _paddle_hits(self, paddle_y: float, ball_y: float) -> bool:
        return paddle_y - self.ball_radius <= ball_y <= paddle_y + self.paddle_height + self.ball_radius

    def _adjust_angle_after_hit(self, paddle_y: float, ball_y: float, direction: int) -> None:
        """Change ball velocity depending on where it hit the paddle."""
        paddle_center = paddle_y + self.paddle_height / 2.0
        offset = (ball_y - paddle_center) / (self.paddle_height / 2.0)
        offset = float(np.clip(offset, -1.0, 1.0))
        speed = math.sqrt(self.ball_vx ** 2 + self.ball_vy ** 2)
        max_angle = math.radians(62)
        angle = offset * max_angle
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)

    def _get_obs(self, side: str = "left") -> Observation:
        if self.obs_mode == "pixels":
            # Pixel observations deliberately exclude the score bar. The agent sees
            # the play area only: paddles, ball, borders, and centre line.
            frame = self.render(show_score=False)
            frame = frame[int(self.top): int(self.bottom), :, :].copy()
            if self.render_scale > 1:
                frame = frame[:: self.render_scale, :: self.render_scale].copy()
            if side == "right":
                frame = np.flip(frame, axis=1).copy()
            return frame
        return self._get_tabular_state(side=side)

    def _bin(self, value: float, low: float, high: float, n_bins: int) -> int:
        value = float(np.clip(value, low, high))
        if high <= low:
            return 0
        scaled = (value - low) / (high - low)
        return int(np.clip(math.floor(scaled * n_bins), 0, n_bins - 1))

    def _get_tabular_state(self, side: str = "left") -> TabularState:
        n_y = 12
        n_x = 16
        n_speed = 5

        left_y_bin = self._bin(self.left_paddle_y, self.min_paddle_y, self.max_paddle_y, n_y)
        right_y_bin = self._bin(self.right_paddle_y, self.min_paddle_y, self.max_paddle_y, n_y)
        ball_x_bin = self._bin(self.ball_x, 0, self.width_px, n_x)
        ball_y_bin = self._bin(self.ball_y, self.top, self.bottom, n_y)
        speed = math.sqrt(self.ball_vx ** 2 + self.ball_vy ** 2)
        speed_bin = self._bin(speed, self.ball_speed_start, self.ball_speed_max, n_speed)
        vx_sign = 0 if self.ball_vx < 0 else 1
        vy_sign = 0 if self.ball_vy < -0.4 else (2 if self.ball_vy > 0.4 else 1)

        if side == "left":
            return (left_y_bin, right_y_bin, ball_x_bin, ball_y_bin, vx_sign, vy_sign, speed_bin)
        if side == "right":
            # Mirror x-axis and swap paddles so the right agent sees itself as "my paddle".
            mirrored_x = (n_x - 1) - ball_x_bin
            mirrored_vx = 0 if vx_sign == 1 else 1
            return (right_y_bin, left_y_bin, mirrored_x, ball_y_bin, mirrored_vx, vy_sign, speed_bin)
        raise ValueError("side must be 'left' or 'right'")

    def _get_info(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "obs_mode": self.obs_mode,
            "steps": self.steps,
            "elapsed_seconds": self.elapsed_seconds,
            "max_seconds": self.max_seconds,
            "fps": self.fps,
            "left_score": self.left_score,
            "right_score": self.right_score,
            "score_diff": self.left_score - self.right_score,
            "left_points": self.left_points,
            "right_points": self.right_points,
            "points_diff": self.left_points - self.right_points,
            "left_hits": self.left_hits,
            "right_hits": self.right_hits,
            "left_total_reward": self.left_total_reward,
            "right_total_reward": self.right_total_reward,
            "left_paddle_y": self.left_paddle_y,
            "right_paddle_y": self.right_paddle_y,
            "paddle_height": self.paddle_height,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_vx": self.ball_vx,
            "ball_vy": self.ball_vy,
            "ball_speed": math.sqrt(self.ball_vx ** 2 + self.ball_vy ** 2),
            "opponent_type": self.active_opponent,
            "done_reason": self.done_reason,
            "winner": self.winner,
            "left_won": self.winner == "left",
            "right_won": self.winner == "right",
            # point_draw is True when nobody scored before the timeout. In that
            # case, winner may still be left/right because cumulative reward is
            # used as the official decider.
            "point_draw": self.done_reason == "timeout",
            "reward_decider_used": self.done_reason == "timeout",
            "draw": self.done_reason == "timeout" and self.winner is None,
            "last_scorer": self.last_scorer,
        }

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def render(self, show_score: bool = True) -> np.ndarray:
        """Return a full-size RGB frame.

        Parameters
        ----------
        show_score:
            If True, draw cumulative reward scores in the top score bar. Pixel observations use show_score=False and crop the play area.
        """
        img = np.zeros((self.height_px, self.width_px, 3), dtype=np.uint8)
        img[:, :] = np.array([8, 13, 28], dtype=np.uint8)

        # Subtle court gradient/stripes.
        for y in range(self.height_px):
            shade = int(8 + 10 * y / max(1, self.height_px - 1))
            img[y, :, 2] = np.clip(img[y, :, 2] + shade, 0, 255)

        # Border.
        self._draw_rect_px(img, 0, 0, self.width_px, int(self.top), (33, 45, 78))
        self._draw_rect_px(img, 0, int(self.bottom), self.width_px, self.height_px - int(self.bottom), (33, 45, 78))
        self._draw_rect_px(img, 0, 0, 10, self.height_px, (33, 45, 78))
        self._draw_rect_px(img, self.width_px - 10, 0, 10, self.height_px, (33, 45, 78))

        # Centre dashed line.
        dash_h = 18
        for y in range(int(self.top) + 8, int(self.bottom), dash_h * 2):
            self._draw_rect_px(img, self.width_px // 2 - 2, y, 4, dash_h, (55, 68, 112))

        # Live cumulative reward score.
        if show_score:
            self._draw_score(img)

        # Paddles.
        self._draw_paddle_px(img, self.left_x, self.left_paddle_y, (62, 235, 180), (176, 255, 226))
        if self.level == 1:
            self._draw_rect_px(img, int(self.right_x), int(self.top), self.paddle_width, int(self.bottom - self.top), (85, 95, 125))
        else:
            self._draw_paddle_px(img, self.right_x, self.right_paddle_y, (255, 96, 128), (255, 179, 193))

        # Ball.
        self._draw_circle_px(img, self.ball_x, self.ball_y, self.ball_radius + 3, (255, 255, 215))
        self._draw_circle_px(img, self.ball_x, self.ball_y, self.ball_radius, (255, 222, 89))

        return img

    def _draw_score(self, img: np.ndarray) -> None:
        """Draw numeric left/right scores in the reserved top bar.

        The digits are neutral gray/white rather than paddle-colored so pixel
        feature extractors do not confuse score text with paddles or ball.
        """
        color = (238, 242, 255)
        shadow = (20, 27, 48)
        y = 8
        left_text = str(int(round(self.left_score)))
        right_text = str(int(round(self.right_score)))
        self._draw_number_px(img, left_text, 22, y, color=color, shadow=shadow)
        right_w = self._number_width_px(right_text)
        self._draw_number_px(img, right_text, self.width_px - 22 - right_w, y, color=color, shadow=shadow)

    _DIGIT_SEGMENTS = {
        "0": "abcfed", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
        "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd",
        "-": "g",
    }

    @classmethod
    def _number_width_px(cls, text: str, digit_w: int = 16, gap: int = 5) -> int:
        return max(1, len(text)) * digit_w + max(0, len(text) - 1) * gap

    def _draw_number_px(self, img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], shadow: Tuple[int, int, int]) -> None:
        digit_w = 16
        gap = 5
        for i, ch in enumerate(text):
            dx = x + i * (digit_w + gap)
            self._draw_digit_px(img, ch, dx + 1, y + 1, shadow)
            self._draw_digit_px(img, ch, dx, y, color)

    def _draw_digit_px(self, img: np.ndarray, ch: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
        segs = self._DIGIT_SEGMENTS.get(ch, "")
        t = 3
        w = 16
        h = 24
        # Segment coordinates: a top, b upper-right, c lower-right, d bottom, e lower-left, f upper-left, g middle.
        boxes = {
            "a": (x + 3, y, w - 6, t),
            "b": (x + w - t, y + 3, t, h // 2 - 4),
            "c": (x + w - t, y + h // 2 + 2, t, h // 2 - 5),
            "d": (x + 3, y + h - t, w - 6, t),
            "e": (x, y + h // 2 + 2, t, h // 2 - 5),
            "f": (x, y + 3, t, h // 2 - 4),
            "g": (x + 3, y + h // 2 - 1, w - 6, t),
        }
        for seg in segs:
            self._draw_rect_px(img, *boxes[seg], color)

    @staticmethod
    def _draw_rect_px(img: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int]) -> None:
        x0 = max(0, int(x)); x1 = min(img.shape[1], int(x + w))
        y0 = max(0, int(y)); y1 = min(img.shape[0], int(y + h))
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)

    def _draw_paddle_px(self, img: np.ndarray, x: float, y: float, color: Tuple[int, int, int], edge: Tuple[int, int, int]) -> None:
        x = int(round(x)); y = int(round(y))
        self._draw_rect_px(img, x - 2, y - 2, self.paddle_width + 4, self.paddle_height + 4, edge)
        self._draw_rect_px(img, x, y, self.paddle_width, self.paddle_height, color)
        # Small highlight stripe.
        self._draw_rect_px(img, x + 2, y + 4, max(2, self.paddle_width // 4), self.paddle_height - 8, (230, 255, 245))

    @staticmethod
    def _draw_circle_px(img: np.ndarray, cx: float, cy: float, r: int, color: Tuple[int, int, int]) -> None:
        cx = int(round(cx)); cy = int(round(cy)); r = int(r)
        y0 = max(0, cy - r); y1 = min(img.shape[0], cy + r + 1)
        x0 = max(0, cx - r); x1 = min(img.shape[1], cx + r + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        patch = img[y0:y1, x0:x1]
        patch[mask] = np.array(color, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Helper functions for states, training, replay, and competition
# ---------------------------------------------------------------------------


def random_argmax(values: Sequence[float], rng: Optional[np.random.Generator] = None) -> int:
    """Argmax with random tie-breaking."""
    rng = rng or np.random.default_rng()
    values = np.asarray(values, dtype=float)
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def ensure_q_state(Q: Dict[TabularState, np.ndarray], state: TabularState, n_actions: int = 3) -> None:
    """Create Q[state] if it does not exist yet."""
    if state not in Q:
        Q[state] = np.zeros(n_actions, dtype=np.float32)


def epsilon_greedy_action(Q: Dict[TabularState, np.ndarray], state: TabularState, n_actions: int, epsilon: float, rng=None) -> int:
    """Sample an epsilon-greedy action from a tabular Q dictionary."""
    rng = rng or np.random.default_rng()
    ensure_q_state(Q, state, n_actions)
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return random_argmax(Q[state], rng)


def describe_observation(env: PaddleDuelEnv) -> None:
    """Print a student-friendly explanation of the observation returned by env.reset/step."""
    print(env.observation_description())
    print("\nAction meanings:")
    for k, v in ACTION_NAMES.items():
        print(f"  {k}: {v}")


def decode_tabular_state(state: TabularState) -> Dict[str, int]:
    """Return a dictionary explaining a tabular state tuple."""
    names = [
        "my_paddle_y_bin",
        "other_paddle_y_bin",
        "ball_x_bin",
        "ball_y_bin",
        "ball_vx_sign",
        "ball_vy_sign",
        "speed_bin",
    ]
    return dict(zip(names, map(int, state)))


def run_episode(
    env: PaddleDuelEnv,
    agent_or_policy: Union[AgentProtocol, Callable],
    seed: Optional[int] = None,
    training: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Run one single-agent episode and return total reward plus final info.

    The learning agent controls the left paddle. The right paddle is scripted by
    the environment. agent_or_policy may be an object with select_action(...) or
    a simple function.
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    if hasattr(agent_or_policy, "reset"):
        agent_or_policy.reset()

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = _select_action(agent_or_policy, obs, info, training=training)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    return float(total_reward), info


def _select_action(agent_or_policy: Union[AgentProtocol, Callable], obs: Observation, info: Dict[str, Any], training: bool = False) -> int:
    if hasattr(agent_or_policy, "select_action"):
        return int(agent_or_policy.select_action(obs, info=info, training=training))
    try:
        return int(agent_or_policy(obs, info))
    except TypeError:
        return int(agent_or_policy(obs))


def record_episode(
    env: PaddleDuelEnv,
    agent_or_policy: Union[AgentProtocol, Callable],
    seed: Optional[int] = None,
    training: bool = False,
) -> Episode:
    """Run one episode and keep rendered frames for replay.

    This is the main helper behind animate_episode(...). Students can inspect
    episode.frames, episode.actions, episode.rewards, and episode.infos to debug
    what their agent actually did.
    """
    obs, info = env.reset(seed=seed)
    if hasattr(agent_or_policy, "reset"):
        agent_or_policy.reset()

    frames = [env.render()]
    observations = [obs]
    actions: List[int] = []
    rewards: List[float] = []
    infos = [info]
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = _select_action(agent_or_policy, obs, info, training=training)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frames.append(env.render())
        observations.append(obs)
        actions.append(int(action))
        rewards.append(float(reward))
        infos.append(info)

    return Episode(
        frames=frames,
        observations=observations,
        actions=actions,
        rewards=rewards,
        infos=infos,
        total_reward=float(total_reward),
        terminated=terminated,
        truncated=truncated,
    )



def record_duel_episode(
    env: PaddleDuelEnv,
    left_agent: AgentProtocol,
    right_agent: AgentProtocol,
    seed: Optional[int] = None,
    training: bool = False,
) -> DuelEpisode:
    """Run a two-agent episode and keep both agents' trajectories.

    The returned object contains lists for both sides:
    left_observations/right_observations, left_actions/right_actions,
    left_rewards/right_rewards, infos, frames, and the final winner.
    """
    obs_left, obs_right, info = env.reset_duel(seed=seed)
    left_agent.reset()
    right_agent.reset()

    frames = [env.render()]
    left_observations = [obs_left]
    right_observations = [obs_right]
    left_actions: List[int] = []
    right_actions: List[int] = []
    left_rewards: List[float] = []
    right_rewards: List[float] = []
    infos = [info]
    left_total_reward = 0.0
    right_total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        left_action = int(left_agent.select_action(obs_left, info=info, training=training))
        right_action = int(right_agent.select_action(obs_right, info=_mirror_info_for_right(info), training=training))
        obs_left, obs_right, reward_left, reward_right, terminated, truncated, info = env.step_duel(left_action, right_action)
        left_total_reward += float(reward_left)
        right_total_reward += float(reward_right)
        frames.append(env.render())
        left_observations.append(obs_left)
        right_observations.append(obs_right)
        left_actions.append(left_action)
        right_actions.append(right_action)
        left_rewards.append(float(reward_left))
        right_rewards.append(float(reward_right))
        infos.append(info)

    return DuelEpisode(
        frames=frames,
        left_observations=left_observations,
        right_observations=right_observations,
        left_actions=left_actions,
        right_actions=right_actions,
        left_rewards=left_rewards,
        right_rewards=right_rewards,
        infos=infos,
        left_total_reward=float(left_total_reward),
        right_total_reward=float(right_total_reward),
        winner=info.get("winner"),
        terminated=terminated,
        truncated=truncated,
    )

def animate_episode(episode: Episode, interval: int = 35, figsize=(8, 4.5)):
    """Create an HTML replay animation from a recorded episode."""
    if plt is None or FuncAnimation is None or HTML is None:
        raise ImportError("animate_episode requires matplotlib and IPython.display")
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    im = ax.imshow(episode.frames[0])
    title = ax.set_title("Paddle Duel replay")

    def update(i):
        im.set_data(episode.frames[i])
        info = episode.infos[i]
        if i == 0:
            title.set_text("Start")
        else:
            action = ACTION_NAMES.get(episode.actions[i - 1], str(episode.actions[i - 1]))
            reward = episode.rewards[i - 1]
            title.set_text(
                f"t={i:03d} | action={action} | reward={reward:+.3f} | "
                f"score={info.get('left_score', 0):.1f}-{info.get('right_score', 0):.1f} | "
                f"speed={info.get('ball_speed', 0):.1f} | reason={info.get('done_reason')}"
            )
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(episode.frames), interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_duel_episode(duel_episode, interval=30, figsize=(8, 4.5)):
    """Replay a two-agent DuelEpisode produced by record_duel_episode(...)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    im = ax.imshow(duel_episode.frames[0])
    title = ax.set_title("Paddle Duel replay")

    def update(i):
        im.set_data(duel_episode.frames[i])
        info = duel_episode.infos[i]

        if i == 0:
            title.set_text("Start")
        else:
            left_action = ACTION_NAMES.get(
                duel_episode.left_actions[i - 1],
                str(duel_episode.left_actions[i - 1])
            )
            right_action = ACTION_NAMES.get(
                duel_episode.right_actions[i - 1],
                str(duel_episode.right_actions[i - 1])
            )
            left_reward = duel_episode.left_rewards[i - 1]
            right_reward = duel_episode.right_rewards[i - 1]

            title.set_text(
                f"t={i:03d} | "
                f"L: {left_action}, r={left_reward:+.3f}, total={info.get('left_score', 0):+.3f} | "
                f"R: {right_action}, r={right_reward:+.3f}, total={info.get('right_score', 0):+.3f} | "
                f"winner={info.get('winner')}"
            )

        return [im, title]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(duel_episode.frames),
        interval=interval,
        blit=False,
    )
    plt.close(fig)
    return HTML(anim.to_jshtml())


def evaluate_agent(
    env_config: Dict[str, Any],
    agent_or_policy: Union[AgentProtocol, Callable],
    n_episodes: int = 50,
    seed: int = 0,
) -> Dict[str, Any]:
    """Evaluate one agent against the configured environment/opponent."""
    rewards = []
    wins = []
    score_diffs = []
    lengths = []
    left_scores = []
    right_scores = []
    reasons: Dict[str, int] = {}
    for i in range(n_episodes):
        env = PaddleDuelEnv(**env_config)
        total_reward, info = run_episode(env, agent_or_policy, seed=seed + i, training=False)
        rewards.append(total_reward)
        wins.append(float(info.get("winner") == "left"))
        score_diffs.append(float(info.get("score_diff", 0)))
        lengths.append(int(info.get("steps", 0)))
        left_scores.append(float(info.get("left_score", 0)))
        right_scores.append(float(info.get("right_score", 0)))
        reason = info.get("done_reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "win_rate": float(np.mean(wins)),
        "mean_score_diff": float(np.mean(score_diffs)),
        "mean_left_score": float(np.mean(left_scores)),
        "mean_right_score": float(np.mean(right_scores)),
        "mean_steps": float(np.mean(lengths)),
        "reasons": reasons,
    }


def evaluate_duel_agents(
    left_agent,
    right_agent,
    env_config=None,
    n_episodes=10,
    seed=0,
):
    env_config = env_config or {"level": 5, "max_seconds": 10, "render_scale": 2}

    results = []

    for i in range(n_episodes):
        result = play_rally(
            left_agent,
            right_agent,
            env_config=env_config,
            seed=seed + i,
            record=False,
        )
        results.append(result)

    left_scores = [r["left_score"] for r in results]
    right_scores = [r["right_score"] for r in results]
    steps = [r["steps"] for r in results]
    left_wins = [r["winner"] == "left" for r in results]
    right_wins = [r["winner"] == "right" for r in results]
    draws = [r["winner"] == "draw" for r in results]

    return {
        "n_episodes": n_episodes,
        "left_agent": getattr(left_agent, "name", "left_agent"),
        "right_agent": getattr(right_agent, "name", "right_agent"),
        "left_win_rate": float(np.mean(left_wins)),
        "right_win_rate": float(np.mean(right_wins)),
        "draw_rate": float(np.mean(draws)),
        "mean_left_score": float(np.mean(left_scores)),
        "mean_right_score": float(np.mean(right_scores)),
        "mean_score_diff_left_minus_right": float(np.mean(np.array(left_scores) - np.array(right_scores))),
        "mean_steps": float(np.mean(steps)),
        "raw_results": results,
    }


def play_rally(
    left_agent: AgentProtocol,
    right_agent: AgentProtocol,
    env_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    record: bool = False,
) -> Dict[str, Any]:
    """Pit two agent objects against each other for one rally.

    The left and right agents both receive side-adjusted observations. A right
    agent can therefore be written using the same logic as a left agent. The
    episode ends as soon as the first player scores, or after the time limit if
    nobody scores. On timeout, cumulative reward decides the winner.
    """
    env_config = dict(env_config or {})
    level = env_config.setdefault("level", 3)
    # Levels 4 and 5 are pixel-only. For levels 1-3 we default to tabular.
    if "obs_mode" not in env_config and int(level) not in (4, 5):
        env_config["obs_mode"] = "tabular"
    env = PaddleDuelEnv(**env_config)
    obs_left, obs_right, info = env.reset_duel(seed=seed)
    left_agent.reset()
    right_agent.reset()

    frames = [env.render()] if record else []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        left_action = int(left_agent.select_action(obs_left, info=info, training=False))
        # Right agent receives mirrored observation but the same global info. It
        # can use the observation only for fair competition if desired.
        right_action = int(right_agent.select_action(obs_right, info=_mirror_info_for_right(info), training=False))
        obs_left, obs_right, reward_left, reward_right, terminated, truncated, info = env.step_duel(left_action, right_action)
        if record:
            frames.append(env.render())

    result = {
        "left_agent": getattr(left_agent, "name", "left_agent"),
        "right_agent": getattr(right_agent, "name", "right_agent"),
        "left_score": info["left_score"],
        "right_score": info["right_score"],
        "left_points": info["left_points"],
        "right_points": info["right_points"],
        "left_total_reward": info["left_total_reward"],
        "right_total_reward": info["right_total_reward"],
        "winner": info["winner"],
        "steps": info["steps"],
        "done_reason": info["done_reason"],
        "info": info,
    }
    if record:
        result["frames"] = frames
    return result



# Backward-compatible alias: previous drafts called this helper play_match(...).
play_match = play_rally

def _mirror_info_for_right(info: Dict[str, Any]) -> Dict[str, Any]:
    """Create a side-adjusted info dictionary for a right-side agent."""
    mirrored = dict(info)
    mirrored["left_paddle_y"] = info["right_paddle_y"]
    mirrored["right_paddle_y"] = info["left_paddle_y"]
    mirrored["ball_x"] = -info["ball_x"]
    mirrored["ball_vx"] = -info["ball_vx"]
    mirrored["left_score"] = info["right_score"]
    mirrored["right_score"] = info["left_score"]
    mirrored["score_diff"] = -info["score_diff"]
    return mirrored


def rank_agents_against_benchmark(
    agents: List[AgentProtocol],
    benchmark_agents: Optional[List[AgentProtocol]] = None,
    env_config: Optional[Dict[str, Any]] = None,
    n_rallyes: int = 10,
    seed: int = 0,
):
    """Evaluate submitted agents against fixed benchmark opponents.

    Ranking key follows the proposed project rule:
    1. Higher cumulative reward score is better.
    2. If tied, higher win rate / Pong points is better.
    3. If still tied, fewer steps to reach the same result is better.
    """
    benchmark_agents = benchmark_agents or [TrackingAgent(side="right", noise=0.10, name="BenchmarkTracking")]
    rows = []
    for agent_idx, agent in enumerate(agents):
        total_score = 0.0
        total_points = 0.0
        total_steps = 0
        wins = 0
        for b_idx, benchmark in enumerate(benchmark_agents):
            for m in range(n_rallyes):
                rally_seed = seed + 1000 * agent_idx + 100 * b_idx + m
                result = play_rally(agent, benchmark, env_config=env_config, seed=rally_seed, record=False)
                total_score += float(result["left_score"])
                total_points += float(result.get("left_points", 0))
                total_steps += int(result["steps"])
                wins += int(result["winner"] == "left")
        n = max(1, n_rallyes * len(benchmark_agents))
        rows.append({
            "agent": getattr(agent, "name", f"agent_{agent_idx}"),
            "avg_score": total_score / n,
            "avg_points": total_points / n,
            "win_rate": wins / n,
            "avg_steps": total_steps / n,
        })
    rows = sorted(rows, key=lambda r: (r["avg_score"], r["win_rate"], r["avg_points"], -r["avg_steps"]), reverse=True)
    return rows


def moving_average(x: Sequence[float], window: int = 50) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    window = max(1, min(window, len(x)))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training_curves(history: Dict[str, Sequence[float]], window: int = 50, title: str = "Training curves"):
    """Plot reward and win/score curves from a history dictionary."""
    if plt is None:
        raise ImportError("plot_training_curves requires matplotlib")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rewards = np.asarray(history.get("rewards", []), dtype=float)
    wins = np.asarray(history.get("wins", []), dtype=float)
    scores = np.asarray(history.get("score_diffs", []), dtype=float)

    if len(rewards):
        axes[0].plot(rewards, alpha=0.25, label="episode reward")
        ma = moving_average(rewards, window)
        axes[0].plot(np.arange(len(ma)) + window - 1, ma, label=f"moving avg ({window})")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()

    if len(wins):
        ma = moving_average(wins, window)
        axes[1].plot(np.arange(len(ma)) + window - 1, ma, label=f"win rate avg ({window})")
    axes[1].set_title("Win rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    if len(scores):
        ma = moving_average(scores, window)
        axes[2].plot(scores, alpha=0.25, label="score diff")
        axes[2].plot(np.arange(len(ma)) + window - 1, ma, label=f"moving avg ({window})")
    axes[2].set_title("Score difference")
    axes[2].set_xlabel("Episode")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def show_frame(env: PaddleDuelEnv, figsize=(8, 4.5)):
    """Display the current environment frame in a notebook."""
    if plt is None:
        raise ImportError("show_frame requires matplotlib")
    plt.figure(figsize=figsize)
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# Pixel helpers
# ---------------------------------------------------------------------------


def rgb_to_grayscale_float(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB frame to a float32 grayscale image in [0, 1]."""
    frame = frame.astype(np.float32) / 255.0
    gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
    return gray.astype(np.float32)


def pixel_to_torch_tensor(frame: np.ndarray):
    """Convert an RGB frame to a PyTorch tensor with shape (1, H, W)."""
    import torch
    gray = rgb_to_grayscale_float(frame)
    return torch.from_numpy(gray).unsqueeze(0)


def extract_simple_pixel_features(frame: np.ndarray) -> np.ndarray:
    """Extract a small vector from a pixel frame.

    This helper is intentionally simple and transparent. It is useful for Level 4
    before students write a CNN: it locates bright/cyan/red/yellow regions and
    returns approximate object coordinates.
    """
    img = frame.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # Color masks. These are deliberately broad because frames may be downscaled.
    cyan_mask = (img[..., 1] > 0.65) & (img[..., 0] < 0.55)
    red_mask = (img[..., 0] > 0.65) & (img[..., 1] < 0.55)
    yellow_mask = (img[..., 0] > 0.65) & (img[..., 1] > 0.55) & (img[..., 2] < 0.65)

    def center(mask):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return np.array([0.5, 0.5], dtype=np.float32)
        return np.array([xs.mean() / max(1, w - 1), ys.mean() / max(1, h - 1)], dtype=np.float32)

    left = center(cyan_mask)
    right = center(red_mask)
    ball = center(yellow_mask)
    return np.concatenate([left, right, ball]).astype(np.float32)
