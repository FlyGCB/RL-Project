"""
Student agent template for Paddle Duel RL Arena.

Copy this file, rename the class if needed, and implement select_action(...).
Your agent must return one of:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_STAY = 2
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from paddle_duel_env import BaseAgent, ACTION_UP, ACTION_DOWN, ACTION_STAY


class Agent(BaseAgent):
    """Skeleton submitted agent.

    The competition notebook will call:
        agent.reset()
        action = agent.select_action(observation, info=info, training=False)

    Your job is to decide how the agent uses observation/info to choose actions.
    """

    def __init__(self, name: str = "StudentAgent", seed: Optional[int] = None):
        self.name = name
        self.rng = np.random.default_rng(seed)
        # Add your model/table/parameters here.
        # Example:
        # self.Q = {}

    def reset(self) -> None:
        """Called before each episode/match."""
        # Reset temporary episode variables here, if your agent needs them.
        pass

    def select_action(self, observation, info: Optional[Dict[str, Any]] = None, training: bool = False) -> int:
        """Return ACTION_UP, ACTION_DOWN, or ACTION_STAY.

        Parameters
        ----------
        observation:
            Tabular tuple for levels 1-3, or RGB image for levels 4-5.
        info:
            Extra diagnostic dictionary. During final competition, instructors may
            restrict which info fields are allowed, so your main logic should rely
            on observation where possible. In two-agent/self-play settings, a right-side
            agent receives a mirrored observation so the same action interface can be used.
        training:
            True during training, False during evaluation/competition.
        """
        # TODO: replace this with your trained policy.
        return ACTION_STAY
