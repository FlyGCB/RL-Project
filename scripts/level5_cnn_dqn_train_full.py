from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from collections import Counter, deque
import random
import time

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "paddle_duel_env.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless training run, no display available
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from paddle_duel_env import (
    PaddleDuelEnv,
    BaseAgent,
    RandomAgent,
    TrackingAgent,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_STAY,
    ACTION_NAMES,
    record_duel_episode,
    animate_duel_episode,
    evaluate_duel_agents,
    play_rally,
)

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


FRAME_HEIGHT = 84
FRAME_WIDTH = 84
FRAME_STACK = 4
N_ACTIONS = 3


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")

    frame_tensor = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    red = frame_tensor[:, 0:1]
    green = frame_tensor[:, 1:2]
    blue = frame_tensor[:, 2:3]
    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue

    resized = F.interpolate(grayscale, size=(FRAME_HEIGHT, FRAME_WIDTH), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).clamp(0, 255).byte().cpu().numpy()


class FrameStacker:
    def __init__(self, stack_size: int = FRAME_STACK):
        self.stack_size = int(stack_size)
        self.frames = deque(maxlen=self.stack_size)

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        processed = preprocess_frame(first_frame)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(processed.copy())
        return self.get_state()

    def append(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess_frame(frame))
        return self.get_state()

    def get_state(self) -> np.ndarray:
        if len(self.frames) != self.stack_size:
            raise RuntimeError("Frame stack is not initialized.")
        return np.stack(self.frames, axis=0).astype(np.uint8)


class PixelReplayBuffer:
    def __init__(self, capacity: int = 30_000, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.buffer.append((
            np.asarray(state, dtype=np.uint8),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.uint8),
            bool(done),
        ))

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError("Cannot sample more transitions than are stored.")
        transitions = self.rng.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class CNNQNetwork(nn.Module):
    def __init__(self, input_channels: int = FRAME_STACK, n_actions: int = N_ACTIONS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, FRAME_HEIGHT, FRAME_WIDTH)
            flattened_size = int(np.prod(self.features(dummy).shape[1:]))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class CNNDQNAgent(BaseAgent):
    def __init__(
        self,
        input_channels: int = FRAME_STACK,
        n_actions: int = N_ACTIONS,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.99995,
        buffer_capacity: int = 30_000,
        batch_size: int = 64,
        min_buffer_size: int = 5_000,
        target_update_steps: int = 2_000,
        train_frequency: int = 4,
        name: str = "CNN-DQN",
        seed: int = 0,
        device=DEVICE,
    ):
        self.name = name
        self.input_channels = int(input_channels)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)

        self.batch_size = int(batch_size)
        self.min_buffer_size = int(min_buffer_size)
        self.target_update_steps = int(target_update_steps)
        self.train_frequency = int(train_frequency)

        self.device = device
        self.rng = np.random.default_rng(seed)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.policy_net = CNNQNetwork(input_channels=input_channels, n_actions=n_actions).to(self.device)
        self.target_net = CNNQNetwork(input_channels=input_channels, n_actions=n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = PixelReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.frame_stacker = FrameStacker(stack_size=input_channels)

        self.environment_steps = 0
        self.optimization_steps = 0
        self.current_state = None

    def reset(self) -> None:
        self.frame_stacker = FrameStacker(stack_size=self.input_channels)
        self.current_state = None

    def reset_exploration(self) -> None:
        self.epsilon = self.epsilon_start

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def initialize_state(self, observation) -> np.ndarray:
        self.current_state = self.frame_stacker.reset(observation)
        return self.current_state

    def append_observation(self, observation) -> np.ndarray:
        self.current_state = self.frame_stacker.append(observation)
        return self.current_state

    def select_action_from_state(self, state, training: bool = False, force_random: bool = False) -> int:
        if force_random:
            return int(self.rng.integers(self.n_actions))
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        if training:
            self.policy_net.train()
        return int(torch.argmax(q_values, dim=1).item())

    def select_action(self, observation, info: Optional[Dict[str, Any]] = None, training: bool = False) -> int:
        if self.current_state is None:
            state = self.initialize_state(observation)
        else:
            state = self.append_observation(observation)
        return self.select_action_from_state(state, training=training)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self) -> Optional[float]:
        if len(self.replay_buffer) < self.min_buffer_size:
            return None
        if self.environment_steps % self.train_frequency != 0:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device) / 255.0
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device) / 255.0
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor).max(dim=1, keepdim=True).values
            target_q_values = rewards_tensor + self.gamma * (1.0 - dones_tensor) * next_q_values

        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.optimization_steps += 1

        if self.environment_steps % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path) -> None:
        checkpoint = {
            "name": self.name,
            "input_channels": self.input_channels,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "epsilon": self.epsilon,
            "batch_size": self.batch_size,
            "min_buffer_size": self.min_buffer_size,
            "target_update_steps": self.target_update_steps,
            "train_frequency": self.train_frequency,
            "environment_steps": self.environment_steps,
            "optimization_steps": self.optimization_steps,
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, seed: int = 0, device=DEVICE):
        checkpoint = torch.load(path, map_location=device)
        agent = cls(
            input_channels=checkpoint["input_channels"],
            n_actions=checkpoint["n_actions"],
            gamma=checkpoint["gamma"],
            epsilon_start=checkpoint["epsilon_start"],
            epsilon_min=checkpoint["epsilon_min"],
            epsilon_decay=checkpoint["epsilon_decay"],
            batch_size=checkpoint["batch_size"],
            min_buffer_size=checkpoint["min_buffer_size"],
            target_update_steps=checkpoint["target_update_steps"],
            train_frequency=checkpoint["train_frequency"],
            name=checkpoint["name"],
            seed=seed,
            device=device,
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = checkpoint["epsilon"]
        agent.environment_steps = checkpoint["environment_steps"]
        agent.optimization_steps = checkpoint["optimization_steps"]
        return agent


class OpponentPool:
    def __init__(self, network_factory, capacity: int = 8, seed: int = 0):
        self.network_factory = network_factory
        self.capacity = int(capacity)
        self.snapshots: List[Dict[str, torch.Tensor]] = []
        self.rng = random.Random(seed)

    def seed_with(self, state_dict) -> None:
        self.snapshots.append({k: v.detach().cpu().clone() for k, v in state_dict.items()})

    def add_snapshot(self, state_dict) -> None:
        cloned = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        if len(self.snapshots) >= self.capacity and len(self.snapshots) > 1:
            self.snapshots.pop(1)  # evict oldest non-pinned snapshot; index 0 stays.
        self.snapshots.append(cloned)

    def sample_network(self, device):
        state_dict = self.rng.choice(self.snapshots)
        net = self.network_factory().to(device)
        net.load_state_dict(state_dict)
        net.eval()
        return net

    def __len__(self) -> int:
        return len(self.snapshots)


def select_greedy_action(net, state, device, scale: float = 255.0) -> int:
    '''Greedy action from any frozen/opponent network (no exploration).'''
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0) / scale
        q_values = net(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def make_self_play_opponent_provider(agent: "CNNDQNAgent", opponent_pool: OpponentPool, p_live: float = 0.8, seed: int = 0):
    '''Returns opponent_provider(episode_idx) -> (network, is_live).

    With probability p_live, the right paddle mirrors the agent's current
    live weights (true self-play). Otherwise, a frozen snapshot is sampled
    from the opponent pool.
    '''
    rng = random.Random(seed)

    def provider(episode_idx: int):
        if len(opponent_pool) == 0 or rng.random() < p_live:
            return agent.policy_net, True
        return opponent_pool.sample_network(agent.device), False

    return provider


FROZEN_L4_PATH = PROJECT_ROOT / "saved_agents" / "cnn_dqn_level4.pt"

frozen_opponent = CNNDQNAgent.load(FROZEN_L4_PATH, seed=999, device=DEVICE)
frozen_opponent.policy_net.eval()
print("Loaded frozen Level 4 opponent:", frozen_opponent.name)
print("Environment steps at save time:", frozen_opponent.environment_steps)


def train_duel_cnn_dqn(
    agent: CNNDQNAgent,
    opponent_provider,
    n_episodes: int,
    seed: int = 0,
    max_seconds: float = 15.0,
    render_scale: int = 2,
    forced_random_episodes: int = 0,
    print_every: int = 100,
    opponent_pool: Optional[OpponentPool] = None,
    snapshot_every: Optional[int] = None,
):
    history = {
        "rewards": [], "wins": [], "hits": [], "steps": [],
        "movement_actions": [], "stay_actions": [], "losses": [],
        "epsilon": [], "opponent_is_live": [],
    }

    agent.reset_exploration()
    start_time = time.perf_counter()

    for episode_idx in range(n_episodes):
        env = PaddleDuelEnv(level=5, max_seconds=max_seconds, render_scale=render_scale)
        obs_left, obs_right, info = env.reset_duel(seed=seed + episode_idx)
        agent.reset()

        right_net, is_live = opponent_provider(episode_idx)

        stacker_left = FrameStacker(stack_size=agent.input_channels)
        stacker_right = FrameStacker(stack_size=agent.input_channels)
        state_left = stacker_left.reset(obs_left)
        state_right = stacker_right.reset(obs_right)

        terminated = truncated = False
        total_reward = 0.0
        episode_losses = []
        movement_actions = 0
        stay_actions = 0
        steps = 0
        force_random = episode_idx < forced_random_episodes

        while not (terminated or truncated):
            left_action = agent.select_action_from_state(state_left, training=True, force_random=force_random)
            right_action = select_greedy_action(right_net, state_right, agent.device, scale=255.0)

            if left_action in (ACTION_UP, ACTION_DOWN):
                movement_actions += 1
            else:
                stay_actions += 1

            obs_left, obs_right, reward_left, reward_right, terminated, truncated, info = env.step_duel(left_action, right_action)
            done = terminated or truncated

            next_state_left = stacker_left.append(obs_left)
            state_right = stacker_right.append(obs_right)

            agent.store_transition(state_left, left_action, reward_left, next_state_left, done)

            agent.environment_steps += 1
            loss = agent.optimize()
            if loss is not None:
                episode_losses.append(loss)

            state_left = next_state_left
            total_reward += float(reward_left)
            steps += 1

            if not force_random:
                agent.decay_exploration()

        history["rewards"].append(total_reward)
        history["wins"].append(float(info.get("winner") == "left"))
        history["hits"].append(int(info.get("left_hits", 0)))
        history["steps"].append(steps)
        history["movement_actions"].append(movement_actions)
        history["stay_actions"].append(stay_actions)
        history["losses"].append(float(np.mean(episode_losses)) if episode_losses else np.nan)
        history["epsilon"].append(agent.epsilon)
        history["opponent_is_live"].append(bool(is_live))

        if opponent_pool is not None and snapshot_every and (episode_idx + 1) % snapshot_every == 0:
            opponent_pool.add_snapshot(agent.policy_net.state_dict())

        if (episode_idx + 1) % print_every == 0 or episode_idx == 0:
            start_index = max(0, episode_idx - print_every + 1)
            recent_reward = np.mean(history["rewards"][start_index:episode_idx + 1])
            recent_win_rate = np.mean(history["wins"][start_index:episode_idx + 1])
            print(
                f"Episode {episode_idx + 1:4d}/{n_episodes} | reward={recent_reward:7.3f} | "
                f"win_rate={recent_win_rate:6.3f} | epsilon={agent.epsilon:6.3f} | "
                f"opponent={'live' if is_live else 'pool'} | buffer={len(agent.replay_buffer):6d}"
            )

    history["training_seconds"] = time.perf_counter() - start_time
    return history


def moving_average(values, window=100):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def plot_history(history, metric, ylabel, window=100, ignore_nan=False, title_prefix=""):
    values = np.asarray(history[metric], dtype=float)
    if ignore_nan:
        values = values[np.isfinite(values)]
    if len(values) == 0:
        print(f"No finite values available for {metric}.")
        return
    smoothed = moving_average(values, window)
    start = window - 1 if len(values) >= window else 0
    x = np.arange(start, start + len(smoothed))
    plt.figure(figsize=(10, 5))
    plt.plot(x, smoothed)
    plt.xlabel("Training episode")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix}: {ylabel}")
    plt.grid(alpha=0.25)
    plt.show()


CNN_DQN_CONFIG = {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.99995,
    "buffer_capacity": 30_000,
    "batch_size": 64,
    "min_buffer_size": 5_000,
    "target_update_steps": 2_000,
    "train_frequency": 4,
}

N_EPISODES_5A = 1500
FORCED_RANDOM_EPISODES_5A = 200
TRAIN_SEED = 0

agent_5a = CNNDQNAgent(name="Level 5A CNN-DQN (vs frozen L4)", seed=15, **CNN_DQN_CONFIG)

opponent_provider_5a = lambda episode_idx: (frozen_opponent.policy_net, False)

history_5a = train_duel_cnn_dqn(
    agent_5a,
    opponent_provider_5a,
    n_episodes=N_EPISODES_5A,
    seed=TRAIN_SEED,
    forced_random_episodes=FORCED_RANDOM_EPISODES_5A,
)

print(
    "\n5A training complete."
    f"\nTime: {history_5a['training_seconds']:.1f} seconds"
    f"\nEnvironment steps: {agent_5a.environment_steps:,}"
    f"\nOptimization steps: {agent_5a.optimization_steps:,}"
)


plot_history(history_5a, "rewards", "Moving-average reward", title_prefix="Level 5A CNN-DQN")
plot_history(history_5a, "wins", "Moving-average win rate", title_prefix="Level 5A CNN-DQN")
plot_history(history_5a, "hits", "Moving-average paddle hits", title_prefix="Level 5A CNN-DQN")
plot_history(history_5a, "steps", "Moving-average episode length", title_prefix="Level 5A CNN-DQN")
plot_history(history_5a, "losses", "Moving-average Huber loss", ignore_nan=True, title_prefix="Level 5A CNN-DQN")
plot_history(history_5a, "epsilon", "Epsilon", window=1, title_prefix="Level 5A CNN-DQN")


EVAL_ENV_CONFIG = {"level": 5, "max_seconds": 30, "render_scale": 2}
N_EVAL_EPISODES = 50
EVAL_SEED = 10_000

def evaluate_5x(agent, label):
    results = {}

    vs_frozen = evaluate_duel_agents(
        left_agent=agent, right_agent=frozen_opponent,
        env_config=EVAL_ENV_CONFIG, n_episodes=N_EVAL_EPISODES, seed=EVAL_SEED,
    )
    results["vs frozen L4 (agent=left)"] = vs_frozen

    vs_random = evaluate_duel_agents(
        left_agent=agent, right_agent=RandomAgent(side="right", seed=101),
        env_config=EVAL_ENV_CONFIG, n_episodes=N_EVAL_EPISODES, seed=EVAL_SEED,
    )
    results["vs Random (agent=left)"] = vs_random

    vs_tracking = evaluate_duel_agents(
        left_agent=TrackingAgent(side="left", noise=0.10, seed=202), right_agent=agent,
        env_config=EVAL_ENV_CONFIG, n_episodes=N_EVAL_EPISODES, seed=EVAL_SEED,
    )
    results["vs Tracking (agent=right)"] = vs_tracking

    print(f"\n===== {label} =====")
    for name, result in results.items():
        print(f"\n{name}")
        for key in ["left_win_rate", "right_win_rate", "draw_rate", "mean_left_score", "mean_right_score", "mean_steps"]:
            print(f"  {key}: {result[key]:.4f}")

    return results


eval_5a = evaluate_5x(agent_5a, "Level 5A CNN-DQN")


SAVE_DIR = PROJECT_ROOT / "saved_agents"
SAVE_DIR.mkdir(exist_ok=True)

SAVE_PATH_5A = SAVE_DIR / "cnn_dqn_level5_5a.pt"
agent_5a.save(SAVE_PATH_5A)
print("Saved 5A agent to:", SAVE_PATH_5A)


OPPONENT_POOL_CAPACITY = 8
P_LIVE = 0.8
SNAPSHOT_EVERY = 200
N_EPISODES_5B = 1500

agent_5b = CNNDQNAgent.load(SAVE_PATH_5A, seed=16, device=DEVICE)
agent_5b.name = "Level 5B CNN-DQN (self-play)"

# Warm start: keep the weights learned in 5A. Re-open exploration modestly
# rather than resetting to epsilon=1.0, since the existing policy is not
# garbage, only the opponent distribution is about to change.
agent_5b.epsilon_start = 0.3
agent_5b.epsilon_min = 0.05
agent_5b.epsilon_decay = 0.99985
agent_5b.epsilon = agent_5b.epsilon_start

opponent_pool = OpponentPool(
    network_factory=lambda: CNNQNetwork(FRAME_STACK, N_ACTIONS),
    capacity=OPPONENT_POOL_CAPACITY,
    seed=43,
)
opponent_pool.seed_with(frozen_opponent.policy_net.state_dict())

opponent_provider_5b = make_self_play_opponent_provider(agent_5b, opponent_pool, p_live=P_LIVE, seed=17)

history_5b = train_duel_cnn_dqn(
    agent_5b,
    opponent_provider_5b,
    n_episodes=N_EPISODES_5B,
    seed=TRAIN_SEED + 10_000,
    forced_random_episodes=0,
    opponent_pool=opponent_pool,
    snapshot_every=SNAPSHOT_EVERY,
)

print(
    "\n5B training complete."
    f"\nTime: {history_5b['training_seconds']:.1f} seconds"
    f"\nEnvironment steps: {agent_5b.environment_steps:,}"
    f"\nOptimization steps: {agent_5b.optimization_steps:,}"
    f"\nFinal opponent pool size: {len(opponent_pool)}"
)


plot_history(history_5b, "rewards", "Moving-average reward", title_prefix="Level 5B CNN-DQN self-play")
plot_history(history_5b, "wins", "Moving-average win rate", title_prefix="Level 5B CNN-DQN self-play")
plot_history(history_5b, "hits", "Moving-average paddle hits", title_prefix="Level 5B CNN-DQN self-play")
plot_history(history_5b, "steps", "Moving-average episode length", title_prefix="Level 5B CNN-DQN self-play")
plot_history(history_5b, "losses", "Moving-average Huber loss", ignore_nan=True, title_prefix="Level 5B CNN-DQN self-play")
plot_history(history_5b, "epsilon", "Epsilon", window=1, title_prefix="Level 5B CNN-DQN self-play")

live_fraction = np.mean(history_5b["opponent_is_live"])
print(f"Fraction of episodes played against the live mirror: {live_fraction:.3f} (target ~{P_LIVE})")


eval_5b = evaluate_5x(agent_5b, "Level 5B CNN-DQN (self-play)")

h2h_5b_vs_5a = evaluate_duel_agents(
    left_agent=agent_5b, right_agent=agent_5a,
    env_config=EVAL_ENV_CONFIG, n_episodes=N_EVAL_EPISODES, seed=EVAL_SEED,
)
print("\n5B (left) vs 5A (right) head-to-head:")
for key in ["left_win_rate", "right_win_rate", "draw_rate", "mean_left_score", "mean_right_score"]:
    print(f"  {key}: {h2h_5b_vs_5a[key]:.4f}")

self_mirror = evaluate_duel_agents(
    left_agent=agent_5b, right_agent=agent_5b,
    env_config=EVAL_ENV_CONFIG, n_episodes=N_EVAL_EPISODES, seed=EVAL_SEED,
)
print("\n5B vs itself (mirror match, sanity check for degenerate draws):")
for key in ["left_win_rate", "right_win_rate", "draw_rate", "mean_left_score", "mean_right_score"]:
    print(f"  {key}: {self_mirror[key]:.4f}")


def compact_comparison_table(eval_a, eval_b, label_a="5A", label_b="5B"):
    columns = ["opponent", f"{label_a} win_rate", f"{label_b} win_rate", f"{label_a} mean_score", f"{label_b} mean_score"]
    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for opponent in eval_a.keys():
        result_a = eval_a[opponent]
        result_b = eval_b[opponent]
        agent_is_left = "agent=left" in opponent
        win_key = "left_win_rate" if agent_is_left else "right_win_rate"
        score_key = "mean_left_score" if agent_is_left else "mean_right_score"
        row = [
            opponent,
            f"{result_a[win_key]:.4f}",
            f"{result_b[win_key]:.4f}",
            f"{result_a[score_key]:.4f}",
            f"{result_b[score_key]:.4f}",
        ]
        print(" | ".join(row))


compact_comparison_table(eval_5a, eval_5b)


SAVE_PATH_5B = SAVE_DIR / "cnn_dqn_level5_selfplay.pt"
agent_5b.save(SAVE_PATH_5B)
print("Saved final Level 5 CNN-DQN agent to:", SAVE_PATH_5B)

reloaded_5b = CNNDQNAgent.load(SAVE_PATH_5B, seed=999, device=DEVICE)
print(
    "Reloaded:", reloaded_5b.name,
    "| environment steps:", reloaded_5b.environment_steps,
    "| optimization steps:", reloaded_5b.optimization_steps,
)


reloaded_eval = evaluate_duel_agents(
    left_agent=reloaded_5b, right_agent=frozen_opponent,
    env_config=EVAL_ENV_CONFIG, n_episodes=20, seed=20_000,
)
print("Reloaded 5B agent vs frozen L4:")
for key in ["left_win_rate", "right_win_rate", "draw_rate", "mean_left_score", "mean_right_score"]:
    print(f"  {key}: {reloaded_eval[key]:.4f}")


REPLAY_SEED = 22
REPLAY_SECONDS = 15

duel_episode = record_duel_episode(
    PaddleDuelEnv(level=5, max_seconds=REPLAY_SECONDS, render_scale=2),
    left_agent=agent_5b,
    right_agent=frozen_opponent,
    seed=REPLAY_SEED,
)

left_action_counts = Counter(duel_episode.left_actions)
right_action_counts = Counter(duel_episode.right_actions)

print("===== 5B (left) vs frozen L4 (right) replay =====")
print("LEFT (5B) action counts:", {ACTION_NAMES[k]: v for k, v in left_action_counts.items()})
print("RIGHT (frozen L4) action counts:", {ACTION_NAMES[k]: v for k, v in right_action_counts.items()})
print("Left total reward:", duel_episode.left_total_reward)
print("Right total reward:", duel_episode.right_total_reward)
print("Winner:", duel_episode.winner)


anim = animate_duel_episode(duel_episode, interval=30)
print("\nFULL TRAINING RUN COMPLETE")

