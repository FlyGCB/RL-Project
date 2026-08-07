"""
Level 5 timing calibration.

Runs a handful of REAL two-agent (step_duel) episodes for both the
Feature-DQN and CNN-DQN architectures, with both sides doing genuine
forward passes (and, once the buffer is warm, real optimizer steps),
to measure actual steps/sec on this machine. This is meant to give an
honest, non-guessed extrapolation for how long 1500-episode 5A/5B runs
would take, rather than a hand-waved estimate.

Note: the dominant cost per step is env.render() (pure numpy, CPU-bound,
called twice per step_duel call), not the neural network forward/backward
pass. So steps/sec here is mostly a function of single-thread CPU speed,
not GPU tier - a faster GPU alone will not scale training time down much.
"""

from pathlib import Path
import sys
import time
import random
from collections import deque

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "paddle_duel_env.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if not (PROJECT_ROOT / "paddle_duel_env.py").exists():
    raise FileNotFoundError(
        "Could not find paddle_duel_env.py in the current directory or its "
        "parent. Run this script from the project root (the folder "
        "containing paddle_duel_env.py) or from notebooks/ or scripts/."
    )
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from paddle_duel_env import (
    PaddleDuelEnv,
    BaseAgent,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_STAY,
    extract_simple_pixel_features,
)

torch.set_num_threads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

N_ACTIONS = 3

# ---------------------------------------------------------------------
# Feature-DQN (mirrors Level_4_Feature_DQN_Baseline.ipynb)
# ---------------------------------------------------------------------

FEATURE_DIM = 12


def build_feature_state(frame, previous_coordinates=None):
    current = extract_simple_pixel_features(frame).astype(np.float32)
    if previous_coordinates is None:
        diff = np.zeros_like(current, dtype=np.float32)
    else:
        diff = (current - previous_coordinates).astype(np.float32)
    return np.concatenate([current, diff]).astype(np.float32), current


class FeatureQNetwork(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, n_actions=N_ACTIONS):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------
# CNN-DQN (mirrors Level_4_CNN_DQN_Pixel_Baseline.ipynb)
# ---------------------------------------------------------------------

FRAME_HEIGHT = 84
FRAME_WIDTH = 84
FRAME_STACK = 4


def preprocess_frame(frame):
    frame_tensor = torch.as_tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    r, g, b = frame_tensor[:, 0:1], frame_tensor[:, 1:2], frame_tensor[:, 2:3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    resized = F.interpolate(gray, size=(FRAME_HEIGHT, FRAME_WIDTH), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).clamp(0, 255).byte().cpu().numpy()


class FrameStacker:
    def __init__(self, stack_size=FRAME_STACK):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, first_frame):
        processed = preprocess_frame(first_frame)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(processed.copy())
        return self.get_state()

    def append(self, frame):
        self.frames.append(preprocess_frame(frame))
        return self.get_state()

    def get_state(self):
        return np.stack(self.frames, axis=0).astype(np.uint8)


class CNNQNetwork(nn.Module):
    def __init__(self, input_channels=FRAME_STACK, n_actions=N_ACTIONS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, FRAME_HEIGHT, FRAME_WIDTH)
            flat = int(np.prod(self.features(dummy).shape[1:]))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------------
# Minimal replay buffer + DQN step logic shared by both architectures
# ---------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = self.rng.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.stack(s), np.array(a), np.array(r, dtype=np.float32), np.stack(ns), np.array(d, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


def run_calibration(name, is_pixel_cnn, n_episodes=15, max_seconds=15, batch_size=64,
                     min_buffer_size=200, train_frequency=4):
    print(f"\n===== {name} =====")

    if is_pixel_cnn:
        policy = CNNQNetwork().to(DEVICE)
        target = CNNQNetwork().to(DEVICE)
        buffer = ReplayBuffer(capacity=20_000)
        stacker_left = FrameStacker()
        stacker_right = FrameStacker()
        scale = 255.0
    else:
        policy = FeatureQNetwork().to(DEVICE)
        target = FeatureQNetwork().to(DEVICE)
        buffer = ReplayBuffer(capacity=20_000)
        scale = 1.0

    target.load_state_dict(policy.state_dict())
    target.eval()
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    def act(state):
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0) / scale
            q = policy(t)
        return int(torch.argmax(q, dim=1).item())

    def optimize(step_idx):
        if len(buffer) < min_buffer_size or step_idx % train_frequency != 0:
            return
        s, a, r, ns, d = buffer.sample(batch_size)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE) / scale
        ns_t = torch.as_tensor(ns, dtype=torch.float32, device=DEVICE) / scale
        a_t = torch.as_tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        d_t = torch.as_tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        q_sa = policy(s_t).gather(1, a_t)
        with torch.no_grad():
            next_q = target(ns_t).max(dim=1, keepdim=True).values
            y = r_t + 0.99 * (1.0 - d_t) * next_q
        loss = F.smooth_l1_loss(q_sa, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        if step_idx % 1000 == 0:
            target.load_state_dict(policy.state_dict())

    total_steps = 0
    start = time.perf_counter()

    for ep in range(n_episodes):
        env = PaddleDuelEnv(level=5, max_seconds=max_seconds, render_scale=2)
        obs_left, obs_right, info = env.reset_duel(seed=1000 + ep)

        if is_pixel_cnn:
            state_left = stacker_left.reset(obs_left)
            state_right = stacker_right.reset(obs_right)
        else:
            state_left, prev_left = build_feature_state(obs_left)
            state_right, prev_right = build_feature_state(obs_right)

        terminated = truncated = False
        while not (terminated or truncated):
            left_action = act(state_left)
            right_action = act(state_right)  # mirrors self-play: same net, both sides

            obs_left, obs_right, r_left, r_right, terminated, truncated, info = env.step_duel(left_action, right_action)
            done = terminated or truncated

            if is_pixel_cnn:
                next_state_left = stacker_left.append(obs_left)
            else:
                next_state_left, prev_left = build_feature_state(obs_left, prev_left)

            buffer.push(state_left, left_action, r_left, next_state_left, done)

            total_steps += 1
            optimize(total_steps)

            state_left = next_state_left
            if is_pixel_cnn:
                state_right = stacker_right.append(obs_right)
            else:
                state_right, prev_right = build_feature_state(obs_right, prev_right)

    elapsed = time.perf_counter() - start
    steps_per_sec = total_steps / elapsed
    avg_steps_per_ep = total_steps / n_episodes

    print(f"Episodes: {n_episodes} | Total steps: {total_steps} | Wall time: {elapsed:.1f}s")
    print(f"Steps/sec: {steps_per_sec:.1f} | Avg steps/episode: {avg_steps_per_ep:.1f}")

    for target_episodes in (1500, 2000):
        est_seconds = target_episodes * avg_steps_per_ep / steps_per_sec
        print(f"  -> Estimated time for {target_episodes} episodes: {est_seconds/60:.1f} min ({est_seconds/3600:.2f} h)")

    return steps_per_sec, avg_steps_per_ep


if __name__ == "__main__":
    run_calibration("Feature-DQN (self-play, mirrored net)", is_pixel_cnn=False, n_episodes=15)
    run_calibration("CNN-DQN (self-play, mirrored net)", is_pixel_cnn=True, n_episodes=15)
