
"""Optional realtime WASD demo for Paddle Duel.

Run from the project root:
    python extras/play_realtime_wasd.py

Requires pygame:
    pip install pygame
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pygame
from paddle_duel_env import PaddleDuelEnv, ACTION_UP, ACTION_DOWN, ACTION_STAY

pygame.init()
env = PaddleDuelEnv(level=2, max_seconds=60)
obs, info = env.reset(seed=0)
screen = pygame.display.set_mode((env.width_px, env.height_px))
pygame.display.set_caption("Paddle Duel RL Arena - WASD demo")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
big_font = pygame.font.SysFont(None, 32)
terminated = truncated = False
started = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); raise SystemExit
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            obs, info = env.reset(); terminated = truncated = False; started = False

    keys = pygame.key.get_pressed()
    moving_up = keys[pygame.K_w] or keys[pygame.K_UP]
    moving_down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    if moving_up:
        action = ACTION_UP
    elif moving_down:
        action = ACTION_DOWN
    else:
        action = ACTION_STAY

    # Pause at the beginning until the human enters the first movement.
    if not started and action in (ACTION_UP, ACTION_DOWN):
        started = True

    if started and not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surface, (0, 0))

    # Text score overlay in the top bar. The render also draws numeric scores,
    # but text is convenient during live play and stays above the play area.
    left_text = font.render(f"Left reward: {info['left_score']:.1f}  round: {info['left_points']}", True, (238, 242, 255))
    right_text = font.render(f"Right reward: {info['right_score']:.1f}  round: {info['right_points']}", True, (238, 242, 255))
    screen.blit(left_text, (74, 10))
    screen.blit(right_text, (env.width_px - right_text.get_width() - 74, 10))

    if not started:
        msg = big_font.render("Press W/S or arrow keys to start", True, (255, 255, 255))
        screen.blit(msg, ((env.width_px - msg.get_width()) // 2, env.height_px // 2 - 20))
    if terminated or truncated:
        text = font.render(f"Game over: {info['done_reason']} | winner={info['winner']} | reward={info['left_score']:.1f}-{info['right_score']:.1f} | Press R", True, (255, 255, 255))
        screen.blit(text, (20, env.height_px - 28))

    pygame.display.flip()
    clock.tick(env.fps)
