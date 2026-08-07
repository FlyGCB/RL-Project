"""Backfill real training-run outputs into the Level 5 notebooks' cell outputs.

Source of truth for every value written here:
- logs/feature_dqn_FULL_TRAIN_log.txt  (verbatim console text)
- logs/cnn_dqn_FULL_TRAIN_log.txt       (verbatim console text)
- logs/replay_regen.json                (real re-inference on the saved
  final checkpoints: exact same seed=22 replay, same numbers as the original run)

Nothing here is fabricated. Where the periodic console log did not capture a metric
(hits / steps / losses moving averages -- only reward, win_rate, epsilon were printed
every 100 episodes), the corresponding plot cell is left without an image and a
stream note explains why, instead of inventing numbers.
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf

PROJECT_ROOT = Path(__file__).parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

replay_data = json.loads((PROJECT_ROOT / "logs" / "replay_regen.json").read_text(encoding="utf-8"))

EPISODES = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

DISCLOSURE_MD = """\
> **Note on this notebook's saved outputs.** The full-scale run (1500/1500 episodes for both 5A and 5B) was executed as a converted `.py` script on the training machine so it could run for several hours in the background, rather than inside this Jupyter kernel directly. The cell outputs below were backfilled from that real run so the notebook is self-documenting:
>
> - All printed text (training summaries, evaluation tables, comparison table, checkpoint save/reload messages) is copied **verbatim** from the actual run's console log.
> - The **reward / win-rate / epsilon** training-curve plots are reconstructed from the same run's periodic console summary (printed every 100 episodes: episode 1, 100, 200, ..., 1500 -- 16 points), not the full 1500-point moving average `plot_history` would show under a live kernel run. The shape/trend is real; the resolution is coarser.
> - The **hits / episode-length / loss** plots could not be reconstructed -- those three metrics were tracked internally during training but were never part of the periodic console print, so no data survives outside the finished process. Their cells note this instead of showing an image.
> - The final **replay + animation** cells were re-executed for real in this pass (fast: pure inference on the already-trained saved checkpoints, seed=22, no training) -- those outputs are live, not backfilled, and match the original run's numbers exactly (deterministic greedy policy).
>
> Source logs: `logs/feature_dqn_FULL_TRAIN_log.txt`, `logs/cnn_dqn_FULL_TRAIN_log.txt`.
"""


def fig_to_png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def make_plot_output(values, ylabel, title, ymax=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(EPISODES, values, marker="o")
    ax.set_xlabel("Training episode (16-pt periodic-log reconstruction, not full moving average)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    b64 = fig_to_png_b64(fig)
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": ["<Figure size 1000x500 with 1 Axes>"]},
        "metadata": {},
    }


def stream(text):
    return {"output_type": "stream", "name": "stdout", "text": text}


def no_data_note(metric):
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": (
            f"[Backfill note] No reconstructable data for '{metric}': this metric was tracked during "
            f"training but was not part of the periodic console summary (only reward/win_rate/epsilon "
            f"were printed every 100 episodes), and the in-memory history was not serialized before the "
            f"process exited. Re-run this cell live to regenerate it.\n"
        ),
    }


def html_result(html, execution_count):
    return {
        "output_type": "execute_result",
        "data": {"text/html": html, "text/plain": ["<IPython.core.display.HTML object>"]},
        "metadata": {},
        "execution_count": execution_count,
    }


def build_feature_outputs():
    a_reward = [10.000,1.658,2.258,2.258,1.850,2.744,2.422,2.433,2.413,3.236,2.528,4.221,3.240,4.338,4.037,3.329]
    a_win =    [1.000,0.170,0.230,0.230,0.190,0.280,0.250,0.250,0.250,0.330,0.260,0.430,0.330,0.440,0.410,0.340]
    a_eps =    [1.000,1.000,1.000,0.306,0.101,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050]
    b_reward = [9.987,3.642,6.327,6.433,6.330,8.042,7.434,6.540,6.250,4.771,5.564,6.162,4.972,4.075,2.676,2.973]
    b_win =    [1.000,0.370,0.670,0.650,0.640,0.810,0.750,0.710,0.670,0.500,0.570,0.640,0.500,0.410,0.270,0.300]
    b_eps =    [0.292,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050]

    out = {}
    out["imports"] = [stream("PyTorch version: 2.5.1+cu121\nDevice: cuda\nGPU: NVIDIA GeForce RTX 3070 Ti\n")]
    out["feature_dim"] = [stream("Feature dimension: 12\n")]
    out["load_frozen"] = [stream("Loaded frozen Level 4 opponent: Level 4 Feature DQN\nEnvironment steps at save time: 305564\n")]
    out["train_5a"] = [stream(
        "Episode    1/1500 | reward= 10.000 | win_rate= 1.000 | epsilon= 1.000 | opponent=pool | buffer=    87\n"
        "Episode  100/1500 | reward=  1.658 | win_rate= 0.170 | epsilon= 1.000 | opponent=pool | buffer= 11443\n"
        "Episode  200/1500 | reward=  2.258 | win_rate= 0.230 | epsilon= 1.000 | opponent=pool | buffer= 23799\n"
        "Episode  300/1500 | reward=  2.258 | win_rate= 0.230 | epsilon= 0.306 | opponent=pool | buffer= 35643\n"
        "Episode  400/1500 | reward=  1.850 | win_rate= 0.190 | epsilon= 0.101 | opponent=pool | buffer= 46754\n"
        "Episode  500/1500 | reward=  2.744 | win_rate= 0.280 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode  600/1500 | reward=  2.422 | win_rate= 0.250 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode  700/1500 | reward=  2.433 | win_rate= 0.250 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode  800/1500 | reward=  2.413 | win_rate= 0.250 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode  900/1500 | reward=  3.236 | win_rate= 0.330 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1000/1500 | reward=  2.528 | win_rate= 0.260 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1100/1500 | reward=  4.221 | win_rate= 0.430 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1200/1500 | reward=  3.240 | win_rate= 0.330 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1300/1500 | reward=  4.338 | win_rate= 0.440 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1400/1500 | reward=  4.037 | win_rate= 0.410 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1500/1500 | reward=  3.329 | win_rate= 0.340 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "\n5A training complete.\nTime: 2381.2 seconds\nEnvironment steps: 203,868\nOptimization steps: 50,468\n"
    )]
    out["plots_5a"] = [
        make_plot_output(a_reward, "Moving-average reward", "Level 5A Feature-DQN: reward (reconstructed)"),
        make_plot_output(a_win, "Moving-average win rate", "Level 5A Feature-DQN: win rate (reconstructed)", ymax=1.0),
        no_data_note("hits"),
        no_data_note("steps"),
        no_data_note("losses"),
        make_plot_output(a_eps, "Epsilon", "Level 5A Feature-DQN: epsilon (reconstructed)", ymax=1.0),
    ]
    out["eval_5a"] = [stream(
        "\n===== Level 5A Feature-DQN =====\n"
        "\nvs frozen L4 (agent=left)\n"
        "  left_win_rate: 0.5400\n  right_win_rate: 0.4600\n  draw_rate: 0.0000\n"
        "  mean_left_score: 5.3560\n  mean_right_score: 4.6108\n  mean_steps: 107.4400\n"
        "\nvs Random (agent=left)\n"
        "  left_win_rate: 0.4800\n  right_win_rate: 0.5200\n  draw_rate: 0.0000\n"
        "  mean_left_score: 4.7460\n  mean_right_score: 5.1320\n  mean_steps: 140.6800\n"
        "\nvs Tracking (agent=right)\n"
        "  left_win_rate: 0.9800\n  right_win_rate: 0.0200\n  draw_rate: 0.0000\n"
        "  mean_left_score: 9.8142\n  mean_right_score: 0.0913\n  mean_steps: 126.8000\n"
    )]
    out["save_5a"] = [stream("Saved 5A agent to: C:\\SUTD\\Project Starter Code_new\\saved_agents\\feature_dqn_level5_5a.pt\n")]
    out["train_5b"] = [stream(
        "Episode    1/1500 | reward=  9.987 | win_rate= 1.000 | epsilon= 0.292 | opponent=live | buffer=    89\n"
        "Episode  100/1500 | reward=  3.642 | win_rate= 0.370 | epsilon= 0.050 | opponent=live | buffer= 12025\n"
        "Episode  200/1500 | reward=  6.327 | win_rate= 0.670 | epsilon= 0.050 | opponent=live | buffer= 28102\n"
        "Episode  300/1500 | reward=  6.433 | win_rate= 0.650 | epsilon= 0.050 | opponent=pool | buffer= 43685\n"
        "Episode  400/1500 | reward=  6.330 | win_rate= 0.640 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode  500/1500 | reward=  8.042 | win_rate= 0.810 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode  600/1500 | reward=  7.434 | win_rate= 0.750 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode  700/1500 | reward=  6.540 | win_rate= 0.710 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode  800/1500 | reward=  6.250 | win_rate= 0.670 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode  900/1500 | reward=  4.771 | win_rate= 0.500 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1000/1500 | reward=  5.564 | win_rate= 0.570 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode 1100/1500 | reward=  6.162 | win_rate= 0.640 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode 1200/1500 | reward=  4.972 | win_rate= 0.500 | epsilon= 0.050 | opponent=pool | buffer= 50000\n"
        "Episode 1300/1500 | reward=  4.075 | win_rate= 0.410 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode 1400/1500 | reward=  2.676 | win_rate= 0.270 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "Episode 1500/1500 | reward=  2.973 | win_rate= 0.300 | epsilon= 0.050 | opponent=live | buffer= 50000\n"
        "\n5B training complete.\nTime: 2383.8 seconds\nEnvironment steps: 402,142\nOptimization steps: 99,537"
        "\nFinal opponent pool size: 8\n"
    )]
    out["plots_5b"] = [
        make_plot_output(b_reward, "Moving-average reward", "Level 5B Feature-DQN self-play: reward (reconstructed)"),
        make_plot_output(b_win, "Moving-average win rate", "Level 5B Feature-DQN self-play: win rate (reconstructed)", ymax=1.0),
        no_data_note("hits"),
        no_data_note("steps"),
        no_data_note("losses"),
        make_plot_output(b_eps, "Epsilon", "Level 5B Feature-DQN self-play: epsilon (reconstructed)", ymax=1.0),
        stream("Fraction of episodes played against the live mirror: 0.801 (target ~0.8)\n"),
    ]
    out["eval_5b"] = [stream(
        "\n===== Level 5B Feature-DQN (self-play) =====\n"
        "\nvs frozen L4 (agent=left)\n"
        "  left_win_rate: 0.2400\n  right_win_rate: 0.7600\n  draw_rate: 0.0000\n"
        "  mean_left_score: 2.3758\n  mean_right_score: 7.6132\n  mean_steps: 86.0400\n"
        "\nvs Random (agent=left)\n"
        "  left_win_rate: 0.2800\n  right_win_rate: 0.7200\n  draw_rate: 0.0000\n"
        "  mean_left_score: 2.7736\n  mean_right_score: 7.1509\n  mean_steps: 100.3800\n"
        "\nvs Tracking (agent=right)\n"
        "  left_win_rate: 0.9800\n  right_win_rate: 0.0200\n  draw_rate: 0.0000\n"
        "  mean_left_score: 9.8133\n  mean_right_score: 0.1855\n  mean_steps: 132.2200\n"
        "\n5B (left) vs 5A (right) head-to-head:\n"
        "  left_win_rate: 0.4400\n  right_win_rate: 0.5600\n  draw_rate: 0.0000\n"
        "  mean_left_score: 4.3955\n  mean_right_score: 5.5242\n"
        "\n5B vs itself (mirror match, sanity check for degenerate draws):\n"
        "  left_win_rate: 0.2000\n  right_win_rate: 0.8000\n  draw_rate: 0.0000\n"
        "  mean_left_score: 1.9572\n  mean_right_score: 7.9518\n"
    )]
    out["comparison_table"] = [stream(
        "opponent | 5A win_rate | 5B win_rate | 5A mean_score | 5B mean_score\n"
        "--------------------------------------------------------------------\n"
        "vs frozen L4 (agent=left) | 0.5400 | 0.2400 | 5.3560 | 2.3758\n"
        "vs Random (agent=left) | 0.4800 | 0.2800 | 4.7460 | 2.7736\n"
        "vs Tracking (agent=right) | 0.0200 | 0.0200 | 0.0913 | 0.1855\n"
    )]
    out["save_reload_5b"] = [stream(
        "Saved final Level 5 Feature-DQN agent to: C:\\SUTD\\Project Starter Code_new\\saved_agents\\feature_dqn_level5_selfplay.pt\n"
        "Reloaded: Level 5B Feature DQN (self-play) | environment steps: 402142 | optimization steps: 99537\n"
    )]
    out["reloaded_eval"] = [stream(
        "Reloaded 5B agent vs frozen L4:\n"
        "  left_win_rate: 0.3000\n  right_win_rate: 0.7000\n  draw_rate: 0.0000\n"
        "  mean_left_score: 2.9764\n  mean_right_score: 7.0150\n"
    )]
    out["replay_text"] = [stream(replay_data["feature"]["text"] + "\n")]
    out["replay_anim"] = [html_result(replay_data["feature"]["html"], 20)]
    return out


def build_cnn_outputs():
    a_reward = [-0.114,0.558,0.553,0.349,1.658,1.465,1.369,0.867,0.659,1.370,1.365,1.065,1.944,2.248,1.346,1.537]
    a_win =    [1.000,0.110,0.140,0.110,0.210,0.150,0.160,0.100,0.090,0.150,0.170,0.120,0.310,0.280,0.210,0.260]
    a_eps =    [1.000,1.000,1.000,0.474,0.262,0.150,0.091,0.057,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050]
    b_reward = [9.979,3.453,2.230,3.135,3.018,3.452,2.443,2.646,2.737,3.722,3.137,3.442,4.347,4.133,3.335,2.130]
    b_win =    [1.000,0.410,0.430,0.470,0.430,0.380,0.350,0.320,0.360,0.490,0.430,0.430,0.570,0.620,0.680,0.770]
    b_eps =    [0.295,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.050]

    out = {}
    out["imports"] = [stream("PyTorch version: 2.5.1+cu121\nDevice: cuda\nGPU: NVIDIA GeForce RTX 3070 Ti\n")]
    out["load_frozen"] = [stream("Loaded frozen Level 4 opponent: Level 4 CNN-DQN\nEnvironment steps at save time: 500379\n")]
    out["train_5a"] = [stream(
        "Episode    1/1500 | reward= -0.114 | win_rate= 1.000 | epsilon= 1.000 | opponent=pool | buffer=   450\n"
        "Episode  100/1500 | reward=  0.558 | win_rate= 0.110 | epsilon= 1.000 | opponent=pool | buffer= 11552\n"
        "Episode  200/1500 | reward=  0.553 | win_rate= 0.140 | epsilon= 1.000 | opponent=pool | buffer= 25287\n"
        "Episode  300/1500 | reward=  0.349 | win_rate= 0.110 | epsilon= 0.474 | opponent=pool | buffer= 30000\n"
        "Episode  400/1500 | reward=  1.658 | win_rate= 0.210 | epsilon= 0.262 | opponent=pool | buffer= 30000\n"
        "Episode  500/1500 | reward=  1.465 | win_rate= 0.150 | epsilon= 0.150 | opponent=pool | buffer= 30000\n"
        "Episode  600/1500 | reward=  1.369 | win_rate= 0.160 | epsilon= 0.091 | opponent=pool | buffer= 30000\n"
        "Episode  700/1500 | reward=  0.867 | win_rate= 0.100 | epsilon= 0.057 | opponent=pool | buffer= 30000\n"
        "Episode  800/1500 | reward=  0.659 | win_rate= 0.090 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode  900/1500 | reward=  1.370 | win_rate= 0.150 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1000/1500 | reward=  1.365 | win_rate= 0.170 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1100/1500 | reward=  1.065 | win_rate= 0.120 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1200/1500 | reward=  1.944 | win_rate= 0.310 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1300/1500 | reward=  2.248 | win_rate= 0.280 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1400/1500 | reward=  1.346 | win_rate= 0.210 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1500/1500 | reward=  1.537 | win_rate= 0.260 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "\n5A training complete.\nTime: 3225.0 seconds\nEnvironment steps: 198,824\nOptimization steps: 48,457\n"
    )]
    out["plots_5a"] = [
        make_plot_output(a_reward, "Moving-average reward", "Level 5A CNN-DQN: reward (reconstructed)"),
        make_plot_output(a_win, "Moving-average win rate", "Level 5A CNN-DQN: win rate (reconstructed)", ymax=1.0),
        no_data_note("hits"),
        no_data_note("steps"),
        no_data_note("losses"),
        make_plot_output(a_eps, "Epsilon", "Level 5A CNN-DQN: epsilon (reconstructed)", ymax=1.0),
    ]
    out["eval_5a"] = [stream(
        "\n===== Level 5A CNN-DQN =====\n"
        "\nvs frozen L4 (agent=left)\n"
        "  left_win_rate: 0.2000\n  right_win_rate: 0.8000\n  draw_rate: 0.0000\n"
        "  mean_left_score: 1.7203\n  mean_right_score: 7.7087\n  mean_steps: 256.2400\n"
        "\nvs Random (agent=left)\n"
        "  left_win_rate: 0.4600\n  right_win_rate: 0.5400\n  draw_rate: 0.0000\n"
        "  mean_left_score: 4.5651\n  mean_right_score: 5.3358\n  mean_steps: 131.4200\n"
        "\nvs Tracking (agent=right)\n"
        "  left_win_rate: 0.9800\n  right_win_rate: 0.0200\n  draw_rate: 0.0000\n"
        "  mean_left_score: 9.3906\n  mean_right_score: 0.0946\n  mean_steps: 287.6200\n"
    )]
    out["save_5a"] = [stream("Saved 5A agent to: C:\\SUTD\\Project Starter Code_new\\saved_agents\\cnn_dqn_level5_5a.pt\n")]
    out["train_5b"] = [stream(
        "Episode    1/1500 | reward=  9.979 | win_rate= 1.000 | epsilon= 0.295 | opponent=live | buffer=   103\n"
        "Episode  100/1500 | reward=  3.453 | win_rate= 0.410 | epsilon= 0.050 | opponent=pool | buffer= 15441\n"
        "Episode  200/1500 | reward=  2.230 | win_rate= 0.430 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode  300/1500 | reward=  3.135 | win_rate= 0.470 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode  400/1500 | reward=  3.018 | win_rate= 0.430 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode  500/1500 | reward=  3.452 | win_rate= 0.380 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode  600/1500 | reward=  2.443 | win_rate= 0.350 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode  700/1500 | reward=  2.646 | win_rate= 0.320 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode  800/1500 | reward=  2.737 | win_rate= 0.360 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode  900/1500 | reward=  3.722 | win_rate= 0.490 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode 1000/1500 | reward=  3.137 | win_rate= 0.430 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode 1100/1500 | reward=  3.442 | win_rate= 0.430 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode 1200/1500 | reward=  4.347 | win_rate= 0.570 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1300/1500 | reward=  4.133 | win_rate= 0.620 | epsilon= 0.050 | opponent=pool | buffer= 30000\n"
        "Episode 1400/1500 | reward=  3.335 | win_rate= 0.680 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "Episode 1500/1500 | reward=  2.130 | win_rate= 0.770 | epsilon= 0.050 | opponent=live | buffer= 30000\n"
        "\n5B training complete.\nTime: 6430.7 seconds\nEnvironment steps: 522,814\nOptimization steps: 128,205"
        "\nFinal opponent pool size: 8\n"
    )]
    out["plots_5b"] = [
        make_plot_output(b_reward, "Moving-average reward", "Level 5B CNN-DQN self-play: reward (reconstructed)"),
        make_plot_output(b_win, "Moving-average win rate", "Level 5B CNN-DQN self-play: win rate (reconstructed)", ymax=1.0),
        no_data_note("hits"),
        no_data_note("steps"),
        no_data_note("losses"),
        make_plot_output(b_eps, "Epsilon", "Level 5B CNN-DQN self-play: epsilon (reconstructed)", ymax=1.0),
        stream("Fraction of episodes played against the live mirror: 0.799 (target ~0.8)\n"),
    ]
    out["eval_5b"] = [stream(
        "\n===== Level 5B CNN-DQN (self-play) =====\n"
        "\nvs frozen L4 (agent=left)\n"
        "  left_win_rate: 0.7200\n  right_win_rate: 0.2800\n  draw_rate: 0.0000\n"
        "  mean_left_score: 6.4910\n  mean_right_score: 2.0654\n  mean_steps: 548.2000\n"
        "\nvs Random (agent=left)\n"
        "  left_win_rate: 0.9800\n  right_win_rate: 0.0200\n  draw_rate: 0.0000\n"
        "  mean_left_score: 9.5801\n  mean_right_score: 0.1195\n  mean_steps: 169.1200\n"
        "\nvs Tracking (agent=right)\n"
        "  left_win_rate: 0.8000\n  right_win_rate: 0.2000\n  draw_rate: 0.0000\n"
        "  mean_left_score: 6.6099\n  mean_right_score: 1.8789\n  mean_steps: 494.6600\n"
        "\n5B (left) vs 5A (right) head-to-head:\n"
        "  left_win_rate: 0.8600\n  right_win_rate: 0.1400\n  draw_rate: 0.0000\n"
        "  mean_left_score: 8.5631\n  mean_right_score: 1.2920\n"
        "\n5B vs itself (mirror match, sanity check for degenerate draws):\n"
        "  left_win_rate: 0.4200\n  right_win_rate: 0.5800\n  draw_rate: 0.0000\n"
        "  mean_left_score: 4.1202\n  mean_right_score: 5.6909\n"
    )]
    out["comparison_table"] = [stream(
        "opponent | 5A win_rate | 5B win_rate | 5A mean_score | 5B mean_score\n"
        "--------------------------------------------------------------------\n"
        "vs frozen L4 (agent=left) | 0.2000 | 0.7200 | 1.7203 | 6.4910\n"
        "vs Random (agent=left) | 0.4600 | 0.9800 | 4.5651 | 9.5801\n"
        "vs Tracking (agent=right) | 0.0200 | 0.2000 | 0.0946 | 1.8789\n"
    )]
    out["save_reload_5b"] = [stream(
        "Saved final Level 5 CNN-DQN agent to: C:\\SUTD\\Project Starter Code_new\\saved_agents\\cnn_dqn_level5_selfplay.pt\n"
        "Reloaded: Level 5B CNN-DQN (self-play) | environment steps: 522814 | optimization steps: 128205\n"
    )]
    out["reloaded_eval"] = [stream(
        "Reloaded 5B agent vs frozen L4:\n"
        "  left_win_rate: 0.5500\n  right_win_rate: 0.4500\n  draw_rate: 0.0000\n"
        "  mean_left_score: 2.9178\n  mean_right_score: 2.8911\n"
    )]
    out["replay_text"] = [stream(replay_data["cnn"]["text"] + "\n")]
    out["replay_anim"] = [html_result(replay_data["cnn"]["html"], 20)]
    return out


def patch_notebook(path, cell_id_to_key, outputs_by_key):
    nb = nbf.read(path, as_version=4)

    # insert disclosure markdown cell right after the first (intro) markdown cell
    intro_idx = 0
    disclosure_cell = nbf.v4.new_markdown_cell(DISCLOSURE_MD)
    nb.cells.insert(intro_idx + 1, disclosure_cell)

    exec_count = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        exec_count += 1
        key = cell_id_to_key.get(cell.get("id"))
        cell["execution_count"] = exec_count
        if key and key in outputs_by_key:
            outs = outputs_by_key[key]
            # fix execution_count on execute_result outputs to match this cell
            fixed_outs = []
            for o in outs:
                if o.get("output_type") == "execute_result":
                    o = dict(o)
                    o["execution_count"] = exec_count
                fixed_outs.append(o)
            cell["outputs"] = fixed_outs
        else:
            cell["outputs"] = []

    nb = nbf.from_dict(json.loads(json.dumps(nb)))
    nbf.validate(nb)
    nbf.write(nb, path)
    print(f"Patched {path.name}: {exec_count} code cells, disclosure cell inserted.")


FEATURE_MAP = {
    "d75612e5": "imports",
    "749a9056": "feature_dim",
    "a0135be5": "load_frozen",
    "22fba961": "train_5a",
    "8aabd652": "plots_5a",
    "11a6c074": "eval_5a",
    "8edae986": "save_5a",
    "65db97c2": "train_5b",
    "c8c4336a": "plots_5b",
    "36a71a44": "eval_5b",
    "4346590f": "comparison_table",
    "43b434e4": "save_reload_5b",
    "63861787": "reloaded_eval",
    "81d0a275": "replay_text",
    "49ddfe11": "replay_anim",
}

CNN_MAP = {
    "0d89b374": "imports",
    "ab27d5ad": "load_frozen",
    "e9b9438e": "train_5a",
    "837cfbf2": "plots_5a",
    "dec42870": "eval_5a",
    "0d4b8c99": "save_5a",
    "81904ad0": "train_5b",
    "51f7eb34": "plots_5b",
    "1026c460": "eval_5b",
    "dc20fb43": "comparison_table",
    "089bfa66": "save_reload_5b",
    "b1d74dfc": "reloaded_eval",
    "881ca57b": "replay_text",
    "c47b059f": "replay_anim",
}

if __name__ == "__main__":
    patch_notebook(NB_DIR / "Level_5_Feature_DQN_SelfPlay.ipynb", FEATURE_MAP, build_feature_outputs())
    patch_notebook(NB_DIR / "Level_5_CNN_DQN_SelfPlay.ipynb", CNN_MAP, build_cnn_outputs())
