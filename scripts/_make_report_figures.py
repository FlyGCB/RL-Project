"""Generate standalone PNG figures for the report from the same real,
verified data used to backfill the Level 5 notebooks (see _backfill_notebooks.py
for the source-of-truth log line references)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
OUT = PROJECT_ROOT / "report_assets"
OUT.mkdir(exist_ok=True)

EPISODES = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]


def plot_pair(a, b, ylabel, title, fname, ymax=None, a_label="5A", b_label="5B (self-play)"):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(EPISODES, a, marker="o", label=a_label, color="#2a78d6")
    ax.plot(EPISODES, b, marker="o", label=b_label, color="#eb6834")
    ax.set_xlabel("Training episode (16-pt periodic-log reconstruction)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    if ymax is not None:
        ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    print("wrote", fname)


# Feature-DQN
feat_a_reward = [10.000,1.658,2.258,2.258,1.850,2.744,2.422,2.433,2.413,3.236,2.528,4.221,3.240,4.338,4.037,3.329]
feat_a_win =    [1.000,0.170,0.230,0.230,0.190,0.280,0.250,0.250,0.250,0.330,0.260,0.430,0.330,0.440,0.410,0.340]
feat_b_reward = [9.987,3.642,6.327,6.433,6.330,8.042,7.434,6.540,6.250,4.771,5.564,6.162,4.972,4.075,2.676,2.973]
feat_b_win =    [1.000,0.370,0.670,0.650,0.640,0.810,0.750,0.710,0.670,0.500,0.570,0.640,0.500,0.410,0.270,0.300]

plot_pair(feat_a_win, feat_b_win, "Win rate vs training-time opponent",
          "Feature-DQN: win rate, 5A vs 5B", "feature_dqn_winrate.png", ymax=1.0)
plot_pair(feat_a_reward, feat_b_reward, "Episode reward",
          "Feature-DQN: reward, 5A vs 5B", "feature_dqn_reward.png")

# CNN-DQN
cnn_a_reward = [-0.114,0.558,0.553,0.349,1.658,1.465,1.369,0.867,0.659,1.370,1.365,1.065,1.944,2.248,1.346,1.537]
cnn_a_win =    [1.000,0.110,0.140,0.110,0.210,0.150,0.160,0.100,0.090,0.150,0.170,0.120,0.310,0.280,0.210,0.260]
cnn_b_reward = [9.979,3.453,2.230,3.135,3.018,3.452,2.443,2.646,2.737,3.722,3.137,3.442,4.347,4.133,3.335,2.130]
cnn_b_win =    [1.000,0.410,0.430,0.470,0.430,0.380,0.350,0.320,0.360,0.490,0.430,0.430,0.570,0.620,0.680,0.770]

plot_pair(cnn_a_win, cnn_b_win, "Win rate vs training-time opponent",
          "CNN-DQN: win rate, 5A vs 5B", "cnn_dqn_winrate.png", ymax=1.0)
plot_pair(cnn_a_reward, cnn_b_reward, "Episode reward",
          "CNN-DQN: reward, 5A vs 5B", "cnn_dqn_reward.png")

print("Done ->", OUT)
