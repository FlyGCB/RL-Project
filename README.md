# Paddle Duel RL Arena — Reproducibility Instructions

SUTD 51.512 Reinforcement Learning for Embodied AI — Project submission.

**Repository:** https://github.com/FlyGCB/RL-Project

**Official report: `Final_Project_Report.pdf`** (covers all five levels,
including Level 5 self-play).

## 1. Environment setup

- Python 3.10 (tested on 3.10.11, a plain python.org install — **not** a conda
  environment; the two are known to conflict on this project's dev machines).
- Install dependencies:

  ```
  python -m pip install -r requirements.txt
  ```

- For GPU-accelerated training (Level 4/5 neural agents), install a CUDA-enabled
  PyTorch build matching your GPU driver, e.g.:

  ```
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

  Verify with:

  ```
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

  Training was run and verified on an NVIDIA RTX 3070 Ti with `torch==2.5.1+cu121`.
  CPU-only execution works but is significantly slower (the environment's
  rendering step is pure NumPy and is the actual bottleneck, not the neural
  network forward/backward pass — see Level 5 notes below).

- Running notebooks needs `jupyter` (`python -m pip install jupyter`) or any
  Jupyter-compatible IDE (VS Code, JupyterLab). Converting a notebook to a
  plain script (used for the Level 5 background training run, see §4) needs
  `nbconvert` and `nbformat`, both already in `requirements.txt`.

## 2. Project structure

```
paddle_duel_env.py           Fixed environment (do not modify) — provided by the course
student_agent_template.py    BaseAgent-compatible template for a submittable agent
requirements.txt
README.md
Final_Project_Report.pdf      Official submitted report (all 5 levels)
notebooks/
  0. Paddle Duel RL Arena - Environment Walkthrough and Project Intro.ipynb
  Level_1_Qlearning_SARSA_ExpectedSARSA_Comparison.ipynb
  Level_1_Comparison_Exploration_and_Diagnostics.ipynb
  Level_2_Qlearning_vs_SARSA_Normal.ipynb
  Level_2_Qlearning_Exploration_Schedules.ipynb
  Level_3_Expected_SARSA_Robustness.ipynb
  Level_4_Feature_DQN_Baseline.ipynb
  Level_4_CNN_DQN_Pixel_Baseline.ipynb
  Level_5_Feature_DQN_SelfPlay.ipynb
  Level_5_CNN_DQN_SelfPlay.ipynb
scripts/                      Level 5 background-training entry points (converted from the
                               notebooks, see §4), the timing calibration script, and the
                               report-generation pipeline (_backfill_notebooks.py,
                               _make_report_figures.py, _make_report_pdf.py)
saved_agents/                 All trained checkpoints (Q-tables + PyTorch weights)
report_assets/                Report figures (PNGs) and the standalone training-curves HTML
logs/                         Console logs from the Level 5 full-scale training runs
Extras/play_realtime_wasd.py  Optional manual-play demo (pygame), not part of the graded pipeline
```

The blank official course templates (`Level 1-5 — Starter Notebook.ipynb`) and the
course-provided `Project_Instructions_51_512.docx/.pdf` are intentionally not part of
this submission (see `.gitignore`) — they are the unmodified starting point, not our
work product; `notebooks/` above lists only the notebooks we actually wrote.

## 3. Run order

Levels are cumulative in difficulty but independent to *run* — each notebook
loads only the checkpoints it needs. All paths below are relative to
`notebooks/`. Suggested reading/reproduction order:

1. `0. Paddle Duel RL Arena - Environment Walkthrough and Project Intro.ipynb`
   — environment API tour.
2. **Level 1** (tabular, solo vs wall): `Level_1_Qlearning_SARSA_ExpectedSARSA_Comparison.ipynb`,
   then `Level_1_Comparison_Exploration_and_Diagnostics.ipynb`.
3. **Level 2** (tabular, vs scripted opponent): `Level_2_Qlearning_vs_SARSA_Normal.ipynb`,
   then `Level_2_Qlearning_Exploration_Schedules.ipynb`.
4. **Level 3** (tabular, vs opponent zoo): `Level_3_Expected_SARSA_Robustness.ipynb`.
5. **Level 4** (pixel-only, vs scripted opponents): `Level_4_Feature_DQN_Baseline.ipynb`
   and `Level_4_CNN_DQN_Pixel_Baseline.ipynb` (independent of each other; both
   load nothing from Level 1-3).
6. **Level 5** (two learned agents, self-play): `Level_5_Feature_DQN_SelfPlay.ipynb`
   and `Level_5_CNN_DQN_SelfPlay.ipynb`. Each **requires** the matching Level 4
   checkpoint (`feature_dqn_level4.pt` / `cnn_dqn_level4.pt`) to already exist
   in `saved_agents/`, since it is loaded as the frozen 5A opponent and as the
   pinned seed of the 5B opponent pool.

All notebooks are self-contained: each defines its own environment/agent
classes inline rather than importing across notebooks, per the course's
"keep the environment and reusable agent code in .py files, notebooks for
experiments" guidance being satisfied by `paddle_duel_env.py` as the one
shared reusable file.

## 4. Reproducing the Level 5 results specifically

The Level 5 notebooks (`Level_5_Feature_DQN_SelfPlay.ipynb`,
`Level_5_CNN_DQN_SelfPlay.ipynb`) train for 1500 + 1500 episodes per
architecture (5A fixed-opponent baseline, then 5B self-play). This takes
roughly **80 minutes (Feature-DQN)** and **160 minutes (CNN-DQN)** on an
RTX 3070 Ti — too long to run interactively in a browser tab reliably, so the
actual training that produced the checkpoints in `saved_agents/` was run as a
converted, non-interactive Python script rather than inside the Jupyter
kernel:

```
python -m jupyter nbconvert --to script "notebooks\Level_5_Feature_DQN_SelfPlay.ipynb" --output-dir scripts --output level5_feature_dqn_train_full
python -m jupyter nbconvert --to script "notebooks\Level_5_CNN_DQN_SelfPlay.ipynb" --output-dir scripts --output level5_cnn_dqn_train_full
```

then, after adding `matplotlib.use("Agg")` right after the `import matplotlib.pyplot as plt`
line (headless run, no display attached), run from the project root so the
scripts' own `Path.cwd()`-based project-root detection finds `paddle_duel_env.py`:

```
python scripts\level5_feature_dqn_train_full.py > logs\feature_dqn_FULL_TRAIN_log.txt 2>&1
python scripts\level5_cnn_dqn_train_full.py    > logs\cnn_dqn_FULL_TRAIN_log.txt    2>&1
```

The raw console logs from the actual run that produced the submitted
checkpoints are kept in `logs/*_FULL_TRAIN_log.txt` for
verification. The two `.ipynb` files' cell outputs have been backfilled with
this same run's real text/plots so the notebooks are self-documenting when
opened directly — see the disclosure note at the top of each Level 5
notebook for exactly which outputs are live vs. backfilled from these logs.

To fully re-run from scratch instead of trusting the checkpoints: delete
`feature_dqn_level5_5a.pt`, `feature_dqn_level5_selfplay.pt`,
`cnn_dqn_level5_5a.pt`, `cnn_dqn_level5_selfplay.pt` from `saved_agents/`
(keep `feature_dqn_level4.pt` / `cnn_dqn_level4.pt` — those are Level 4's
deliverables and are required inputs to Level 5) and re-run the two notebooks
top to bottom, or the two converted scripts above.

**Smoke-test before a full run**: both Level 5 notebooks include a
"Smoke-test option" markdown note before the 5A and 5B training cells —
temporarily shrink `N_EPISODES_5A`/`N_EPISODES_5B`, `FORCED_RANDOM_EPISODES_5A`,
`SNAPSHOT_EVERY`, and `min_buffer_size` per those notes to confirm the
pipeline runs end-to-end (~1 minute) before committing to the full multi-hour
run.

## 5. Reloading a trained agent for evaluation (no retraining)

Every saved agent can be reloaded and evaluated without retraining. Example
for the final Level 5 deliverables:

```python
from pathlib import Path
import torch
# Feature-DQN (see Level_5_Feature_DQN_SelfPlay.ipynb for the FeatureDQNAgent class)
agent = FeatureDQNAgent.load(Path("saved_agents/feature_dqn_level5_selfplay.pt"), seed=999)
# CNN-DQN (see Level_5_CNN_DQN_SelfPlay.ipynb for the CNNDQNAgent class)
agent = CNNDQNAgent.load(Path("saved_agents/cnn_dqn_level5_selfplay.pt"), seed=999)
```

Tabular agents (Levels 1-3, `.pkl` files under `saved_agents/`) are reloaded
via the corresponding notebook's own `load`/`from_pickle` helper — see each
notebook's save/reload cell.

The Level 5 self-play checkpoints (`feature_dqn_level5_selfplay.pt`,
`cnn_dqn_level5_selfplay.pt`) are the agents intended for the competition
interface: the environment mirrors observations for the right side, so the
same weights are valid dropped into either the Left or Right slot of the
teaching team's competition notebook.

## 6. Checkpoint index

| File | Level | Method |
|---|---|---|
| `q_learning.pkl`, `sarsa.pkl`, `expected_sarsa.pkl` | 1 | Tabular, solo vs wall |
| `q_learning_level2.pkl`, `q_learning_constant_level2.pkl`, `q_learning_linear_level2.pkl`, `q_learning_exponential_level2.pkl`, `sarsa_level2.pkl` | 2 | Tabular, vs scripted opponent, exploration-schedule comparison |
| `expected_sarsa_level3_zoo.pkl` | 3 | Tabular, vs opponent zoo |
| `feature_dqn_level4.pt` | 4 | Feature-DQN, pixel-derived features, vs scripted opponents |
| `cnn_dqn_level4.pt` | 4 | CNN-DQN, raw pixels, vs scripted opponents |
| `feature_dqn_level5_5a.pt` | 5A | Feature-DQN, vs frozen Level 4 opponent |
| `feature_dqn_level5_selfplay.pt` | 5B (**final deliverable**) | Feature-DQN, self-play + opponent pool |
| `cnn_dqn_level5_5a.pt` | 5A | CNN-DQN, vs frozen Level 4 opponent |
| `cnn_dqn_level5_selfplay.pt` | 5B (**final deliverable**) | CNN-DQN, self-play + opponent pool |

## 7. Known environment gotchas

- Windows PowerShell buffers a Python process's stdout when redirected to a
  file (not a TTY) — periodic training-progress prints only land on disk in
  large chunks, not line-by-line in real time. This does not affect
  correctness, only how live the log file looks while a run is in progress.
- `ReplayBuffer.sample()` / `PixelReplayBuffer.sample()` raise `ValueError` if
  `batch_size > len(buffer)`. Keep `min_buffer_size >= batch_size` (default
  `batch_size=64`) in any agent config you edit, including smoke tests.
- `TrackingAgent` is side-aware and reads unmirrored `info` — when
  benchmarking against it through `evaluate_duel_agents`/`play_rally`, it
  must be placed on the **left** side (see the note in both Level 5
  notebooks and in `Level_3_Expected_SARSA_Robustness.ipynb`).

## 8. Group members and contributions

| Name | Student ID | Contributions |
|---|---|---|
| Duan Xu | 1010728 | Designed and implemented Level 5 (two-agent self-play): built the fixed-opponent baseline (5A) and self-play training with an opponent pool (5B) for both the Feature-DQN and CNN-DQN architectures, ran the full-scale training and evaluation, and produced the final competition-ready checkpoints. |
| Manan Bimal Mehta | 1010949 | Designed and implemented Levels 1–3 (tabular Q-learning, SARSA, and Expected SARSA agents; exploration-schedule comparison; opponent-zoo robustness evaluation) and Level 4 (Feature-DQN and CNN-DQN pixel-based agents vs. scripted opponents), including all training, evaluation, and baseline comparisons for these levels. |
| Zheng Chengsheng | 1010719 | Wrote and verified the final project report, consolidating results and figures across all five levels, cross-checking reported metrics against the actual notebook/training outputs, and compiling the reproducibility documentation. |
