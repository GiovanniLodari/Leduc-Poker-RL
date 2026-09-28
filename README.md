# Leduc Poker RL

Reinforcement-learning experiments on **Leduc Poker**, a small imperfect-information poker game, using [OpenSpiel](https://github.com/google-deepmind/open_spiel). The project compares **NFSP** (*Neural Fictitious Self-Play*) with **DQN**, **CFR** and **Deep CFR**, measures how exploitable the learned strategies are (*exploitability*) and tries variants of the NFSP network.

## Contents

| Path | Content |
|---|---|
| `leduc_poker_project/replica_deepmind.py` | NFSP and DQN training (JAX) on Leduc Poker |
| `leduc_poker_project/deepmind_nsfp_*.py` | NFSP variants: attention network, small or deep MLP, warm-up, etc. |
| `leduc_poker_project/comparison_study*.py` | Comparison across algorithms (`nfsp`, `dqn`, `cfr`, `deep_cfr`) |
| `leduc_poker_project/tournament_study.py` | Tournament between agents with Alpha-Rank ranking |
| `leduc_poker_project/visualize_strategy*.py` | Visualization of learned strategies |
| `leduc_poker_project/run_experiments.py` | Runs a series of experiments in sequence |
| `leduc_poker_project/dashboard/` | Local monitor (FastAPI) to follow training |
| `leduc_poker_project/logs/`, `plots/` | Run logs (JSONL) and generated plots |
| `archive/` | Plots from earlier runs |

During training, metrics are written to `leduc_poker_project/logs/` and plots to `leduc_poker_project/plots/`. Checkpoints go to `leduc_poker_project/checkpoints/`, which is not in the repository because it is heavy.

## Installation

An environment where OpenSpiel can be installed is required (Linux, macOS or WSL).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install open_spiel            # or build from source, see the OpenSpiel guide
```

## Usage

Run the scripts from the repository root.

```bash
# NFSP training on Leduc Poker
python leduc_poker_project/replica_deepmind.py --algo nfsp --episodes 1000000

# Comparison with other algorithms
python leduc_poker_project/comparison_study.py --algo cfr --iterations 100000

# Alpha-Rank tournament between checkpoints (path:algo:label)
python leduc_poker_project/tournament_study.py --agents <path>:nfsp:NFSP_16 <path>:dqn:DQN --plot

# Strategy learned by a checkpoint
python leduc_poker_project/visualize_strategy.py --path <checkpoint>/params.pkl --algo nfsp
```

The training scripts also start the monitor at <http://localhost:8000>, which shows the runs in `logs/`.
