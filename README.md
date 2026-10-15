# UAV-HMAPPO: Hierarchical Multi-Agent PPO for UAV Networks

Hierarchical Multi-Agent Proximal Policy Optimisation (MAPPO) for joint user
clustering, UAV deployment and resource allocation in a multi-UAV downlink
network. A slow high-level policy decides where each UAV flies and which users
it serves; a fast low-level policy splits each UAV's bandwidth and sets its
transmit power. Both levels use a centralised critic with decentralised actors.
The method is compared against a flat MAPPO agent and a K-means geometric
baseline on a fixed benchmark of 50 user layouts.

## Team

- R. Neelakesav
- B. Preetham
- D. Manish

**Guide:** Dr. Vinnakota Venu Balaji

## Folder structure

```
uav-mappo/
├── agents/
│   ├── mappo.py            # flat MAPPO (actor, centralised critic, PPO update)
│   └── hierarchical.py     # two-level hierarchical MAPPO
├── baselines/
│   └── kmeans_baseline.py  # K-means clustering + greedy deployment baseline
├── configs/
│   └── default.yaml        # environment, PPO and training settings
├── envs/
│   ├── uav_env.py          # Gymnasium multi-UAV environment and reward
│   └── channel_model.py    # air-to-ground path loss, SINR and rate model
├── scripts/
│   ├── train.py            # train hierarchical (default) or flat (--flat) MAPPO
│   ├── evaluate.py         # compare all methods on the fixed benchmark
│   └── demo.py             # Streamlit demo of the trained policy
├── utils/                  # config loading, metrics, plotting, eval scenarios
├── tests/
│   └── test_env.py         # environment, channel, metric and baseline tests
├── results/                # trained models, logs, comparison tables and plots
└── requirements.txt
```

## Setup

```bash
python -m venv venv
# Windows:      venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Run the tests
pytest -q

# K-means baseline
python baselines/kmeans_baseline.py

# Train hierarchical MAPPO (or add --flat for flat MAPPO)
python scripts/train.py --episodes 3000
python scripts/train.py --episodes 3000 --flat

# Evaluate all methods on the 50 fixed scenarios (writes results/comparison.csv)
python scripts/evaluate.py

# Interactive demo
streamlit run scripts/demo.py
```

## Results

Mean over the 50 fixed evaluation scenarios (3 UAVs, 20 users, 200 steps),
from `results/comparison.csv`. Coverage is the fraction of users receiving
at least 1 Mbps.

| Method | Coverage | Avg rate (Mbps) | Total throughput (Mbps) | Jain fairness | Energy eff. (kbit/J) | Spectral eff. (bps/Hz) | Return |
|---|---|---|---|---|---|---|---|
| K-means | 90.3% ± 5.8% | 9.59 | 191.7 | 0.563 | 380.7 | 3.20 | 284.6 |
| Flat MAPPO | 53.8% ± 14.9% | 4.49 | 89.8 | 0.278 | 146.9 | 1.54 | 128.6 |
| Hierarchical MAPPO | 59.3% ± 8.5% | 5.35 | 107.0 | 0.278 | 177.1 | 1.79 | 153.9 |

Hierarchical MAPPO outperforms flat MAPPO on coverage, throughput and energy
efficiency, with lower variance across scenarios. Both learned agents still
trail the K-means baseline, mainly because the learned UAVs do not settle
precisely over their user clusters.
