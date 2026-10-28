# Hierarchical deep reinforcement learning for the capacitated lot-sizing problem

Code, benchmark heuristics and trained models of the article

> **Hierarchical deep reinforcement learning for the capacitated lot-sizing problem: From stochastic multi-item to sequence-dependent setups**

The repository reproduces the Hierarchical Deep Reinforcement Learning (HDRL) framework proposed in the paper and the benchmark heuristics used for comparison. In short, a **high-level agent** selects which products to produce in each period (and, with sequence-dependent setups, in which order), and a **low-level agent** sequentially decides the production quantity of each selected product under the remaining capacity. Both agents are trained with Proximal Policy Optimization (PPO), and the high-level agent is warm started by behaviour cloning of the AMBS heuristic.

With this repository you can:

- **train** the hierarchical PPO agents on any of the benchmark instances of the paper,
- **test** the trained models provided in `Trained Models/` with the evaluation protocol of the paper,
- **run the AMBS and AMBS-SD benchmark heuristics** (calibration and evaluation).

> This repository is anonymised for the review process. Author information will be restored in the final published version.

---

## Repository structure

```
.
├── main.ipynb                       # Train and test the HDRL agents (S-CLSP and S-CLSP-SD)
├── AMBS_Heuristic_SCLSP.ipynb       # AMBS benchmark heuristic, Case I
├── AMBS_Heuristic_SCLSP_SD.ipynb    # AMBS-SD benchmark heuristic, Case II
├── CLSP_HDRL/                       # Core implementation (imported by main.ipynb)
│   ├── algo_ppo_hier.py             #   hierarchical PPO for the S-CLSP
│   ├── algo_ppo_hier2.py            #   hierarchical PPO for the S-CLSP-SD
│   ├── envs.py                      #   S-CLSP environment
│   ├── envs2.py                     #   S-CLSP-SD environment
│   ├── nets.py                      #   high-level and low-level networks
│   ├── buffers.py                   #   rollout buffers and returns
│   ├── dists.py                     #   action distributions (masked Bernoulli, Beta, Plackett-Luce)
│   └── warm_start.py                #   AMBS warm start (behaviour cloning) and calibrated AMBS thresholds
└── Trained Models/
    ├── Case I SCLSP/                #   instances 1.5 ... 1.24
    │   └── <instance ID>/high_joint_best.pt, low_joint_best.pt
    └── Case II SCLSP-SD/            #   instances 2.1 ... 2.16
        └── <instance ID>/high_joint_best.pt, low_joint_best.pt
```

---

## Benchmark instances

All instances share mean demand μ = 4, holding cost h = 1, backorder cost b = 9 and unit batch size. Demand is **low** variability, U{3, 5}, or **high** variability, U{0, 8}; cf is the capacity factor (1.1 = tight, 1.5 = loose). Within every block of four IDs the order is: (cf 1.1, low), (cf 1.5, low), (cf 1.1, high), (cf 1.5, high).

**Case I, SCLSP (`1.1` – `1.24`)**

| K (products) | IDs |
|---|---|
| 5 | 1.1 – 1.4 |
| 10 | 1.5 – 1.8 |
| 15 | 1.9 – 1.12 |
| 20 | 1.13 – 1.16 |
| 25 | 1.17 – 1.20 |
| 30 | 1.21 – 1.24 |

**Case II, SCLSP-SD (`2.1` – `2.16`)**, sequence-dependent setup times and costs. Type A: two families. Type B: five families arranged on a line.

| Setup structure | K | IDs |
|---|---|---|
| Type A | 15 | 2.1 – 2.4 |
| Type A | 20 | 2.5 – 2.8 |
| Type B | 15 | 2.9 – 2.12 |
| Type B | 20 | 2.13 – 2.16 |

Example: `"1.19"` = K = 25, cf = 1.1, high demand variability; `"2.14"` = Type B, K = 20, cf = 1.5, low demand variability.

Trained models are provided for instances **1.5 – 1.24** and **2.1 – 2.16**. The instances `1.1` – `1.4` (K = 5) can be trained with `main.ipynb`, but no trained models are included for them.

---

## Requirements

- Python 3.10 or newer and Jupyter (or Google Colab).
- Python packages: `numpy`, `torch`, `matplotlib`, `tqdm`, `wandb`.
  `wandb` is only imported by the training cell and is **not used** unless `use_wandb` is set to `True` in the hyperparameters (it is `False` by default), but it must be installed to run training.
- For the AMBS notebooks only: `pybind11` (installed automatically by the notebook) and a C++17 compiler with OpenMP support (`g++`).
- A GPU is optional: training uses CUDA when available, and the test cell always runs on CPU.

```bash
pip install numpy torch matplotlib tqdm wandb pybind11
```

---

## 1. Train the HDRL agents (`main.ipynb`)

Run the cells of `main.ipynb` in order. For a new run you normally only edit **`INSTANCE_ID`**, the first setting of the instance cell; everything else has working defaults.

1. **Instance and hyperparameters.** Set `INSTANCE_ID` (for instance `"1.5"` or `"2.9"`) and run the cell. The ID selects the problem (`1.x` = S-CLSP, Case I; `2.x` = S-CLSP-SD, Case II), builds the instance and all hyperparameters (`HYP`) and picks the matching environment and trainer. Nothing is trained in this cell.
2. **Train.** Trains the high-level and low-level agents with PPO on the selected instance.
3. **Test.** Evaluates the trained pair (see the next section).

Main settings of the instance cell:

| Setting | What it does |
|---|---|
| `INSTANCE_ID` | Paper instance to train (see the table above). |
| `WARM_START` | `True` (default): pretrain the high-level network with the AMBS heuristic before PPO. The pretrained weights are cached in `runs/<exp_name>/high_warmstart.pt` and reused in later runs. |
| `WARM_START_FORCE_RETRAIN` | `True`: ignore that cache and pretrain again. |
| `HIDDEN_SIZES_HIGH`, `HIDDEN_SIZES_LOW` | MLP widths of the two agents (default `(256, 256)`). |
| `ENV_*` | Overrides of the environment helpers. `None` = chosen automatically from the number of products K. |
| `LOW_RETURN_H`, `LOW_SUCC_H`, `LOW_REWARD_SCALE` | Window and scale of the low-level return. |

Training details (defaults): 8,000 PPO updates, episodes of 256 periods, seeds 123 (training) and 10,000 (validation). Policies are validated periodically on 50 independent runs and the best checkpoint is stored.

Everything produced by a run is written to `runs/<exp_name>/`, where `<exp_name>` encodes the instance:

- `high_joint_best.pt`, `low_joint_best.pt`: best checkpoints of the high-level and low-level agents,
- `high_warmstart.pt`: cached warm-start weights (when `WARM_START = True`),
- other training outputs (e.g. learning curves).

Approximate cost on a single Google Colab Pro instance (high-RAM runtime): about 40 minutes of warm start plus 9 hours of training for K = 10, and about 1 hour of warm start plus 18 hours of training for K = 30.

---

## 2. Test a trained model

The test cell evaluates the policy with the protocol used in the paper: **10,000 independent trajectories** of 228 periods, of which the first 128 are discarded as warm-up (costs are averaged over the remaining 100), with **greedy** action selection, test seed base 12, on CPU. It prints the mean cost per period.

**Option A: test the models of this repository**

1. Download the two files of the instance you want from `Trained Models/`, for example
   `Trained Models/Case I SCLSP/1.5/high_joint_best.pt` and `low_joint_best.pt`.
2. Copy both files **next to `main.ipynb`** (the notebook reads them from its working directory, so no path is needed).
3. In `main.ipynb` set the **same `INSTANCE_ID`** as the model and keep the default settings (the networks must have the same architecture, so do not change `HIDDEN_SIZES*`).
4. Run the instance cell, **skip the training cell**, and run the test cell.

**Option B: test the models you have just trained.** Run the three cells in order. If the two `.pt` files are not next to the notebook, the test cell uses those saved in `runs/<exp_name>/` by the training cell.

---

## 3. Benchmark heuristics

The two AMBS notebooks calibrate and evaluate the benchmark heuristic of the paper for one instance. In each notebook:

1. set `INSTANCE_ID` in the last cell (`"1.x"` in `AMBS_Heuristic_SCLSP.ipynb`, `"2.x"` in `AMBS_Heuristic_SCLSP_SD.ipynb`),
2. run the cells in order.

The first two cells write (`%writefile`, so a Jupyter or Colab kernel is needed) and compile (`g++ -O3 -fopenmp` with `pybind11`) a C++ extension, `ambs_core` or `ambs_sd_core`, in the working directory. The last cell runs the grid search over the thresholds (`Bmin`, `Hmax`, `Zmax`; plus the setup-cost penalty `lam` for AMBS-SD) with 1,000 trajectories per configuration, and then evaluates the best configuration with the same protocol as the HDRL test (10,000 trajectories, 128 + 100 periods, seed base 12). It prints the thresholds found and the average cost, broken down into holding, backorder and setup costs.

- **`AMBS_Heuristic_SCLSP.ipynb`**: AMBS heuristic for the S-CLSP (Case I).
- **`AMBS_Heuristic_SCLSP_SD.ipynb`**: AMBS-SD, the extension to sequence-dependent setups used in Case II (production sequencing by cheapest insertion and a setup-cost-aware product selection penalty `lam`). With sequence-independent setups and `lam = 0` it makes the same decisions as AMBS.

The calibrated thresholds of every instance, used by the HDRL warm start, are stored in `AMBS_PARAMS_REGISTRY` in `CLSP_HDRL/warm_start.py`.

---

## Reproducibility notes

- Seeds: training 123, validation 10,000, test 12. Global seeds and deterministic PyTorch settings are set at the start of training and testing.
- The C++ AMBS code draws one independent random stream per trajectory. The streams are reproducible but not bit-identical to those of a NumPy implementation, so small differences in the average cost are expected.
- Training results can still vary across hardware (CPU or GPU, library versions). Testing the provided `.pt` files with the default settings follows the same protocol used to obtain the HDRL costs reported in the paper.
- Trained models are stored per instance. Both files of the same instance (`high_joint_best.pt` and `low_joint_best.pt`) are needed together, and they must be used with the instance they were trained on.

---
