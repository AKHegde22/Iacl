# In-Context Continual World Models (`ACL-WM`) — Experimental Results Summary

## 1. Main 4-Architecture $\times$ 3-Seed Ablation & Zero-Shot CEM-MPC Benchmark

Each architecture is identically scaled ($4\text{ layers}, d_{\text{model}}=256, n_{\text{head}}=16, d_{\text{head}}=16$) and evaluated across 3 random seeds (`seed=1, 2, 3`) on sequential 3-mode non-stationary CartPole/Pendulum physics transitions ($K=25$ transitions per mode across **Mode A**: Inverted Gravity + High Damping, $\kappa > 0 \to$ **Mode B**: Strong Lateral Cross-Wind + Light Pole $\to$ **Mode C**: Inverted Motor Torque $\kappa < 0$ + Elastic Spring Field) without task-ID labels.

| Architecture (`model_type`) | Params | Backward-Transfer `MSE(A \| A,B,C)` $\downarrow$ | All-Mode `MSE(Mean \| A,B,C)` $\downarrow$ | 4-Step `RolloutMSE` $\downarrow$ | `ForgetRatio(A)` $\downarrow$ | Live CEM-MPC `Total Reward` $\uparrow$ | Live CEM-MPC `Recall(A) Reward` $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`self_ref_v0` (SRWM + ACL Objective — Ours)** | **1.48M** | **`0.0263 ± 0.0014`** | **`0.0324 ± 0.0010`** | **`0.0017 ± 0.0001`** | **`1.158 ± 0.008`** | **`+23.90 ± 4.51`** | **`+20.62 ± 0.52`** |
| `fast_weight` (DeltaNet — Standard Fast Weights) | 2.22M | `20.3652 ± 0.0493` | `12.2573 ± 0.0465` | `0.6804 ± 0.0008` | `1363.206 ± 29.336` | `-103.14 ± 21.62` | `-108.12 ± 19.28` |
| `fast_transformers` (Linear Transformer) | 2.21M | `20.3773 ± 0.0458` | `12.2589 ± 0.0478` | `0.6812 ± 0.0028` | `747.683 ± 35.064` | `-106.89 ± 13.96` | `-115.03 ± 13.66` |
| `transformer` (Standard Causal Transformer, Sliding KV) | 2.21M | `20.4008 ± 0.0879` | `12.2617 ± 0.0525` | `0.6801 ± 0.0025` | `1351.024 ± 65.308` | `-109.57 ± 12.43` | `-116.21 ± 12.68` |

---

## 2. Key Scientific Findings

1. **Elimination of In-Context Catastrophic Forgetting (`774x` Lower Backward-Transfer MSE)**:
   - Standard in-context sequence learners (`fast_weight`, `fast_transformers`, `transformer`) adapt accurately to whichever physics mode is currently in context (`MSE(A|A) = 0.015–0.027`, `MSE(C|A,B,C) = 0.024–0.026`), but **catastrophically overwrite** earlier physics modes when new dynamical regimes arrive (`MSE(A|A,B,C) = 20.37–20.40`, `ForgetRatio(A) = 747.7x–1363.2x`).
   - By contrast, **`SRWM + ACL Objective` (`self_ref_v0`)** meta-learns to partition its fast-weight matrix $\mathbf{W}_{A,B,C}$ across sequential dynamical regimes in zero gradient steps at test time, achieving **`MSE(A|A,B,C) = 0.0263 ± 0.0014`** (**774$\times$ lower error** than standard baselines) and a near-ideal **`ForgetRatio(A) = 1.158 ± 0.008`** (while using **33% fewer parameters**, `1.48M` vs. `2.21M–2.22M`).

2. **Zero-Shot Closed-Loop CEM-MPC Under Sudden Physics Flips ($A \to B \to C \to A$)**:
   - In live 120-step closed-loop control episodes switching across `Phase 1 (Mode A)` $\to$ `Phase 2 (Mode B)` $\to$ `Phase 3 (Mode C, inverted motor torque)` $\to$ `Phase 4 (Recall Mode A)` without task-ID labels:
     - **`SRWM + ACL` (`self_ref_v0`)** retains Mode A's positive motor-torque polarity ($\kappa_A > 0$) inside $\mathbf{W}_{A,B,C}$ after adapting to Mode C's inverted motor-torque polarity ($\kappa_C < 0$), enabling CEM-MPC to drive the displaced cart ($|x_0| \in [0.75, 1.05]\text{ m}$) back to the origin and maintain **`100%` upright stabilization** in `Phase 4 (Recall A)` with **`+20.62 ± 0.52` Recall(A) reward** and **`+23.90 ± 4.51` total episode reward**.
     - Standard baselines (`fast_weight`, `fast_transformers`, `transformer`), whose fast weights were overwritten by Mode C's inverted motor gain ($\kappa_C < 0$), plan actions with **inverted polarity** upon returning to Mode A in `Phase 4`, accelerating the cart away from the origin and collapsing to **`-108.12` to `-116.21` Recall(A) reward**.

3. **High-Interference Correlated Sensory Stress Test ($\cos(\boldsymbol{\xi}_A, \boldsymbol{\xi}_B) \approx 0.80$)**:
   - Even when the environmental regime vectors share a dominant episode carrier $\mathbf{u}_{\text{shared}} + 0.45 \mathbf{u}_m$ (`v2` stress test), `SRWM + ACL` (`self_ref_v0`) still achieves **2.9$\times$ lower backward-transfer MSE** (`13.15 ± 0.07` vs. `37.84–38.03`) and **8$\times$–16$\times$ lower forgetting ratio** (`70.6` vs. `561.1–1158.4`) than `fast_weight`, `fast_transformers`, and `transformer`.

4. **Controlled Meta-Objective Ablation (Applying $\mathcal{L}_{\text{ACL-WM}}$ Across All Architectures)**:
   - When our proposed backward-transfer meta-objective $\mathcal{L}_{\text{ACL-WM}}$ is applied to train all four architectures (`results/ablation_acl_all_architectures.json`), `ForgetRatio(A)` drops from `748x–1363x` down to `1.096x–1.184x` across all architectures (`self_ref_v0`: `1.096 ± 0.058`, `fast_weight`: `1.148 ± 0.044`, `fast_transformers`: `1.184 ± 0.053`, `transformer`: `1.132 ± 0.049`), confirming that the $\mathcal{L}_{\text{ACL-WM}}$ meta-objective is essential for preventing in-context catastrophic forgetting in continual world models.

---

## 3. Generated Artifacts & Plots
- [results/ablation_and_mpc_results.json](file:///Users/akshaykumar/Documents/Projects/Iacl/results/ablation_and_mpc_results.json): Full 4-architecture $\times$ 3-seed metrics and per-step CEM-MPC trajectories.
- [results/ablation_acl_all_architectures.json](file:///Users/akshaykumar/Documents/Projects/Iacl/results/ablation_acl_all_architectures.json): Controlled ablation training all 4 architectures with the $\mathcal{L}_{\text{ACL-WM}}$ objective.
- [results/training_curves.png](file:///Users/akshaykumar/Documents/Projects/Iacl/results/training_curves.png): Validation curves for backward-transfer `MSE(A|ABC)`, all-mode `MSE(Mean)`, and `ForgetRatio(A)`.
- [results/mpc_switch_curves.png](file:///Users/akshaykumar/Documents/Projects/Iacl/results/mpc_switch_curves.png): Live closed-loop CEM-MPC prediction MSE and step control reward across $A \to B \to C \to A$ switch boundaries.
