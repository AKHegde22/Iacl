"""
Download all experiment artifacts (metrics.json, full_run_summary.json, and aggregated JSON)
from the Modal Volume `acl-wm-results` into `results/` and regenerate summary plots.
"""

import argparse
import json
import os
import shutil
import modal
import numpy as np


def download_v3_and_plot(include_checkpoints: bool = False):
    os.makedirs("results/runs", exist_ok=True)
    # Preserve the v1 controlled ablation (where all 4 architectures used ACL objective)
    if os.path.exists("results/ablation_and_mpc_results.json") and not os.path.exists(
        "results/ablation_acl_all_architectures.json"
    ):
        shutil.copy2(
            "results/ablation_and_mpc_results.json",
            "results/ablation_acl_all_architectures.json",
        )

    vol = modal.Volume.from_name("acl-wm-results")
    entries = list(vol.listdir("/v3", recursive=True))
    print(f"Found {len(entries)} entries in Modal volume '/v3'.")

    downloaded = 0
    for entry in entries:
        rel_path = entry.path.lstrip("/")
        if rel_path.startswith("v3/"):
            sub_rel = rel_path[len("v3/"):]
        else:
            sub_rel = rel_path

        if hasattr(entry, "type") and int(entry.type) == 2:
            continue
        if not include_checkpoints and sub_rel.endswith(".pt"):
            continue

        if sub_rel == "ablation_and_mpc_results.json":
            local_path = os.path.join("results", "ablation_and_mpc_results.json")
        else:
            local_path = os.path.join("results/runs", sub_rel)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {entry.path} -> {local_path} ...")
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        downloaded += 1

    print(f"Successfully downloaded {downloaded} v3 files.")
    generate_plots()


def generate_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open("results/ablation_and_mpc_results.json", "r") as f:
        all_results = json.load(f)

    models = [
        ("self_ref_v0", "SRWM + ACL (Ours)", "#d62728"),
        ("fast_weight", "DeltaNet (Standard Fast Weights)", "#1f77b4"),
        ("fast_transformers", "Linear Transformer", "#2ca02c"),
        ("transformer", "Standard Causal Transformer", "#9467bd"),
    ]

    # 1. Training & Validation Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)
    for m_key, m_label, color in models:
        steps_list = []
        mse_a_list = []
        mse_mean_list = []
        forget_list = []
        for seed in [1, 2, 3]:
            m_file = os.path.join("results/runs", f"{m_key}_seed{seed}", "metrics.json")
            if not os.path.exists(m_file):
                continue
            with open(m_file, "r") as f:
                m_data = json.load(f)
            hist = m_data["history"]
            steps = [h["step"] / h[-1]["step"] * 100.0 if False else (i + 1) / len(hist) * 100.0 for i, h in enumerate(hist)]
            steps_list.append(steps)
            mse_a_list.append([h["mse_t0_after_123"] for h in hist])
            mse_mean_list.append([h["mse_mean_after_123"] for h in hist])
            forget_list.append([h["forgetting_ratio_t0"] for h in hist])

        if not mse_a_list:
            continue
        x = np.array(steps_list[0])
        mse_a_arr = np.array(mse_a_list)
        mse_mean_arr = np.array(mse_mean_list)
        forget_arr = np.array(forget_list)

        for ax, arr in zip(axes, [mse_a_arr, mse_mean_arr, forget_arr]):
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(x, mean, label=m_label, color=color, linewidth=2.2, marker="o", markersize=4)
            ax.fill_between(x, np.maximum(mean - std, 1e-4), mean + std, color=color, alpha=0.18)

    axes[0].set_title("Backward-Transfer MSE(A | A,B,C) ↓", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Training Progress (%)")
    axes[0].set_ylabel("1-Step State-Delta MSE (log scale)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("All-Mode Mean MSE(A,B,C | A,B,C) ↓", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Training Progress (%)")
    axes[1].set_ylabel("Mean 1-Step MSE (log scale)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("In-Context Forgetting Ratio (MSE(A|ABC) / MSE(A|A)) ↓", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Training Progress (%)")
    axes[2].set_ylabel("Forgetting Ratio (log scale, 1.0 = No Forgetting)")
    axes[2].set_yscale("log")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Ideal (1.0x)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best", fontsize=8.5)

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=160)
    plt.close(fig)

    # 2. Live Zero-Shot CEM-MPC Switch Boundary Curves (A -> B -> C -> A)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), dpi=160, sharex=True)
    for m_key, m_label, color in models:
        m_runs = [r for r in all_results if r["model_type"] == m_key]
        if not m_runs:
            continue
        mse_curves = np.array([r["mpc_summary"]["step_mse_curve"] for r in m_runs])
        rew_curves = np.array([r["mpc_summary"]["step_reward_curve"] for r in m_runs])
        t_steps = np.arange(1, mse_curves.shape[1] + 1)

        mse_mean = mse_curves.mean(axis=0)
        mse_std = mse_curves.std(axis=0)
        axes[0].plot(t_steps, mse_mean, label=m_label, color=color, linewidth=2.0)
        axes[0].fill_between(t_steps, np.maximum(mse_mean - mse_std, 1e-4), mse_mean + mse_std, color=color, alpha=0.15)

        rew_mean = rew_curves.mean(axis=0)
        rew_std = rew_curves.std(axis=0)
        axes[1].plot(t_steps, rew_mean, label=m_label, color=color, linewidth=2.0)
        axes[1].fill_between(t_steps, rew_mean - rew_std, rew_mean + rew_std, color=color, alpha=0.15)

    phase_bounds = [30, 60, 90]
    phase_names = [
        (15, "Phase 1: Mode A\n(Inverted Gravity)"),
        (45, "Phase 2: Mode B\n(Cross-Wind Force)"),
        (75, "Phase 3: Mode C\n(Inverted Motor Torque)"),
        (105, "Phase 4: Recall Mode A\n(Zero-Shot Backward Transfer)"),
    ]
    for ax in axes:
        for b in phase_bounds:
            ax.axvline(b + 0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.grid(True, alpha=0.3)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Online 1-Step Prediction MSE (log)", fontsize=10.5)
    axes[0].set_title(
        "Live Zero-Shot CEM-MPC Under Sudden Physics Switches (A → B → C → A, No Task-ID Labels)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].legend(loc="upper left", fontsize=9)

    for xpos, label in phase_names:
        axes[0].text(
            xpos,
            axes[0].get_ylim()[1] * 0.45,
            label,
            ha="center",
            va="top",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    axes[1].set_ylabel("Step Control Reward (CEM-MPC)", fontsize=10.5)
    axes[1].set_xlabel("Live Episode Timestep t (Switches at t = 30, 60, 90)", fontsize=10.5)
    axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1.0)

    plt.tight_layout()
    plt.savefig("results/mpc_switch_curves.png", dpi=160)
    plt.close(fig)
    print("Generated results/training_curves.png and results/mpc_switch_curves.png.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-checkpoints", action="store_true")
    args = parser.parse_args()
    download_v3_and_plot(include_checkpoints=args.with_checkpoints)
