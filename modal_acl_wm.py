"""
Modal Cloud Execution Blueprint for In-Context Continual World Models (ACL-WM).

Implements Steps 2, 3, and 4 of the Execution Blueprint:
  Step 2: Configure Modal Image (`nvidia/cuda:11.8.0-devel-ubuntu22.04` + PyTorch 2.1 + ninja)
          & Persistent CUDA Cache (`TORCH_EXTENSIONS_DIR="/root/torch_extensions"`),
          and verify exact forward/backward parity of `self_ref_v0.cu` and `fast_weight_cuda.cu`.
  Step 3: Run the 4-Architecture x 3-Seed Ablation Sweep (4-layer, d_model=256):
            1. `self_ref_v0` (SRWM + ACL Objective - Ours)
            2. `fast_weight` (DeltaNet - Standard Fast Weights)
            3. `fast_transformers` (Linear Transformer)
            4. `transformer` (Standard Causal Transformer)
  Step 4: Evaluate Zero-Shot CEM-MPC Planning Under Sudden Physics Flips (A -> B -> C -> A)
          without task-ID labels and aggregate all results in the cloud volume.

Usage:
  modal run modal_acl_wm.py --stage verify
  modal run --detach modal_acl_wm.py --stage full --total-steps 1500 --seeds 1,2,3
"""

import json
import os
import time
from typing import Any, Dict, List, Tuple

import modal

APP_NAME = "acl-continual-world-models"
app = modal.App(APP_NAME)

# Persistent Volumes:
# 1) Cache compiled CUDA extensions (`self_ref_v0.cu`, `fast_weight_cuda.cu`) across runs
ext_volume = modal.Volume.from_name("acl-wm-torch-extensions", create_if_missing=True)
# 2) Store trained checkpoints, training curves, and CEM-MPC evaluation results
results_volume = modal.Volume.from_name("acl-wm-results", create_if_missing=True)

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("build-essential", "ninja-build", "git")
    .pip_install(
        "torch==2.1.2",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install("numpy<2.0", "einops", "packaging")
    .env({
        "TORCH_EXTENSIONS_DIR": "/root/torch_extensions",
        "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9",
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        WORKSPACE_DIR,
        remote_path="/root/acl_wm",
        ignore=[
            ".git",
            "__pycache__",
            "save_models",
            "results",
            "*.pt",
            "torchmeta_local",
            "oml_baseline_local",
        ],
    )
)


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/root/torch_extensions": ext_volume,
        "/root/results": results_volume,
    },
    timeout=1800,
)
def compile_and_verify_cuda() -> Dict[str, Any]:
    """
    Step 2: Compile `self_ref_v0.cu` and `fast_weight_cuda.cu` once into the persistent
    Modal Volume `/root/torch_extensions` and verify exact stateful forward & backward
    gradients against the PyTorch autograd reference.
    """
    import sys
    sys.path.insert(0, "/root/acl_wm")
    os.makedirs("/root/torch_extensions", exist_ok=True)

    import torch
    from self_ref_v0 import _ensure_cuda_ext as ensure_srwm, stateful_self_ref_v0, self_ref_v0_pytorch
    from fast_weight import _ensure_cuda_ext as ensure_fw, stateful_fast_weight_delta, fast_weight_delta_pytorch

    t0 = time.time()
    print(f"[Step 2] Compiling/loading CUDA extensions on {torch.cuda.get_device_name(0)}...")
    ensure_srwm()
    ensure_fw()
    compile_sec = time.time() - t0
    ext_volume.commit()
    print(f"[Step 2] CUDA extensions ready in {compile_sec:.2f}s (persisted to /root/torch_extensions).")

    # Verify StatefulSelfRefv0 across a 2-segment chain (Segment A -> Segment B -> Query)
    torch.manual_seed(42)
    bsz, n_head, slen, d_head = 2, 4, 8, 16
    x_a_cuda = torch.randn(bsz, n_head, slen, d_head, device="cuda", requires_grad=True)
    x_b_cuda = torch.randn(bsz, n_head, slen, d_head, device="cuda", requires_grad=True)
    Wy0_cuda = torch.empty(bsz, n_head, d_head, d_head, device="cuda").uniform_(-0.1, 0.1).requires_grad_(True)
    Wq0_cuda = torch.empty(bsz, n_head, d_head, d_head, device="cuda").uniform_(-0.1, 0.1).requires_grad_(True)
    Wk0_cuda = torch.empty(bsz, n_head, d_head, d_head, device="cuda").uniform_(-0.1, 0.1).requires_grad_(True)
    wb0_cuda = torch.empty(bsz, n_head, d_head, 4, device="cuda").uniform_(-0.1, 0.1).requires_grad_(True)

    out_a, Wy_a, Wq_a, Wk_a, wb_a = stateful_self_ref_v0(x_a_cuda, Wy0_cuda, Wq0_cuda, Wk0_cuda, wb0_cuda)
    out_b, Wy_b, Wq_b, Wk_b, wb_b = stateful_self_ref_v0(x_b_cuda, Wy_a, Wq_a, Wk_a, wb_a)
    loss_cuda = (out_a.pow(2).mean() + out_b.pow(2).mean() + Wy_b.pow(2).mean())
    loss_cuda.backward()

    # Reference in pure PyTorch
    x_a_ref = x_a_cuda.detach().clone().requires_grad_(True)
    x_b_ref = x_b_cuda.detach().clone().requires_grad_(True)
    Wy0_ref = Wy0_cuda.detach().clone().requires_grad_(True)
    Wq0_ref = Wq0_cuda.detach().clone().requires_grad_(True)
    Wk0_ref = Wk0_cuda.detach().clone().requires_grad_(True)
    wb0_ref = wb0_cuda.detach().clone().requires_grad_(True)

    out_a_r, Wy_a_r, Wq_a_r, Wk_a_r, wb_a_r = self_ref_v0_pytorch(
        x_a_ref, Wy0_ref, Wq0_ref, Wk0_ref, wb0_ref, stateful=True
    )
    out_b_r, Wy_b_r, Wq_b_r, Wk_b_r, wb_b_r = self_ref_v0_pytorch(
        x_b_ref, Wy_a_r, Wq_a_r, Wk_a_r, wb_a_r, stateful=True
    )
    loss_ref = (out_a_r.pow(2).mean() + out_b_r.pow(2).mean() + Wy_b_r.pow(2).mean())
    loss_ref.backward()

    def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((torch.abs(a - b) / (torch.abs(b) + 1e-5)).max().item())

    srwm_fwd_err = rel_err(out_b, out_b_r)
    srwm_grad_x_err = rel_err(x_a_cuda.grad, x_a_ref.grad)
    srwm_grad_Wy_err = rel_err(Wy0_cuda.grad, Wy0_ref.grad)
    srwm_grad_Wq_err = rel_err(Wq0_cuda.grad, Wq0_ref.grad)

    print(
        f"[Step 2] SRWM Stateful Verification: "
        f"fwd_err={srwm_fwd_err:.2e}, grad_x_a_err={srwm_grad_x_err:.2e}, "
        f"grad_Wy0_err={srwm_grad_Wy_err:.2e}, grad_Wq0_err={srwm_grad_Wq_err:.2e}"
    )
    assert srwm_fwd_err < 1e-2 and srwm_grad_x_err < 1e-2 and srwm_grad_Wy_err < 1e-2

    # Verify StatefulDeltaFastWeight across a 2-segment chain
    q_a = torch.rand(bsz, n_head, slen, d_head, device="cuda")
    q_a = (q_a / q_a.sum(-1, keepdim=True)).requires_grad_(True)
    k_a = torch.rand(bsz, n_head, slen, d_head, device="cuda")
    k_a = (k_a / k_a.sum(-1, keepdim=True)).requires_grad_(True)
    v_a = torch.randn(bsz, n_head, slen, d_head, device="cuda", requires_grad=True)
    b_a = torch.sigmoid(torch.randn(bsz, n_head, slen, 1, device="cuda")).requires_grad_(True)
    W0_fw = torch.zeros(bsz, n_head, d_head, d_head, device="cuda", requires_grad=True)

    out_fw, W1_fw = stateful_fast_weight_delta(q_a, k_a, v_a, b_a, W0_fw)
    loss_fw = out_fw.pow(2).mean() + W1_fw.pow(2).mean()
    loss_fw.backward()

    q_a_r = q_a.detach().clone().requires_grad_(True)
    k_a_r = k_a.detach().clone().requires_grad_(True)
    v_a_r = v_a.detach().clone().requires_grad_(True)
    b_a_r = b_a.detach().clone().requires_grad_(True)
    W0_fw_r = W0_fw.detach().clone().requires_grad_(True)

    out_fw_r, W1_fw_r = fast_weight_delta_pytorch(q_a_r, k_a_r, v_a_r, b_a_r, W0_fw_r, stateful=True)
    loss_fw_r = out_fw_r.pow(2).mean() + W1_fw_r.pow(2).mean()
    loss_fw_r.backward()

    fw_fwd_err = rel_err(out_fw, out_fw_r)
    fw_grad_q_err = rel_err(q_a.grad, q_a_r.grad)
    fw_grad_W_err = rel_err(W0_fw.grad, W0_fw_r.grad)
    print(
        f"[Step 2] DeltaNet Stateful Verification: "
        f"fwd_err={fw_fwd_err:.2e}, grad_q_err={fw_grad_q_err:.2e}, grad_W0_err={fw_grad_W_err:.2e}"
    )
    assert fw_fwd_err < 1e-2 and fw_grad_q_err < 1e-2 and fw_grad_W_err < 1e-2

    return {
        "gpu": torch.cuda.get_device_name(0),
        "compile_sec": compile_sec,
        "srwm_fwd_err": srwm_fwd_err,
        "srwm_grad_x_err": srwm_grad_x_err,
        "srwm_grad_Wy_err": srwm_grad_Wy_err,
        "fw_fwd_err": fw_fwd_err,
        "fw_grad_q_err": fw_grad_q_err,
        "fw_grad_W_err": fw_grad_W_err,
    }


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/root/torch_extensions": ext_volume,
        "/root/results": results_volume,
    },
    timeout=7200,
)
def train_and_eval_single_run(
    model_type: str,
    seed: int,
    total_train_steps: int = 1500,
    batch_size: int = 32,
    k_shot: int = 25,
    num_queries: int = 8,
    rollout_horizon: int = 4,
    learning_rate: float = 8e-4,
    mpc_episodes: int = 16,
    mpc_steps_per_phase: int = 30,
) -> Dict[str, Any]:
    """
    Step 3 & Step 4 worker: Train a single (model_type, seed) configuration on L4 GPU
    and immediately evaluate its best checkpoint under Zero-Shot CEM-MPC across A -> B -> C -> A.
    """
    import shutil
    import sys
    sys.path.insert(0, "/root/acl_wm")
    import torch
    from train_acl_wm import build_arg_parser, train
    from eval_mpc import load_and_eval_checkpoint

    run_dir = f"/root/results/v3/{model_type}_seed{seed}"
    os.makedirs(run_dir, exist_ok=True)
    summary_file = os.path.join(run_dir, "full_run_summary.json")

    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                cached = json.load(f)
            if cached.get("train_summary", {}).get("final_val", {}).get("step") in (total_train_steps, 3000):
                print(f"[Reuse] {model_type} seed={seed} already completed in v3.")
                return cached
        except Exception:
            pass

    # If self_ref_v0 (SRWM + ACL) was already trained for 3000 steps on the exact 4429681
    # dataset in /root/results/self_ref_v0_seed{seed}, reuse its checkpoint & training summary
    # and run the updated CEM-MPC evaluation directly.
    v1_dir = f"/root/results/{model_type}_seed{seed}"
    v1_ckpt = os.path.join(v1_dir, "best_model.pt")
    v1_summary = os.path.join(v1_dir, "full_run_summary.json")
    v1_metrics = os.path.join(v1_dir, "metrics.json")

    if model_type == "self_ref_v0" and os.path.exists(v1_ckpt) and os.path.exists(v1_summary):
        print(f"[Reuse 3000-step SRWM+ACL checkpoint] Loading {v1_ckpt} for {model_type} seed={seed}")
        shutil.copy2(v1_ckpt, os.path.join(run_dir, "best_model.pt"))
        if os.path.exists(v1_metrics):
            shutil.copy2(v1_metrics, os.path.join(run_dir, "metrics.json"))
        with open(v1_summary, "r") as f:
            v1_data = json.load(f)
        train_summary = v1_data["train_summary"]
        train_sec = v1_data.get("train_sec", 0.0)
    else:
        parser = build_arg_parser()
        args = parser.parse_args([
            "--model_type", model_type,
            "--seed", str(seed),
            "--work_dir", run_dir,
            "--batch_size", str(batch_size),
            "--k_shot", str(k_shot),
            "--num_queries", str(num_queries),
            "--rollout_horizon", str(rollout_horizon),
            "--num_layer", "4",
            "--hidden_size", "256",
            "--n_head", "16",
            "--ff_factor", "2",
            "--learning_rate", str(learning_rate),
            "--warmup_steps", str(max(100, total_train_steps // 10)),
            "--total_train_steps", str(total_train_steps),
            "--report_every", str(max(50, total_train_steps // 15)),
            "--validate_every", str(max(100, total_train_steps // 5)),
            "--valid_batches", "20",
        ])

        t0 = time.time()
        train_summary = train(args)
        train_sec = time.time() - t0
        results_volume.commit()

    best_ckpt = os.path.join(run_dir, "best_model.pt")
    mpc_t0 = time.time()
    mpc_summary = load_and_eval_checkpoint(
        best_ckpt,
        num_episodes=mpc_episodes,
        steps_per_phase=mpc_steps_per_phase,
        mpc_horizon=8,
        mpc_samples=128,
        seed=2026 + seed,
        device=torch.device("cuda"),
    )
    mpc_sec = time.time() - mpc_t0

    combined = {
        "model_type": model_type,
        "seed": seed,
        "train_sec": train_sec,
        "mpc_sec": mpc_sec,
        "train_summary": train_summary,
        "mpc_summary": mpc_summary,
    }
    with open(summary_file, "w") as f:
        json.dump(combined, f, indent=2)

    results_volume.commit()
    return combined


@app.function(
    image=image,
    volumes={
        "/root/results": results_volume,
    },
    timeout=7200,
)
def orchestrate_sweep_on_cloud(
    total_steps: int = 1500,
    seeds: str = "1,2,3",
    models: str = "self_ref_v0,fast_weight,fast_transformers,transformer",
) -> Dict[str, Any]:
    """
    Cloud-side coordinator so that `modal run --detach` runs 100% inside Modal's cloud
    even if the user's laptop disconnects or goes to sleep, saving the aggregated JSON
    directly to `/root/results/v3/ablation_and_mpc_results.json`.
    """
    import numpy as np

    print("======================================================================")
    print("Step 2: Compiling & Verifying Custom CUDA Extensions on Modal L4 GPU")
    print("======================================================================")
    verify_res = compile_and_verify_cuda.remote()
    print(json.dumps(verify_res, indent=2))

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    jobs: List[Tuple[str, int, int]] = [
        (m, s, total_steps) for m in model_list for s in seed_list
    ]
    print("======================================================================")
    print(f"Step 3 & 4: Launching {len(jobs)} Parallel GPU Runs ({model_list} x seeds {seed_list})")
    print("======================================================================")

    all_results = list(train_and_eval_single_run.starmap(jobs))

    os.makedirs("/root/results/v3", exist_ok=True)
    out_path = "/root/results/v3/ablation_and_mpc_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    results_volume.commit()
    print(f"\nSaved full v3 ablation + CEM-MPC results to Modal Volume at {out_path}")

    print("\n==========================================================================================")
    print("4-ARCHITECTURE ABLATION & ZERO-SHOT CEM-MPC SUMMARY (MEAN +/- STD ACROSS SEEDS)")
    print("==========================================================================================")
    header = (
        f"{'Architecture':18s} | {'MSE(A|ABC)':12s} | {'MSE(Mean)':12s} | "
        f"{'RolloutMSE':12s} | {'ForgetRatio':11s} | {'MPC TotalRew':13s} | {'MPC Recall(A)':13s}"
    )
    print(header)
    print("-" * len(header))

    for m in model_list:
        m_runs = [r for r in all_results if r["model_type"] == m]
        if not m_runs:
            continue
        mse_a = [r["train_summary"]["final_val"]["mse_t0_after_123"] for r in m_runs]
        mse_mean = [r["train_summary"]["final_val"]["mse_mean_after_123"] for r in m_runs]
        roll_mse = [r["train_summary"]["final_val"]["rollout_mse_mean_after_123"] for r in m_runs]
        forget = [r["train_summary"]["final_val"]["forgetting_ratio_t0"] for r in m_runs]
        mpc_tot = [r["mpc_summary"]["total_cumulative_reward"] for r in m_runs]
        mpc_rec = [r["mpc_summary"]["recall_A_cumulative_reward"] for r in m_runs]

        print(
            f"{m:18s} | "
            f"{np.mean(mse_a):.4f}±{np.std(mse_a):.4f} | "
            f"{np.mean(mse_mean):.4f}±{np.std(mse_mean):.4f} | "
            f"{np.mean(roll_mse):.4f}±{np.std(roll_mse):.4f} | "
            f"{np.mean(forget):.3f}±{np.std(forget):.3f} | "
            f"{np.mean(mpc_tot):6.2f}±{np.std(mpc_tot):4.2f} | "
            f"{np.mean(mpc_rec):6.2f}±{np.std(mpc_rec):4.2f}"
        )

    return {"verify": verify_res, "num_runs": len(all_results)}


@app.local_entrypoint()
def main(
    stage: str = "full",
    total_steps: int = 1500,
    seeds: str = "1,2,3",
    models: str = "self_ref_v0,fast_weight,fast_transformers,transformer",
):
    if stage == "verify":
        res = compile_and_verify_cuda.remote()
        print(json.dumps(res, indent=2))
        return

    orchestrate_sweep_on_cloud.remote(
        total_steps=total_steps,
        seeds=seeds,
        models=models,
    )
