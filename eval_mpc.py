"""
Zero-Shot CEM-MPC Planning & Switch-Boundary Evaluation for Continual World Models.

Evaluates frozen meta-trained checkpoints in live test episodes where the physics
of the environment switch across Task A -> B -> C -> A without task-ID labels.
Uses Cross-Entropy Method (CEM) Model Predictive Control inside the learned world
model (with fast weights frozen during planning rollouts, beta_t = 0) and logs:
  - Per-step state-prediction MSE across all switch boundaries (A -> B -> C -> A)
  - Cumulative control reward and upright stabilization rate per phase
  - Immediate zero-shot recall MSE & MPC reward upon switching back C -> A
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from continual_physics_dataset import VectorizedCartPolePhysics
from model_acl_wm import ContinualWorldModel


@torch.no_grad()
def cem_mpc_plan(
    model: ContinualWorldModel,
    current_aug_state: torch.Tensor,
    fast_weight_state: Any,
    horizon: int = 8,
    num_Flag_samples: int = 128,
    num_elites: int = 16,
    cem_iters: int = 3,
    gamma: float = 0.98,
) -> torch.Tensor:
    """
    Batched Cross-Entropy Method (CEM) Model Predictive Control inside the frozen
    world model. Evaluates `num_Flag_samples` candidate action trajectories of length
    `horizon` across a batch of `B` parallel environments simultaneously on GPU.

    Args:
        current_aug_state: (B, 11) current augmented Markov state
        fast_weight_state: model fast-weight state for batch size B
    Returns:
        best_action: (B, 1) in [-1, 1]
    """
    bsz = current_aug_state.size(0)
    device = current_aug_state.device
    dtype = current_aug_state.dtype

    # Action distribution parameters: (num_samples, horizon, B, 1)
    mean = torch.zeros(1, horizon, bsz, 1, device=device, dtype=dtype)
    std = torch.ones(1, horizon, bsz, 1, device=device, dtype=dtype) * 0.75

    # Expand current state to (num_samples, B, 11) for parallel query/rollout
    init_states = current_aug_state.unsqueeze(0).expand(num_Flag_samples, bsz, -1).contiguous()

    for _ in range(cem_iters):
        noise = torch.randn(num_Flag_samples, horizon, bsz, 1, device=device, dtype=dtype)
        cand_actions = torch.clamp(mean + std * noise, -1.0, 1.0)

        # Roll out all candidate trajectories using frozen fast weights (beta_t = 0)
        cur_s = init_states
        total_returns = torch.zeros(num_Flag_samples, bsz, device=device, dtype=dtype)
        discount = 1.0

        for h in range(horizon):
            act_h = cand_actions[:, h]  # (num_samples, B, 1)
            pred_delta = model.query(cur_s, act_h, fast_weight_state)
            next_s = VectorizedCartPolePhysics.advance_state_from_pred(cur_s, act_h, pred_delta)
            step_rew = VectorizedCartPolePhysics.compute_mpc_reward(next_s, act_h)
            total_returns = total_returns + discount * step_rew
            discount *= gamma
            cur_s = next_s

        # Select top `num_elites` trajectories per batch element b
        # total_returns: (num_samples, B)
        k_elites = min(num_elites, num_Flag_samples)
        _, elite_idx = torch.topk(total_returns, k=k_elites, dim=0, largest=True)
        # Gather elite actions: elite_idx (k_elites, B) -> (k_elites, horizon, B, 1)
        idx_exp = elite_idx.unsqueeze(1).unsqueeze(-1).expand(k_elites, horizon, bsz, 1)
        elite_actions = torch.gather(cand_actions, dim=0, index=idx_exp)

        mean = elite_actions.mean(dim=0, keepdim=True)
        std = elite_actions.std(dim=0, unbiased=False, keepdim=True).clamp(min=0.05)

    return mean[0, 0]  # (B, 1)


@torch.no_grad()
def evaluate_live_mpc_episode(
    model: ContinualWorldModel,
    num_episodes: int = 16,
    context_k: int = 25,
    steps_per_phase: int = 30,
    mpc_horizon: int = 8,
    mpc_samples: int = 128,
    mpc_elites: int = 16,
    mpc_iters: int = 3,
    seed: int = 2026,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    """
    Evaluate a frozen meta-trained model in a live test episode where the environment
    switches across Task A -> B -> C -> A without task-ID labels.

    Protocol:
      1. Context Adaptation Stream (A -> B -> C):
         The model observes K transitions from Mode A, then Mode B, then Mode C,
         building fast-weight state W_A -> W_{A,B} -> W_{A,B,C} in zero gradient steps.
      2. Live Closed-Loop Control & Switch-Boundary Evaluation (A -> B -> C -> A):
         The agent is dropped into a live 4-phase control episode that switches
         suddenly across Mode A -> Mode B -> Mode C -> Mode A (recall) every
         `steps_per_phase` steps without task-ID labels.
         At every timestep t, CEM-MPC plans actions inside the learned world model
         and we log state-prediction MSE and control reward.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()

    physics = VectorizedCartPolePhysics(dt=0.04)
    bsz = num_episodes

    # Sample episode-specific orthogonal field basis and parameters for Mode A, Mode B, Mode C
    basis = physics.sample_episode_field_basis(bsz, device)
    params_A = physics.sample_mode_params(bsz, 'A', device, field_vec=basis[:, :, 0])
    params_B = physics.sample_mode_params(bsz, 'B', device, field_vec=basis[:, :, 1])
    params_C = physics.sample_mode_params(bsz, 'C', device, field_vec=basis[:, :, 2])

    # 1. Sequential In-Context Adaptation across Mode A -> Mode B -> Mode C
    ctx_s_A, ctx_a_A, ctx_ds_A = physics.sample_trajectory_segment(bsz, context_k, params_A, device)
    ctx_s_B, ctx_a_B, ctx_ds_B = physics.sample_trajectory_segment(bsz, context_k, params_B, device)
    ctx_s_C, ctx_a_C, ctx_ds_C = physics.sample_trajectory_segment(bsz, context_k, params_C, device)

    _, state_A = model.ingest_context(ctx_s_A, ctx_a_A, ctx_ds_A, state=None)
    _, state_AB = model.ingest_context(ctx_s_B, ctx_a_B, ctx_ds_B, state=state_A)
    _, state_ABC = model.ingest_context(ctx_s_C, ctx_a_C, ctx_ds_C, state=state_AB)

    # 2. Live 4-Phase Closed-Loop MPC Episode: A -> B -> C -> A
    # Notice: during Phases 1 (A), 2 (B), 3 (C), and 4 (Recall A), the model uses
    # its accumulated in-context memory W_{A,B,C} (and also we track online-updated state)
    phases: List[Tuple[str, Dict[str, torch.Tensor], Any]] = [
        ('Phase1_A', params_A, state_A),
        ('Phase2_B', params_B, state_AB),
        ('Phase3_C', params_C, state_ABC),
        ('Phase4_Recall_A', params_A, state_ABC),  # Tests backward transfer & recall of A after B & C!
    ]

    step_mse_curve: List[float] = []
    step_reward_curve: List[float] = []
    phase_summaries: Dict[str, Dict[str, float]] = {}

    for phase_name, params_m, phase_fw_state in phases:
        # Start near upright to test continuous stabilization under sudden physics flips
        aug_state = physics.sample_initial_augmented_state(bsz, params_m, device, near_upright=True)
        phase_mses = []
        phase_rewards = []
        phase_upright = []

        for t in range(steps_per_phase):
            # Plan action via CEM-MPC inside frozen world model
            action = cem_mpc_plan(
                model,
                aug_state,
                phase_fw_state,
                horizon=mpc_horizon,
                num_Flag_samples=mpc_samples,
                num_elites=mpc_elites,
                cem_iters=mpc_iters,
            )

            # Predict 1-step delta using world model before observing ground truth
            pred_delta = model.query(aug_state.unsqueeze(0), action.unsqueeze(0), phase_fw_state).squeeze(0)

            # Step true environment
            next_aug_state, true_delta = physics.step_augmented(aug_state, action, params_m)
            step_mse = F.mse_loss(pred_delta, true_delta).item()
            step_rew = physics.compute_mpc_reward(next_aug_state, action).mean().item()
            is_upright = (next_aug_state[:, 3] > 0.75).float().mean().item()  # cos(theta) > 0.75

            phase_mses.append(step_mse)
            phase_rewards.append(step_rew)
            phase_upright.append(is_upright)

            step_mse_curve.append(step_mse)
            step_reward_curve.append(step_rew)
            aug_state = next_aug_state

        phase_summaries[phase_name] = {
            'mean_mse': float(np.mean(phase_mses)),
            'Boundary_first5_mse': float(np.mean(phase_mses[:5])),
            'cumulative_reward': float(np.sum(phase_rewards)),
            'mean_step_reward': float(np.mean(phase_rewards)),
            'upright_rate': float(np.mean(phase_upright)),
        }

    total_cum_reward = float(np.sum(step_reward_curve))
    recall_a_mse = phase_summaries['Phase4_Recall_A']['mean_mse']
    initial_a_mse = phase_summaries['Phase1_A']['mean_mse']
    mpc_forgetting_ratio = recall_a_mse / max(initial_a_mse, 1e-8)

    return {
        'model_type': model.model_type,
        'canonical_type': model.canonical_type,
        'total_cumulative_reward': total_cum_reward,
        'recall_A_cumulative_reward': phase_summaries['Phase4_Recall_A']['cumulative_reward'],
        'recall_A_mse': recall_a_mse,
        'initial_A_mse': initial_a_mse,
        'mpc_forgetting_ratio': mpc_forgetting_ratio,
        'phase_summaries': phase_summaries,
        'step_mse_curve': step_mse_curve,
        'step_reward_curve': step_reward_curve,
    }


def load_and_eval_checkpoint(
    ckpt_path: str,
    num_episodes: int = 16,
    steps_per_phase: int = 30,
    mpc_horizon: int = 8,
    mpc_samples: int = 128,
    seed: int = 2026,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt['args']
    model = ContinualWorldModel(
        model_type=saved_args['model_type'],
        state_dim=VectorizedCartPolePhysics.STATE_DIM,
        action_dim=VectorizedCartPolePhysics.ACTION_DIM,
        delta_dim=VectorizedCartPolePhysics.PHYS_DIM,
        hidden_size=saved_args.get('hidden_size', 256),
        num_layers=saved_args.get('num_layer', 4),
        num_head=saved_args.get('n_head', 16),
        ff_factor=saved_args.get('ff_factor', 2),
        dropout=0.0,
        use_ln=not saved_args.get('not_use_ln', False),
        use_input_softmax=saved_args.get('use_input_softmax', False),
        beta_init=saved_args.get('srwm_beta_init', 0.0),
        init_scaler=saved_args.get('srwm_init_scaler', 1.0),
        q_init_scaler=saved_args.get('srwm_q_init_scaler', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    res = evaluate_live_mpc_episode(
        model,
        num_episodes=num_episodes,
        context_k=saved_args.get('k_shot', 25),
        steps_per_phase=steps_per_phase,
        mpc_horizon=mpc_horizon,
        mpc_samples=mpc_samples,
        seed=seed,
        device=device,
    )
    res['ckpt_path'] = ckpt_path
    res['train_seed'] = saved_args.get('seed', 1)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CEM-MPC under sudden physics flips')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--num_episodes', type=int, default=16)
    parser.add_argument('--steps_per_phase', type=int, default=30)
    parser.add_argument('--mpc_horizon', type=int, default=8)
    parser.add_argument('--mpc_samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    results = load_and_eval_checkpoint(
        args.ckpt_path,
        num_episodes=args.num_episodes,
        steps_per_phase=args.steps_per_phase,
        mpc_horizon=args.mpc_horizon,
        mpc_samples=args.mpc_samples,
        seed=args.seed,
        device=device,
    )
    print(json.dumps({k: v for k, v in results.items() if 'curve' not in k}, indent=2))
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
