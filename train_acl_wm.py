"""
Meta-Training Pipeline for In-Context Continual World Models (ACL-WM).

Trains 4-layer (d_model=256) sequence models across 3 sequential physics modes
(Mode A -> Mode B -> Mode C) using the ACL-WM meta-objective:
  L_{ACL-WM} = sum_{m in {A,B,C}} || hat{ds}_q^{(m)}(W_{A,B,C}) - ds_q^{(m)} ||_2^2
               + lambda_{multi-step} L_{rollout}
               + alpha_{inter} (L_A(W_A) + L_B(W_{A,B}) + L_A(W_{A,B}))
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from continual_physics_dataset import ContinualPhysicsDataset
from model_acl_wm import ContinualWorldModel, MODEL_TYPE_ALIASES


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_acl_wm_loss(
    model: ContinualWorldModel,
    episode_batch: Dict[str, Dict[str, torch.Tensor]],
    lambda_multistep: float = 0.5,
    alpha_intermediate: float = 0.5,
    use_acl: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Execute sequential in-context adaptation across Task 0 -> Task 1 -> Task 2
    and compute the ACL-WM meta-objective with frozen fast-weight queries (beta_t = 0).
    """
    t0 = episode_batch['task_0']
    t1 = episode_batch['task_1']
    t2 = episode_batch['task_2']

    # =========================================================================
    # Phase 1: Ingest Context from Physics Mode 1 -> W_1
    # =========================================================================
    _, state_1 = model.ingest_context(
        t0['ctx_states'], t0['ctx_actions'], t0['ctx_deltas'], state=None
    )
    state_1_q = model.clone_state(state_1)
    pred_t0_after_1 = model.query(t0['query_states'], t0['query_actions'], state_1_q)
    loss_t0_after_1 = F.mse_loss(pred_t0_after_1, t0['query_deltas'])

    # =========================================================================
    # Phase 2: Ingest Context from Physics Mode 2 -> W_{1,2}
    # =========================================================================
    _, state_12 = model.ingest_context(
        t1['ctx_states'], t1['ctx_actions'], t1['ctx_deltas'], state=state_1
    )
    state_12_q = model.clone_state(state_12)
    # Forward/current adaptation on Mode 2
    pred_t1_after_12 = model.query(t1['query_states'], t1['query_actions'], state_12_q)
    loss_t1_after_12 = F.mse_loss(pred_t1_after_12, t1['query_deltas'])

    # Backward transfer (anti-forgetting) on Mode 1 after Mode 2
    pred_t0_after_12 = model.query(t0['query_states'], t0['query_actions'], state_12_q)
    loss_t0_after_12 = F.mse_loss(pred_t0_after_12, t0['query_deltas'])

    # =========================================================================
    # Phase 3: Ingest Context from Physics Mode 3 -> W_{1,2,3} (W_{A,B,C})
    # =========================================================================
    _, state_123 = model.ingest_context(
        t2['ctx_states'], t2['ctx_actions'], t2['ctx_deltas'], state=state_12
    )
    state_123_q = model.clone_state(state_123)

    # Query all 3 physics modes simultaneously against frozen W_{A,B,C}
    q_states_all = torch.cat([t0['query_states'], t1['query_states'], t2['query_states']], dim=0)
    q_actions_all = torch.cat([t0['query_actions'], t1['query_actions'], t2['query_actions']], dim=0)
    pred_all_after_123 = model.query(q_states_all, q_actions_all, state_123_q)

    num_q = t0['query_states'].size(0)
    pred_t0_after_123 = pred_all_after_123[0:num_q]
    pred_t1_after_123 = pred_all_after_123[num_q:2 * num_q]
    pred_t2_after_123 = pred_all_after_123[2 * num_q:3 * num_q]

    loss_t0_after_123 = F.mse_loss(pred_t0_after_123, t0['query_deltas'])
    loss_t1_after_123 = F.mse_loss(pred_t1_after_123, t1['query_deltas'])
    loss_t2_after_123 = F.mse_loss(pred_t2_after_123, t2['query_deltas'])

    # Multi-step autoregressive rollout loss L_rollout on frozen W_{A,B,C}
    roll_actions_all = torch.cat(
        [t0['rollout_actions'], t1['rollout_actions'], t2['rollout_actions']], dim=0
    )
    roll_target_phys_all = torch.cat(
        [t0['rollout_target_phys'], t1['rollout_target_phys'], t2['rollout_target_phys']], dim=0
    )
    pred_phys_traj, _ = model.rollout(q_states_all, roll_actions_all, state_123_q)

    roll_loss_t0 = F.mse_loss(pred_phys_traj[0:num_q], roll_target_phys_all[0:num_q])
    roll_loss_t1 = F.mse_loss(pred_phys_traj[num_q:2 * num_q], roll_target_phys_all[num_q:2 * num_q])
    roll_loss_t2 = F.mse_loss(pred_phys_traj[2 * num_q:3 * num_q], roll_target_phys_all[2 * num_q:3 * num_q])

    if use_acl:
        loss_1step_final = loss_t0_after_123 + loss_t1_after_123 + loss_t2_after_123
        loss_rollout_final = roll_loss_t0 + roll_loss_t1 + roll_loss_t2
        loss_inter = loss_t0_after_1 + loss_t1_after_12 + loss_t0_after_12
    else:
        # Standard sequential training without backward-transfer supervision
        loss_1step_final = loss_t2_after_123
        loss_rollout_final = roll_loss_t2
        loss_inter = loss_t0_after_1 + loss_t1_after_12

    total_loss = (
        loss_1step_final
        + lambda_multistep * loss_rollout_final
        + alpha_intermediate * loss_inter
    )

    mean_1step_abc = (loss_t0_after_123.item() + loss_t1_after_123.item() + loss_t2_after_123.item()) / 3.0
    mean_rollout_abc = (roll_loss_t0.item() + roll_loss_t1.item() + roll_loss_t2.item()) / 3.0
    forgetting_ratio_t0 = loss_t0_after_123.item() / max(loss_t0_after_1.item(), 1e-8)

    metrics = {
        'total_loss': total_loss.item(),
        'mse_t0_after_1': loss_t0_after_1.item(),
        'mse_t1_after_12': loss_t1_after_12.item(),
        'mse_t0_after_12': loss_t0_after_12.item(),
        'mse_t0_after_123': loss_t0_after_123.item(),
        'mse_t1_after_123': loss_t1_after_123.item(),
        'mse_t2_after_123': loss_t2_after_123.item(),
        'mse_mean_after_123': mean_1step_abc,
        'rollout_mse_t0': roll_loss_t0.item(),
        'rollout_mse_t1': roll_loss_t1.item(),
        'rollout_mse_t2': roll_loss_t2.item(),
        'rollout_mse_mean_after_123': mean_rollout_abc,
        'forgetting_ratio_t0': forgetting_ratio_t0,
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate_model(
    model: ContinualWorldModel,
    dataset: ContinualPhysicsDataset,
    num_batches: int = 20,
    lambda_multistep: float = 0.5,
    alpha_intermediate: float = 0.5,
    use_acl: bool = True,
) -> Dict[str, float]:
    model.eval()
    agg: Dict[str, float] = {}
    for b_idx in range(num_batches):
        episode_batch = dataset.sample_episode_batch(step_idx=b_idx, fixed_order=(b_idx % 2 == 0))
        _, m = compute_acl_wm_loss(
            model,
            episode_batch,
            lambda_multistep=lambda_multistep,
            alpha_intermediate=alpha_intermediate,
            use_acl=use_acl,
        )
        for k, v in m.items():
            agg[k] = agg.get(k, 0.0) + v / float(num_batches)
    model.train()
    return agg


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    os.makedirs(args.work_dir, exist_ok=True)

    dataset = ContinualPhysicsDataset(
        batch_size=args.batch_size,
        k_shot=args.k_shot,
        num_queries=args.num_queries,
        rollout_horizon=args.rollout_horizon,
        device=device,
    )
    val_dataset = ContinualPhysicsDataset(
        batch_size=args.batch_size,
        k_shot=args.k_shot,
        num_queries=args.num_queries,
        rollout_horizon=args.rollout_horizon,
        device=device,
    )

    model = ContinualWorldModel(
        model_type=args.model_type,
        state_dim=dataset.physics.STATE_DIM,
        action_dim=dataset.physics.ACTION_DIM,
        delta_dim=dataset.physics.PHYS_DIM,
        hidden_size=args.hidden_size,
        num_layers=args.num_layer,
        num_head=args.n_head,
        ff_factor=args.ff_factor,
        dropout=args.dropout,
        use_ln=not args.not_use_ln,
        use_input_softmax=args.use_input_softmax,
        beta_init=args.srwm_beta_init,
        init_scaler=args.srwm_init_scaler,
        q_init_scaler=args.srwm_q_init_scaler,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_train_steps, eta_min=args.learning_rate * 0.05
    )

    print(
        f"[ACL-WM] Training model_type={args.model_type} ({model.canonical_type}) | "
        f"params={model.num_params():,} | device={device} | seed={args.seed} | "
        f"steps={args.total_train_steps}"
    )

    history = []
    best_val_mse = float('inf')
    best_ckpt_path = os.path.join(args.work_dir, 'best_model.pt')
    latest_ckpt_path = os.path.join(args.work_dir, 'latest_model.pt')

    start_time = time.time()
    running_metrics: Dict[str, float] = {}
    running_count = 0

    for step in range(1, args.total_train_steps + 1):
        model.train()
        # Linear warmup over first args.warmup_steps
        if step <= args.warmup_steps and args.warmup_steps > 0:
            lr_scale = float(step) / float(args.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate * lr_scale

        episode_batch = dataset.sample_episode_batch(step_idx=step)
        loss, step_metrics = compute_acl_wm_loss(
            model,
            episode_batch,
            lambda_multistep=args.lambda_multistep,
            alpha_intermediate=args.alpha_intermediate,
            use_acl=not args.disable_acl,
        )

        model.reset_grad()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if step > args.warmup_steps:
            scheduler.step()

        for k, v in step_metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v
        running_count += 1

        if step % args.report_every == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_m = {k: v / running_count for k, v in running_metrics.items()}
            print(
                f"Step {step:5d}/{args.total_train_steps} | "
                f"loss={avg_m['total_loss']:.5f} | "
                f"MSE(A|ABC)={avg_m['mse_t0_after_123']:.5f} "
                f"MSE(B|ABC)={avg_m['mse_t1_after_123']:.5f} "
                f"MSE(C|ABC)={avg_m['mse_t2_after_123']:.5f} | "
                f"RolloutMSE={avg_m['rollout_mse_mean_after_123']:.5f} | "
                f"ForgetRatio(A)={avg_m['forgetting_ratio_t0']:.3f} | "
                f"({elapsed:.1f}s)"
            )
            running_metrics = {}
            running_count = 0

        if step % args.validate_every == 0 or step == args.total_train_steps:
            set_seed(args.valid_seed)
            val_metrics = evaluate_model(
                model,
                val_dataset,
                num_batches=args.valid_batches,
                lambda_multistep=args.lambda_multistep,
                alpha_intermediate=args.alpha_intermediate,
                use_acl=not args.disable_acl,
            )
            # Restore training RNG state progression
            set_seed(args.seed + step * 101)

            val_record = {'step': step, **val_metrics}
            history.append(val_record)
            print(
                f"==> [VAL step={step}] "
                f"Mean MSE(ABC)={val_metrics['mse_mean_after_123']:.5f} "
                f"(Mode1={val_metrics['mse_t0_after_123']:.5f}, "
                f"Mode2={val_metrics['mse_t1_after_123']:.5f}, "
                f"Mode3={val_metrics['mse_t2_after_123']:.5f}) | "
                f"RolloutMSE={val_metrics['rollout_mse_mean_after_123']:.5f} | "
                f"ForgetRatio(Mode1)={val_metrics['forgetting_ratio_t0']:.3f}"
            )

            ckpt_payload = {
                'step': step,
                'args': vars(args),
                'model_type': args.model_type,
                'canonical_type': model.canonical_type,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }
            torch.save(ckpt_payload, latest_ckpt_path)
            if val_metrics['mse_mean_after_123'] < best_val_mse:
                best_val_mse = val_metrics['mse_mean_after_123']
                torch.save(ckpt_payload, best_ckpt_path)

            with open(os.path.join(args.work_dir, 'metrics.json'), 'w') as f:
                json.dump(
                    {
                        'args': vars(args),
                        'num_params': model.num_params(),
                        'best_val_mse': best_val_mse,
                        'history': history,
                    },
                    f,
                    indent=2,
                )

    return {
        'model_type': args.model_type,
        'seed': args.seed,
        'best_val_mse': best_val_mse,
        'final_val': history[-1] if history else {},
        'work_dir': args.work_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train In-Context Continual World Model (ACL-WM)')
    parser.add_argument(
        '--model_type',
        type=str,
        default='self_ref_v0',
        choices=list(MODEL_TYPE_ALIASES.keys()),
        help='Sequence model architecture',
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--valid_seed', type=int, default=999)
    parser.add_argument('--work_dir', type=str, default='save_models/acl_wm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k_shot', type=int, default=25, help='Transitions per physics mode in context')
    parser.add_argument('--num_queries', type=int, default=8, help='Query transitions per mode')
    parser.add_argument('--rollout_horizon', type=int, default=4, help='Multi-step rollout horizon')
    parser.add_argument('--num_layer', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--ff_factor', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--not_use_ln', action='store_true')
    parser.add_argument('--use_input_softmax', action='store_true', default=False)
    parser.add_argument('--srwm_beta_init', type=float, default=0.0)
    parser.add_argument('--srwm_init_scaler', type=float, default=1.0)
    parser.add_argument('--srwm_q_init_scaler', type=float, default=0.5)
    parser.add_argument('--lambda_multistep', type=float, default=0.5)
    parser.add_argument('--alpha_intermediate', type=float, default=0.5)
    parser.add_argument('--disable_acl', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--total_train_steps', type=int, default=3000)
    parser.add_argument('--report_every', type=int, default=100)
    parser.add_argument('--validate_every', type=int, default=250)
    parser.add_argument('--valid_batches', type=int, default=20)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--cpu', action='store_true')
    return parser


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)
