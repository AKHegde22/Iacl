"""
Vectorized Continual Physics Environment & Trajectory Generator for ACL-WM.

Replaces static Torchmeta image loaders (Omniglot / Mini-ImageNet) with on-the-fly
GPU-vectorized non-stationary dynamical systems across 3 sequential physics modes:
  - Mode A: Inverted Gravity + High Cart/Pole Friction & Damping
  - Mode B: Strong Lateral Cross-Wind Force + Low Pole Mass / Short Pole
  - Mode C: Inverted Action Motor Torque + Elastic Restoring Field

Each token t in the context window is an environment transition tuple (s_t, a_t, ds_t)
where:
  - q_t = [x_t, x_dot_t, sin(theta_t), cos(theta_t), theta_dot_t] in R^5
  - xi_t in R^3 is a continuous environmental sensory field observation drawn from an
    episode-random SO(3) orthonormal basis (so xi_t has zero fixed meaning across episodes
    and carries zero task-ID label, while preventing 1-step kinematic shortcuts).
  - s_t = [q_t (5D), xi_t (3D)] in R^8
  - a_t in [-1, 1]^1
  - ds_t = (q_{t+1} - q_t) / dt in R^5 (scaled state derivative so all coordinates have O(1) variance)
"""

import math
from typing import Dict, Optional, Tuple
import torch


class VectorizedCartPolePhysics:
    """
    GPU-vectorized non-stationary continuous CartPole / Pendulum dynamical system.

    Physical coordinates:
        q_t = [x, x_dot, sin(theta), cos(theta), theta_dot] in R^5
    Full observation state (task-ID-free, no 1-step delta shortcut):
        s_t = [q_t (5D), xi_t (3D)] in R^8
    Action:
        a_t in [-1, 1]^1 (scaled by motor force capacity F_max = 10.0 N)
    Target state delta:
        ds_t = (q_{t+1} - q_t) * DELTA_SCALE in R^5
    """

    PHYS_DIM = 5
    ENV_FIELD_DIM = 3
    STATE_DIM = 8
    ACTION_DIM = 1
    DELTA_SCALE = 10.0

    def __init__(self, dt: float = 0.04, rk_substeps: int = 2, force_mag: float = 10.0):
        self.dt = dt
        self.rk_substeps = rk_substeps
        self.force_mag = force_mag

    def sample_episode_field_basis(
        self,
        batch_size: int,
        device: torch.device,
        fixed: bool = False,
    ) -> torch.Tensor:
        """
        Sample a batch of random 3x3 orthogonal matrices Q in O(3) of shape (B, 3, 3).
        Columns Q[:, :, 0], Q[:, :, 1], Q[:, :, 2] serve as the episode-specific latent
        sensory field bases for the 3 sequential tasks in the episode. Because Q is
        haar-random in O(3) every episode, the field coordinates have zero fixed meaning
        across episodes (never a task-ID label).
        """
        if fixed:
            eye = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3).clone()
            return eye
        g = torch.randn(batch_size, 3, 3, device=device)
        q, r = torch.linalg.qr(g)
        diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        return q * diag_sign

    def sample_mode_params(
        self,
        batch_size: int,
        mode: str,
        device: torch.device,
        field_vec: Optional[torch.Tensor] = None,
        fixed: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample continuous physical parameters for Mode 'A', 'B', or 'C'.
        Directional forces and motor gains take random episode signs (+/-1) so that
        the prior expectation E[ds_t | s_t, a_t] across episodes is near zero; the world
        model must infer the active physics mode from the in-context transitions.
        """
        def unif(low: float, high: float) -> torch.Tensor:
            if fixed:
                return torch.full((batch_size, 1), 0.5 * (low + high), device=device)
            return torch.empty((batch_size, 1), device=device).uniform_(low, high)

        def rand_sign() -> torch.Tensor:
            if fixed:
                return torch.ones((batch_size, 1), device=device)
            return torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32) * 2.0 - 1.0

        if field_vec is None:
            basis = self.sample_episode_field_basis(batch_size, device, fixed=fixed)
            mode_idx = {'A': 0, 'B': 1, 'C': 2}.get(mode, 0)
            field_vec = basis[:, :, mode_idx]  # (B, 3)

        s_dir = rand_sign()
        s_gain = rand_sign()

        if mode == 'A':
            # Mode A: Inverted gravity + high cart & pole damping
            return {
                'gravity': unif(-11.5, -8.5),
                'masscart': unif(1.0, 1.25),
                'masspole': unif(0.12, 0.18),
                'length': unif(0.50, 0.65),
                'mu_cart': unif(0.70, 1.10),
                'mu_pole': unif(0.20, 0.35),
                'wind_force': s_dir * unif(1.5, 3.0),
                'motor_gain': unif(0.95, 1.20),
                'spring_k': torch.zeros((batch_size, 1), device=device),
                'field_vec': field_vec,
            }
        elif mode == 'B':
            # Mode B: Strong lateral cross-wind + low pole mass & short pole + low friction
            return {
                'gravity': unif(8.5, 11.5),
                'masscart': unif(0.85, 1.10),
                'masspole': unif(0.06, 0.10),
                'length': unif(0.35, 0.48),
                'mu_cart': unif(0.03, 0.08),
                'mu_pole': unif(0.02, 0.05),
                'wind_force': s_dir * unif(5.5, 8.0),
                'motor_gain': unif(0.95, 1.20),
                'spring_k': torch.zeros((batch_size, 1), device=device),
                'field_vec': field_vec,
            }
        elif mode == 'C':
            # Mode C: Inverted action motor torque + strong elastic restoring spring
            return {
                'gravity': unif(8.5, 11.5),
                'masscart': unif(0.95, 1.20),
                'masspole': unif(0.10, 0.15),
                'length': unif(0.45, 0.60),
                'mu_cart': unif(0.12, 0.25),
                'mu_pole': unif(0.05, 0.10),
                'wind_force': s_dir * unif(1.0, 2.5),
                'motor_gain': unif(-1.35, -0.95),
                'spring_k': unif(2.8, 4.5),
                'field_vec': field_vec,
            }
        else:
            raise ValueError(f"Unknown physics mode: {mode}")

    def _raw_derivatives(
        self,
        raw_state: torch.Tensor,
        action: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute continuous-time derivatives d/dt [x, x_dot, theta, theta_dot].
        raw_state: (..., 4) -> [x, x_dot, theta, theta_dot]
        action: (..., 1) -> motor command in [-1, 1]
        """
        x = raw_state[..., 0:1]
        x_dot = raw_state[..., 1:2]
        theta = raw_state[..., 2:3]
        theta_dot = raw_state[..., 3:4]

        g = params['gravity']
        mc = params['masscart']
        mp = params['masspole']
        total_mass = mc + mp
        l = params['length']
        polemass_length = mp * l
        mu_c = params['mu_cart']
        mu_p = params['mu_pole']
        wind = params['wind_force']
        gain = params['motor_gain']
        spring_k = params['spring_k']

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        # Total horizontal force on cart: motor + cross-wind - spring - cart damping
        force = (
            gain * self.force_mag * torch.clamp(action, -1.0, 1.0)
            + wind
            - spring_k * x
            - mu_c * x_dot
        )

        # Coupled Euler-Lagrange equations with pivot damping and wind torque on pole
        temp = (force + polemass_length * (theta_dot ** 2) * sin_t) / total_mass
        wind_torque = 0.25 * wind * cos_t
        theta_acc_num = (
            g * sin_t
            - cos_t * temp
            + wind_torque
            - mu_p * theta_dot
        )
        theta_acc_den = l * (4.0 / 3.0 - (mp * (cos_t ** 2)) / total_mass)
        theta_acc = theta_acc_num / (theta_acc_den + 1e-6)
        x_acc = temp - (polemass_length * theta_acc * cos_t) / total_mass

        return torch.cat([x_dot, x_acc, theta_dot, theta_acc], dim=-1)

    @staticmethod
    def raw_to_phys(raw_state: torch.Tensor) -> torch.Tensor:
        """Convert [x, x_dot, theta, theta_dot] (..., 4) -> [x, x_dot, sin(t), cos(t), t_dot] (..., 5)."""
        x = raw_state[..., 0:1]
        x_dot = raw_state[..., 1:2]
        theta = raw_state[..., 2:3]
        theta_dot = raw_state[..., 3:4]
        return torch.cat([x, x_dot, torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)

    @staticmethod
    def phys_to_raw(phys_state: torch.Tensor) -> torch.Tensor:
        """Convert [x, x_dot, sin(t), cos(t), t_dot] (..., 5) -> [x, x_dot, theta, theta_dot] (..., 4)."""
        x = phys_state[..., 0:1]
        x_dot = phys_state[..., 1:2]
        sin_t = phys_state[..., 2:3]
        cos_t = phys_state[..., 3:4]
        theta_dot = phys_state[..., 4:5]
        theta = torch.atan2(sin_t, cos_t)
        return torch.cat([x, x_dot, theta, theta_dot], dim=-1)

    @classmethod
    def compose_state(cls, q_phys: torch.Tensor, field_vec: torch.Tensor) -> torch.Tensor:
        """
        Combine physical state q_t in R^5 with sensory field vector xi_t in R^3
        (including subtle spatial coupling 0.05 * tanh(q_t[..., :3])) -> s_t in R^8.
        """
        xi = field_vec + 0.05 * torch.tanh(q_phys[..., :cls.ENV_FIELD_DIM])
        return torch.cat([q_phys, xi], dim=-1)

    def step_raw(
        self,
        raw_state: torch.Tensor,
        action: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Integrate raw_state [x, x_dot, theta, theta_dot] forward by dt using RK4."""
        h = self.dt / float(self.rk_substeps)
        s = raw_state
        for _ in range(self.rk_substeps):
            k1 = self._raw_derivatives(s, action, params)
            k2 = self._raw_derivatives(s + 0.5 * h * k1, action, params)
            k3 = self._raw_derivatives(s + 0.5 * h * k2, action, params)
            k4 = self._raw_derivatives(s + h * k3, action, params)
            s = s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            theta = torch.atan2(torch.sin(s[..., 2:3]), torch.cos(s[..., 2:3]))
            x = torch.clamp(s[..., 0:1], -4.0, 4.0)
            x_dot = torch.clamp(s[..., 1:2], -8.0, 8.0)
            theta_dot = torch.clamp(s[..., 3:4], -8.0, 8.0)
            s = torch.cat([x, x_dot, theta, theta_dot], dim=-1)
        return s

    def step_augmented(
        self,
        aug_state: torch.Tensor,
        action: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step state s_t = [q_t (5D), xi_t (3D)] in R^8.
        Returns:
            next_aug_state: s_{t+1} in R^8
            scaled_delta_q: ds_t = (q_{t+1} - q_t) * DELTA_SCALE in R^5
        """
        q_t = aug_state[..., :self.PHYS_DIM]
        raw_t = self.phys_to_raw(q_t)
        raw_next = self.step_raw(raw_t, action, params)
        q_next = self.raw_to_phys(raw_next)
        scaled_delta_q = (q_next - q_t) * self.DELTA_SCALE
        next_aug_state = self.compose_state(q_next, params['field_vec'])
        return next_aug_state, scaled_delta_q

    @classmethod
    def advance_state_from_pred(
        cls,
        aug_state: torch.Tensor,
        action: torch.Tensor,
        pred_scaled_delta_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given current state s_t = [q_t, xi_t], action a_t, and predicted scaled delta
        ds_t = (q_{t+1} - q_t) * DELTA_SCALE, construct next state s_{t+1} for
        autoregressive multi-step rollouts and CEM-MPC planning.
        """
        q_t = aug_state[..., :cls.PHYS_DIM]
        xi_prev = aug_state[..., cls.PHYS_DIM:]
        # Recover underlying episode field vector u_m by subtracting previous spatial coupling
        field_vec = xi_prev - 0.05 * torch.tanh(q_t[..., :cls.ENV_FIELD_DIM])

        q_next_unnorm = q_t + (pred_scaled_delta_q / cls.DELTA_SCALE)
        sin_t = q_next_unnorm[..., 2:3]
        cos_t = q_next_unnorm[..., 3:4]
        norm = torch.sqrt(sin_t ** 2 + cos_t ** 2 + 1e-8)
        q_next = torch.cat([
            torch.clamp(q_next_unnorm[..., 0:1], -4.0, 4.0),
            torch.clamp(q_next_unnorm[..., 1:2], -8.0, 8.0),
            sin_t / norm,
            cos_t / norm,
            torch.clamp(q_next_unnorm[..., 4:5], -8.0, 8.0),
        ], dim=-1)
        return cls.compose_state(q_next, field_vec)

    @classmethod
    def compute_mpc_reward(
        cls,
        aug_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dense control reward for upright pole stabilization & cart centering in CEM-MPC.
        aug_state: (..., 8), action: (..., 1) -> returns reward (...,)
        """
        x = aug_state[..., 0]
        x_dot = aug_state[..., 1]
        sin_t = aug_state[..., 2]
        cos_t = aug_state[..., 3]
        theta_dot = aug_state[..., 4]
        act = action[..., 0]

        upright_bonus = cos_t - 0.5 * (sin_t ** 2)
        pos_penalty = 0.25 * (x ** 2)
        vel_penalty = 0.05 * (x_dot ** 2) + 0.05 * (theta_dot ** 2)
        act_penalty = 0.02 * (act ** 2)
        return upright_bonus - pos_penalty - vel_penalty - act_penalty

    def sample_initial_augmented_state(
        self,
        batch_size: int,
        params: Dict[str, torch.Tensor],
        device: torch.device,
        near_upright: bool = False,
    ) -> torch.Tensor:
        """
        Sample initial physical states q_0 and combine with the continuous sensory field
        reading xi_0 to form s_0 = [q_0, xi_0] in R^8.
        """
        if near_upright:
            x = torch.empty((batch_size, 1), device=device).uniform_(-0.25, 0.25)
            x_dot = torch.empty((batch_size, 1), device=device).uniform_(-0.35, 0.35)
            theta = torch.empty((batch_size, 1), device=device).uniform_(-0.20, 0.20)
            theta_dot = torch.empty((batch_size, 1), device=device).uniform_(-0.35, 0.35)
        else:
            x = torch.empty((batch_size, 1), device=device).uniform_(-1.2, 1.2)
            x_dot = torch.empty((batch_size, 1), device=device).uniform_(-2.0, 2.0)
            theta = torch.empty((batch_size, 1), device=device).uniform_(-math.pi, math.pi)
            theta_dot = torch.empty((batch_size, 1), device=device).uniform_(-2.5, 2.5)

        raw_0 = torch.cat([x, x_dot, theta, theta_dot], dim=-1)
        q_0 = self.raw_to_phys(raw_0)
        return self.compose_state(q_0, params['field_vec'])

    def sample_trajectory_segment(
        self,
        batch_size: int,
        seq_len: int,
        params: Dict[str, torch.Tensor],
        device: torch.device,
        init_aug_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of trajectories of length `seq_len` under `params`.
        Rolls out short multi-step sub-trajectories (length 5) with exploratory actions,
        resetting to diverse initial states every 5 steps to cover the state-action manifold.

        Returns:
            states: (seq_len, batch_size, 8)
            actions: (seq_len, batch_size, 1)
            deltas: (seq_len, batch_size, 5)
        """
        states_list = []
        actions_list = []
        deltas_list = []

        sub_traj_len = 5
        aug_state = init_aug_state
        for t in range(seq_len):
            if aug_state is None or (t > 0 and t % sub_traj_len == 0 and init_aug_state is None):
                aug_state = self.sample_initial_augmented_state(batch_size, params, device)
            action = torch.empty((batch_size, 1), device=device).uniform_(-1.0, 1.0)
            next_aug_state, delta_q = self.step_augmented(aug_state, action, params)
            states_list.append(aug_state)
            actions_list.append(action)
            deltas_list.append(delta_q)
            aug_state = next_aug_state

        return (
            torch.stack(states_list, dim=0),
            torch.stack(actions_list, dim=0),
            torch.stack(deltas_list, dim=0),
        )

    def sample_rollout_queries(
        self,
        batch_size: int,
        num_queries: int,
        rollout_horizon: int,
        params: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample `num_queries` independent starting states and roll them out for
        `rollout_horizon` steps under `params` to evaluate both 1-step query MSE
        and multi-step rollout loss L_rollout.

        Returns dict with:
            query_states: (num_queries, batch_size, 8)
            query_actions: (num_queries, batch_size, 1)
            query_deltas: (num_queries, batch_size, 5)
            rollout_actions: (num_queries, rollout_horizon, batch_size, 1)
            rollout_target_phys: (num_queries, rollout_horizon, batch_size, 5)
            rollout_target_deltas: (num_queries, rollout_horizon, batch_size, 5)
        """
        total_b = num_queries * batch_size
        exp_params = {
            k: v.unsqueeze(0).expand(num_queries, batch_size, -1).reshape(total_b, -1)
            for k, v in params.items()
        }

        aug_state = self.sample_initial_augmented_state(total_b, exp_params, device)
        q0_state = aug_state.reshape(num_queries, batch_size, self.STATE_DIM)

        roll_actions = []
        roll_phys = []
        roll_deltas = []

        cur_state = aug_state
        for _ in range(rollout_horizon):
            act = torch.empty((total_b, 1), device=device).uniform_(-1.0, 1.0)
            next_state, delta_q = self.step_augmented(cur_state, act, exp_params)
            roll_actions.append(act.reshape(num_queries, batch_size, self.ACTION_DIM))
            roll_phys.append(next_state[..., :self.PHYS_DIM].reshape(num_queries, batch_size, self.PHYS_DIM))
            roll_deltas.append(delta_q.reshape(num_queries, batch_size, self.PHYS_DIM))
            cur_state = next_state

        roll_actions_t = torch.stack(roll_actions, dim=1)  # (Q, H_roll, B, 1)
        roll_phys_t = torch.stack(roll_phys, dim=1)        # (Q, H_roll, B, 5)
        roll_deltas_t = torch.stack(roll_deltas, dim=1)    # (Q, H_roll, B, 5)

        return {
            'query_states': q0_state,
            'query_actions': roll_actions_t[:, 0],
            'query_deltas': roll_deltas_t[:, 0],
            'rollout_actions': roll_actions_t,
            'rollout_target_phys': roll_phys_t,
            'rollout_target_deltas': roll_deltas_t,
        }


class ContinualPhysicsDataset:
    """
    On-the-fly GPU/CPU generator for 3-task sequential Continual World Modeling (A -> B -> C).
    """

    MODES = ('A', 'B', 'C')
    PERMUTATIONS = (
        ('A', 'B', 'C'),
        ('B', 'C', 'A'),
        ('C', 'A', 'B'),
        ('A', 'C', 'B'),
        ('B', 'A', 'C'),
        ('C', 'B', 'A'),
    )

    def __init__(
        self,
        batch_size: int = 32,
        k_shot: int = 25,
        num_queries: int = 8,
        rollout_horizon: int = 4,
        device: torch.device = torch.device('cpu'),
        dt: float = 0.04,
    ):
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.num_queries = num_queries
        self.rollout_horizon = rollout_horizon
        self.device = device
        self.physics = VectorizedCartPolePhysics(dt=dt)

    def sample_episode_batch(
        self,
        step_idx: int = 0,
        fixed_order: bool = False,
        fixed_params: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate a batch of 3 sequential physics tasks (Task 0 -> Task 1 -> Task 2)
        along with query and multi-step rollout sets for all 3 tasks.
        """
        if fixed_order:
            mode_order = ('A', 'B', 'C')
        else:
            mode_order = self.PERMUTATIONS[step_idx % len(self.PERMUTATIONS)]

        basis = self.physics.sample_episode_field_basis(
            self.batch_size, self.device, fixed=fixed_params
        )

        episode_data: Dict[str, Dict[str, torch.Tensor]] = {}
        for task_idx, mode_name in enumerate(mode_order):
            field_vec = basis[:, :, task_idx]
            params = self.physics.sample_mode_params(
                self.batch_size, mode_name, self.device, field_vec=field_vec, fixed=fixed_params
            )
            ctx_s, ctx_a, ctx_ds = self.physics.sample_trajectory_segment(
                self.batch_size, self.k_shot, params, self.device
            )
            queries = self.physics.sample_rollout_queries(
                self.batch_size, self.num_queries, self.rollout_horizon, params, self.device
            )
            episode_data[f'task_{task_idx}'] = {
                'mode_name': mode_name,
                'params': params,
                'ctx_states': ctx_s,      # (K, B, 8)
                'ctx_actions': ctx_a,     # (K, B, 1)
                'ctx_deltas': ctx_ds,     # (K, B, 5)
                **queries,
            }

        return episode_data


if __name__ == '__main__':
    torch.manual_seed(42)
    ds = ContinualPhysicsDataset(batch_size=8, k_shot=20, num_queries=4, rollout_horizon=4)
    batch = ds.sample_episode_batch(step_idx=0)
    for k, v in batch.items():
        print(
            f"{k} (Mode {v['mode_name']}): "
            f"ctx_states={tuple(v['ctx_states'].shape)}, "
            f"ctx_actions={tuple(v['ctx_actions'].shape)}, "
            f"ctx_deltas={tuple(v['ctx_deltas'].shape)}, "
            f"deltas_std={v['ctx_deltas'].std().item():.3f}, "
            f"query_states={tuple(v['query_states'].shape)}, "
            f"rollout_target_phys={tuple(v['rollout_target_phys'].shape)}"
        )
