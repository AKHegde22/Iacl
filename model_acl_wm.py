"""
In-Context Continual World Models (ACL-WM) built on IDSIA/automated-cl.

Supports the 4-architecture ablation sweep with identical parameter scale
(num_layers=4, hidden_size=256, n_head=16, d_head=16, dim_ff=512):
  1. 'self_ref_v0' ('srwm', 'compat_stateful_srwm'): Self-Referential Weight Matrix (SRWM)
  2. 'fast_weight' ('deltanet', 'stateful_deltanet'): DeltaNet (Standard Fast Weights)
  3. 'fast_transformers' ('linear_transformer'): Linear Transformer (Additive Hebbian Fast Weights)
  4. 'transformer' / 'lstm': Standard Causal Transformer (KV-Cache) / Multi-layer LSTM
"""

from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn

from layer import (
    CausalAttentionFFlayer,
    FastFFlayer,
    LinearTransformerFFlayer,
    SRWMlayer,
    TransformerFFlayers,
)
from continual_physics_dataset import VectorizedCartPolePhysics


MODEL_TYPE_ALIASES = {
    'self_ref_v0': 'self_ref_v0',
    'srwm': 'self_ref_v0',
    'compat_stateful_srwm': 'self_ref_v0',
    'fast_weight': 'fast_weight',
    'deltanet': 'fast_weight',
    'stateful_deltanet': 'fast_weight',
    'fast_transformers': 'fast_transformers',
    'linear_transformer': 'fast_transformers',
    'transformer': 'transformer',
    'causal_transformer': 'transformer',
    'lstm': 'lstm',
}


class LSTMSublayer(nn.Module):
    """Residual LSTM sublayer matching the stateful interface of fast-weight layers."""

    def __init__(self, in_dim: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(in_dim, in_dim, num_layers=1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        get_state: bool = False,
        read_only: bool = False,
    ):
        # x: (slen, B, D)
        slen, bsz, dim = x.size()
        h_in = self.layer_norm(x)
        if read_only:
            # Evaluate each query token independently from frozen (h_t, c_t) without advancing state
            if state is None:
                h0 = torch.zeros(1, bsz * slen, dim, device=x.device, dtype=x.dtype)
                c0 = torch.zeros(1, bsz * slen, dim, device=x.device, dtype=x.dtype)
            else:
                h_prev, c_prev = state  # (1, B, D)
                h0 = h_prev.unsqueeze(1).expand(1, slen, bsz, dim).reshape(1, slen * bsz, dim).contiguous()
                c0 = c_prev.unsqueeze(1).expand(1, slen, bsz, dim).reshape(1, slen * bsz, dim).contiguous()
            q_flat = h_in.reshape(1, slen * bsz, dim)
            out_flat, _ = self.lstm(q_flat, (h0, c0))
            out = out_flat.reshape(slen, bsz, dim)
            new_state = state
        else:
            out, new_state = self.lstm(h_in, state)

        out = x + self.drop(out)
        if get_state:
            return out, new_state
        return out


class ContinualWorldModel(nn.Module):
    """
    Continual World Model processing sequential environment transitions
    (s_t, a_t, ds_t) in-context and answering frozen-weight state-action queries
    (s_q, a_q, 0) -> pred_ds_q across all previously encountered physics regimes.
    """

    def __init__(
        self,
        model_type: str = 'self_ref_v0',
        state_dim: int = VectorizedCartPolePhysics.STATE_DIM,
        action_dim: int = VectorizedCartPolePhysics.ACTION_DIM,
        delta_dim: int = VectorizedCartPolePhysics.PHYS_DIM,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_head: int = 16,
        ff_factor: int = 2,
        dropout: float = 0.05,
        use_ln: bool = True,
        use_input_softmax: bool = False,
        beta_init: float = 0.0,
        init_scaler: float = 1.0,
        q_init_scaler: float = 0.5,
    ):
        super().__init__()
        if model_type not in MODEL_TYPE_ALIASES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Choose from {list(MODEL_TYPE_ALIASES.keys())}"
            )
        self.canonical_type = MODEL_TYPE_ALIASES[model_type]
        self.model_type = model_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.delta_dim = delta_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        assert hidden_size % num_head == 0, "hidden_size must be divisible by num_head"
        self.dim_head = hidden_size // num_head
        self.dim_ff = hidden_size * ff_factor

        # Transition Embedding MLP: maps [s_t, a_t, ds_t] -> x_t in R^{d_model}
        in_features = state_dim + action_dim + delta_dim
        self.input_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Stack of L=4 sequence-mixing / fast-weight layers + residual FF blocks
        seq_layers = []
        ff_layers = []
        for _ in range(num_layers):
            if self.canonical_type == 'self_ref_v0':
                seq_layers.append(
                    SRWMlayer(
                        num_head=num_head,
                        dim_head=self.dim_head,
                        in_dim=hidden_size,
                        dropout=dropout,
                        use_ln=use_ln,
                        use_input_softmax=use_input_softmax,
                        beta_init=beta_init,
                        stateful=True,
                        init_scaler=init_scaler,
                        q_init_scaler=q_init_scaler,
                    )
                )
            elif self.canonical_type == 'fast_weight':
                seq_layers.append(
                    FastFFlayer(
                        num_head=num_head,
                        dim_head=self.dim_head,
                        in_dim=hidden_size,
                        dropout=dropout,
                        stateful=True,
                    )
                )
            elif self.canonical_type == 'fast_transformers':
                seq_layers.append(
                    LinearTransformerFFlayer(
                        num_head=num_head,
                        dim_head=self.dim_head,
                        in_dim=hidden_size,
                        dropout=dropout,
                        stateful=True,
                    )
                )
            elif self.canonical_type == 'transformer':
                seq_layers.append(
                    CausalAttentionFFlayer(
                        num_head=num_head,
                        dim_head=self.dim_head,
                        in_dim=hidden_size,
                        dropout=dropout,
                        stateful=True,
                    )
                )
            elif self.canonical_type == 'lstm':
                seq_layers.append(LSTMSublayer(in_dim=hidden_size, dropout=dropout))

            ff_layers.append(
                TransformerFFlayers(
                    ff_dim=self.dim_ff,
                    res_dim=hidden_size,
                    dropout=dropout,
                    use_layernorm=use_ln,
                )
            )

        self.seq_layers = nn.ModuleList(seq_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

        # State-delta prediction readout head: maps y_t -> hat{Delta s}_t
        self.out_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, delta_dim),
        )

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        for p in self.parameters():
            p.grad = None

    def clone_state(self, state: Any, detach: bool = False) -> Any:
        if state is None:
            return None

        def _clone_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.detach().clone() if detach else t.clone()

        cloned_layers = []
        for layer_state in state:
            if isinstance(layer_state, tuple):
                cloned_layers.append(tuple(_clone_tensor(x) for x in layer_state))
            else:
                cloned_layers.append(_clone_tensor(layer_state))
        return tuple(cloned_layers)

    def expand_state_for_batch(self, state: Any, repeat_factor: int) -> Any:
        """
        Repeat a batch=1 (or batch=B) fast-weight state across `repeat_factor`
        (used in CEM-MPC to evaluate N_pop candidate action sequences in parallel).
        """
        if state is None:
            return None

        def _expand_tensor(t: Optional[torch.Tensor], is_lstm: bool = False) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if is_lstm:
                # (1, B, D) -> (1, repeat_factor * B, D)
                return t.repeat(1, repeat_factor, 1)
            # Fast weight / KV tensors have batch on dim 0: (B, H, ...)
            reps = [repeat_factor] + [1] * (t.dim() - 1)
            return t.repeat(*reps)

        is_lstm = self.canonical_type == 'lstm'
        expanded_layers = []
        for layer_state in state:
            if isinstance(layer_state, tuple):
                expanded_layers.append(tuple(_expand_tensor(x, is_lstm=is_lstm) for x in layer_state))
            else:
                expanded_layers.append(_expand_tensor(layer_state, is_lstm=is_lstm))
        return tuple(expanded_layers)

    def ingest_context(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deltas: torch.Tensor,
        state: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Process a sequence of K context transitions (s_t, a_t, ds_t) and update
        fast-weight memory state W_{t-1} -> W_t.

        Args:
            states: (K, B, state_dim)
            actions: (K, B, action_dim)
            deltas: (K, B, delta_dim)
            state: optional prior fast-weight state tuple across layers
        Returns:
            out_preds: (K, B, delta_dim)
            new_state: tuple of updated fast-weight states per layer
        """
        token_in = torch.cat([states, actions, deltas], dim=-1)
        h = self.input_mlp(token_in)

        new_state_list: List[Any] = []
        for i in range(self.num_layers):
            layer_prior = state[i] if state is not None else None
            h, layer_state = self.seq_layers[i](
                h, state=layer_prior, get_state=True, read_only=False
            )
            h = self.ff_layers[i](h)
            new_state_list.append(layer_state)

        out_preds = self.out_head(h)
        return out_preds, tuple(new_state_list)

    def query(
        self,
        query_states: torch.Tensor,
        query_actions: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        """
        Query the frozen fast-weight matrix (beta_t = 0) with x_query = MLP([s_q, a_q, 0]).
        Does not mutate `state`, and evaluates all Q queries in parallel.

        Args:
            query_states: (Q, B, state_dim)
            query_actions: (Q, B, action_dim)
            state: fast-weight state tuple from `ingest_context`
        Returns:
            pred_deltas: (Q, B, delta_dim)
        """
        q_len, bsz, _ = query_states.shape
        zero_deltas = torch.zeros(
            q_len, bsz, self.delta_dim, device=query_states.device, dtype=query_states.dtype
        )
        token_q = torch.cat([query_states, query_actions, zero_deltas], dim=-1)
        h = self.input_mlp(token_q)

        for i in range(self.num_layers):
            layer_state = state[i] if state is not None else None
            h = self.seq_layers[i](
                h, state=layer_state, get_state=False, read_only=True
            )
            h = self.ff_layers[i](h)

        return self.out_head(h)

    def rollout(
        self,
        init_states: torch.Tensor,
        rollout_actions: torch.Tensor,
        state: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively unroll the world model over H_roll steps with frozen fast weights
        (beta_t = 0) for multi-step rollout loss L_rollout and CEM-MPC planning.

        Args:
            init_states: (Q, B, state_dim)
            rollout_actions: (Q, H_roll, B, action_dim)
            state: frozen fast-weight state tuple
        Returns:
            pred_phys_traj: (Q, H_roll, B, phys_dim)
            pred_delta_traj: (Q, H_roll, B, delta_dim)
        """
        horizon = rollout_actions.size(1)
        cur_states = init_states
        pred_phys_list = []
        pred_delta_list = []

        for h in range(horizon):
            act_h = rollout_actions[:, h]  # (Q, B, action_dim)
            pred_delta = self.query(cur_states, act_h, state)  # (Q, B, delta_dim)
            next_states = VectorizedCartPolePhysics.advance_state_from_pred(
                cur_states, act_h, pred_delta
            )
            pred_phys_list.append(next_states[..., :VectorizedCartPolePhysics.PHYS_DIM])
            pred_delta_list.append(pred_delta)
            cur_states = next_states

        return (
            torch.stack(pred_phys_list, dim=1),
            torch.stack(pred_delta_list, dim=1),
        )
