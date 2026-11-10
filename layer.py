import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_weight import fast_weight_delta, stateful_fast_weight_delta
from self_ref_v0 import self_ref_v0, stateful_self_ref_v0


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / (x.sum(-1, keepdim=True) + 1e-6)


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout, use_layernorm=True,
                 use_res=True):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm
        self.use_res = use_res

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, ff_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, res_dim),
            nn.Dropout(dropout),
        )

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        if self.use_res:
            out = self.ff_layers(out) + x
        else:
             out = self.ff_layers(out)
        return out


# Fast weight layer with feed-forward fast net (DeltaNet)
class FastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, stateful=False,
                 single_state_training=False):
        super(FastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.stateful = stateful
        self.single_state_training = single_state_training
        if stateful:
            self.fw_layer = stateful_fast_weight_delta
        else:
            self.fw_layer = fast_weight_delta

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None, get_state=False, read_only=False):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        head_q = elu_p1(head_q)
        head_k = elu_p1(head_k)

        # normalize k and q, crucial for stable training.
        head_k = sum_norm(head_k)
        head_q = sum_norm(head_q)

        if state is None:
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device, dtype=head_k.dtype)
        else:
            fast_weights = state

        if read_only:
            # beta_t = 0: fast weights are frozen, read out directly via matmul
            out = torch.matmul(head_q, fast_weights)
        elif self.stateful:
            out, fast_weights = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)
        else:
            out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        if get_state:
            assert fast_weights is not None
            if self.single_state_training:
                fast_weights = fast_weights.detach()[0].clone().unsqueeze(0).repeat(bsz, 1, 1, 1)
                return out, fast_weights
            return out, fast_weights
        else:
            return out


# Linear Transformer layer (fast_transformers: additive Hebbian outer-product fast weights)
class LinearTransformerFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, stateful=True):
        super(LinearTransformerFFlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.stateful = stateful

        self.slow_net = nn.Linear(in_dim, num_head * 3 * dim_head, bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None, get_state=False, read_only=False):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()
        out = self.layer_norm(x)

        qkv = self.slow_net(out).view(slen, bsz, self.num_head, 3 * self.dim_head)
        head_q, head_k, head_v = torch.split(qkv, (self.dim_head,) * 3, dim=-1)

        # (B, H, L, E)
        head_q = sum_norm(elu_p1(head_q.permute(1, 2, 0, 3)))
        head_k = sum_norm(elu_p1(head_k.permute(1, 2, 0, 3)))
        head_v = head_v.permute(1, 2, 0, 3)

        if state is None:
            W = torch.zeros(bsz, self.num_head, self.dim_head, self.dim_head,
                            device=x.device, dtype=x.dtype)
            z = torch.zeros(bsz, self.num_head, self.dim_head, 1,
                            device=x.device, dtype=x.dtype)
        else:
            W, z = state

        if read_only:
            # Query against frozen additive memory (W, z) without updating
            num = torch.matmul(head_q, W)  # (B, H, L, E)
            den = torch.matmul(head_q, z).clamp(min=1e-5)  # (B, H, L, 1)
            out = num / den
        else:
            # Causal cumulative outer products within the segment + prior state W, z
            # kv_step: (B, H, L, E, E)
            kv_step = torch.matmul(head_k.unsqueeze(-1), head_v.unsqueeze(-2))
            W_cum = W.unsqueeze(2) + torch.cumsum(kv_step, dim=2)
            z_cum = z.unsqueeze(2) + torch.cumsum(head_k.unsqueeze(-1), dim=2)
            num = torch.matmul(head_q.unsqueeze(-2), W_cum).squeeze(-2)
            den = torch.matmul(head_q.unsqueeze(-2), z_cum).squeeze(-2).clamp(min=1e-5)
            out = num / den
            W = W_cum[:, :, -1]
            z = z_cum[:, :, -1]

        out = out.transpose(1, 2).reshape(bsz, slen, self.num_head * self.dim_head).transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        if get_state:
            return out, (W, z)
        return out


# Standard Causal Transformer layer with bounded sliding-window KV-cache state
class CausalAttentionFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, stateful=True, max_cache_len=30):
        super(CausalAttentionFFlayer, self).__init__()
        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.stateful = stateful
        self.max_cache_len = max_cache_len
        self.scale = 1.0 / math.sqrt(dim_head)

        self.qkv_proj = nn.Linear(in_dim, num_head * 3 * dim_head, bias=False)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None, get_state=False, read_only=False):
        # x shape: (len, B, in_dim)
        slen, bsz, _ = x.size()
        h = self.layer_norm(x)
        qkv = self.qkv_proj(h).view(slen, bsz, self.num_head, 3 * self.dim_head)
        q, k, v = torch.split(qkv, (self.dim_head,) * 3, dim=-1)

        # (B, H, L, E)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        if read_only:
            # Queries attend over frozen context KV cache + each query's own token only
            if state is not None and state[0] is not None:
                k_ctx, v_ctx = state
                # Score against context: (B, H, L, L_ctx)
                scores_ctx = torch.matmul(q, k_ctx.transpose(-2, -1)) * self.scale
                # Score against self: (B, H, L, 1)
                scores_self = (q * k).sum(dim=-1, keepdim=True) * self.scale
                scores = torch.cat([scores_ctx, scores_self], dim=-1)
                attn = F.softmax(scores, dim=-1)
                attn = self.drop(attn)
                attn_ctx = attn[..., :-1]
                attn_self = attn[..., -1:]
                out = torch.matmul(attn_ctx, v_ctx) + attn_self * v
            else:
                out = v
            new_state = state
        else:
            if state is not None and state[0] is not None:
                k_prev, v_prev = state
                prev_len = k_prev.size(2)
                k_all = torch.cat([k_prev, k], dim=2)
                v_all = torch.cat([v_prev, v], dim=2)
            else:
                prev_len = 0
                k_all = k
                v_all = v

            total_len = k_all.size(2)
            scores = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale  # (B, H, L, total_len)
            causal_mask = torch.ones(slen, total_len, device=x.device, dtype=torch.bool).tril(diagonal=prev_len)
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.drop(attn)
            out = torch.matmul(attn, v_all)
            if k_all.size(2) > self.max_cache_len:
                k_all = k_all[:, :, -self.max_cache_len:].contiguous()
                v_all = v_all[:, :, -self.max_cache_len:].contiguous()
            new_state = (k_all, v_all)

        out = out.transpose(1, 2).reshape(bsz, slen, self.num_head * self.dim_head).transpose(0, 1)
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        if get_state:
            return out, new_state
        return out


# self referential weight matrix layer
class SRWMlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, use_ln=True,
                 use_input_softmax=False, beta_init=-1.0, use_res=True, stateful=False,
                 init_scaler=1., q_init_scaler=0.01, unif_init=False,
                 single_state_training=False, no_softmax_on_y=False):
        super(SRWMlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.use_ln = use_ln
        self.use_res = use_res
        self.use_input_softmax = use_input_softmax
        self.no_softmax_on_y = no_softmax_on_y
        if no_softmax_on_y:
            assert use_input_softmax, '`no_softmax_on_y` is True but not `use_input_softmax`'

        self.stateful = stateful
        self.single_state_training = single_state_training
        if no_softmax_on_y:
            from self_ref_v3 import self_ref_v3, stateful_self_ref_v3
            if stateful:
                self.sr_layer = stateful_self_ref_v3
            else:
                self.sr_layer = self_ref_v3
            self.y_lnorm = nn.LayerNorm(dim_head)
        else:
            if stateful:
                self.sr_layer = stateful_self_ref_v0
            else:
                self.sr_layer = self_ref_v0
        n_head = num_head
        d_head = dim_head

        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4),
                                requires_grad=True)
        if use_ln:
            self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        if unif_init:
            self.reset_parameters_unif(init_scaler, q_init_scaler)
        else:
            self.reset_parameters(beta_init, init_scaler, q_init_scaler)

    def reset_parameters(self, beta_init, init_scaler, q_init_scaler=0.01):
        std = init_scaler / math.sqrt(self.dim_head)
        # std = 0.1 / math.sqrt(self.dim_head)
        std_q = q_init_scaler / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0., std=std)
        # nn.init.normal_(self.W_q, mean=0., std=std)
        nn.init.normal_(self.W_q, mean=0., std=std_q)
        nn.init.normal_(self.W_k, mean=0., std=std)
        # tried -1 for beta but 0 seems to be better
        # nn.init.normal_(self.w_b, mean=-5., std=std)
        nn.init.normal_(self.w_b, mean=beta_init, std=std)

    def reset_parameters_unif(self, init_scaler, q_init_scaler=0.01):
        # beta_init not used
        nn.init.uniform_(self.W_y, a=-init_scaler, b=init_scaler)
        nn.init.uniform_(self.W_q, a=-q_init_scaler, b=q_init_scaler)
        nn.init.uniform_(self.W_k, a=-init_scaler, b=init_scaler)
        nn.init.uniform_(self.w_b, a=-init_scaler, b=init_scaler)

    def forward(self, h, state=None, get_state=False, read_only=False):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        if self.use_input_softmax:
            if self.no_softmax_on_y:
                x = F.softmax(x, dim=-1)
                input_to_y = x.clone()
            else:
                x = F.softmax(x, dim=-1)
        # reshape to (B, heads, len, dim)
        x = x.permute(1, 2, 0, 3)

        if state is not None:  # state stores the shift from the base weights.
            W_y_bc, W_q_bc, W_k_bc, w_b_bc = state
            W_y_bc = W_y_bc + self.W_y.clone().repeat(bsz, 1, 1, 1)
            W_q_bc = W_q_bc + self.W_q.clone().repeat(bsz, 1, 1, 1)
            W_k_bc = W_k_bc + self.W_k.clone().repeat(bsz, 1, 1, 1)
            w_b_bc = w_b_bc + self.w_b.clone().repeat(bsz, 1, 1, 1)
        else:
            W_y_bc = self.W_y.clone().repeat(bsz, 1, 1, 1)
            W_q_bc = self.W_q.clone().repeat(bsz, 1, 1, 1)
            W_k_bc = self.W_k.clone().repeat(bsz, 1, 1, 1)
            w_b_bc = self.w_b.clone().repeat(bsz, 1, 1, 1)

        if read_only:
            # beta_t = 0: fast weights are frozen, read out y_t = W_y^T phi(x_t) directly
            out = torch.matmul(x, W_y_bc)
            if self.no_softmax_on_y:
                out = self.y_lnorm(out)
        elif self.no_softmax_on_y:
            if self.stateful:
                out, W_y_bc, W_q_bc, W_k_bc, w_b_bc = self.sr_layer(x, input_to_y, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
            else:
                out = self.sr_layer(x, input_to_y, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
            out = self.y_lnorm(out)
        else:
            if self.stateful:
                out, W_y_bc, W_q_bc, W_k_bc, w_b_bc = self.sr_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
            else:
                out = self.sr_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        if self.use_ln:
            if self.use_res:
                out = self.layer_norm(h) + out
        else:
            if self.use_res:
                out = h + out

        if get_state:
            if self.single_state_training:  # take only batch one.
                W_y_bc = (W_y_bc[0].unsqueeze(0) - self.W_y.detach().clone()).repeat(bsz, 1, 1, 1)
                W_q_bc = (W_q_bc[0].unsqueeze(0) - self.W_q.detach().clone()).repeat(bsz, 1, 1, 1)
                W_k_bc = (W_k_bc[0].unsqueeze(0) - self.W_k.detach().clone()).repeat(bsz, 1, 1, 1)
                w_b_bc = (w_b_bc[0].unsqueeze(0) - self.w_b.detach().clone()).repeat(bsz, 1, 1, 1)
            else:
                W_y_bc = W_y_bc - self.W_y.detach().clone().repeat(bsz, 1, 1, 1)
                W_q_bc = W_q_bc - self.W_q.detach().clone().repeat(bsz, 1, 1, 1)
                W_k_bc = W_k_bc - self.W_k.detach().clone().repeat(bsz, 1, 1, 1)
                w_b_bc = w_b_bc - self.w_b.detach().clone().repeat(bsz, 1, 1, 1)

            state = (W_y_bc, W_q_bc, W_k_bc, w_b_bc)

            return out, state

        return out
